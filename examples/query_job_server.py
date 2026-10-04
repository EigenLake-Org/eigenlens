"""HTTP query jobs: durable queue, process workers, Qdrant search and clustering."""
import argparse
import asyncio
from contextlib import asynccontextmanager, closing
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
import time
import uuid
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route
from vector_tools.artifacts import Store, now
from vector_tools.config import Config
from vector_tools.datasets import canonical
from vector_tools.jobs import Supervisor
from vector_tools.schemas import VTError
from vector_tools.service import Service

FILES = {'plot': 'banking77_dbscan_3d.html', 'report': 'query_report.md',
         'results': 'query_results.json', 'assignments': 'banking77_clustered.csv'}


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=500, ge=3, le=2000)


def query_worker(root, config_path, job_id, owner):
    parent = os.getppid()
    def watch_parent():
        while True:
            time.sleep(.25)
            if os.getppid() != parent:
                os._exit(70)
    threading.Thread(target=watch_parent, daemon=True).start()
    store = Store(root)
    try:
        from qdrant_client import QdrantClient
        from examples.banking77_qdrant import query_and_plot
        config = Config(config_path)
        job = store.job(owner, job_id)
        store.update_job(job_id, 'running')
        manifest = json.loads(Path(config.raw['query_source']['manifest']).read_text())
        source = config.raw['query_source']
        output = Path(root) / 'plots' / job_id
        with closing(QdrantClient(url=source['url'], api_key=os.getenv('QDRANT_API_KEY'), timeout=120)) as client:
            report = query_and_plot(client, source['collection'], manifest,
                                    job['payload']['query'], job['payload']['top_k'], output, config.limits)
        summary = {k:v for k,v in report.items() if k not in {'hits','plot'}}
        ref = store.put(owner, 'query', summary)
        store.update_job(job_id, 'succeeded', {'summary_id':ref})
    except BaseException as exc:
        store.update_job(job_id, 'failed', error=f'{type(exc).__name__}: query execution failed; inspect server logs')
        import traceback
        traceback.print_exc()


class QueryJobs(Service):
    """Reuse the existing durable store, process supervisor, status and cancellation."""
    def __init__(self, config, worker_target=query_worker):
        self.config = config
        self.store = Store(config.root)
        self.supervisor = Supervisor(config, worker_target=worker_target)

    def submit(self, request, key=None):
        if key is not None and not 1 <= len(key) <= 128:
            raise VTError('Idempotency-Key must contain 1..128 characters')
        payload = request.model_dump()
        digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT id, request_hash FROM jobs WHERE owner=? AND idempotency_key=?', ('local', key)).fetchone() if key else None
            if old:
                if old['request_hash'] != digest:
                    raise VTError('Idempotency key already used for a different request')
                return self.get_job('local', old['id'])
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= self.config.limits.max_pending_jobs:
                raise VTError('Job queue is full; retry later')
            job_id = 'job_' + uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,owner,status,payload,created_at,updated_at,idempotency_key,request_hash) VALUES(?,?,?,?,?,?,?,?)',
                       (job_id, 'local', 'queued', canonical(payload), now(), now(), key, digest))
        return self.get_job('local', job_id)


def create_app(config_path='config/query-jobs.json', token=None, worker_target=query_worker):
    token = token or os.getenv('VECTOR_QUERY_TOKEN')
    if not token:
        raise ValueError('Set VECTOR_QUERY_TOKEN; API authentication is required')

    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = QueryJobs(Config(str(Path(config_path).resolve())), worker_target)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.jobs.close)

    async def health(request):
        return JSONResponse({'status':'ok', 'service':'vector-query-jobs'})

    def authorized(request):
        return hmac.compare_digest(request.headers.get('authorization',''), 'Bearer ' + token)

    async def endpoint(request: Request):
        if not authorized(request):
            return JSONResponse({'error':'Bearer token required'}, status_code=401)
        service = request.app.state.jobs
        job_id = request.path_params.get('job_id')
        try:
            if request.method == 'POST' and job_id is None:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 16384:
                        return JSONResponse({'error':'Request too large'}, status_code=413)
                try:
                    spec = QueryRequest.model_validate_json(body)
                except ValidationError:
                    return JSONResponse({'error':'Expected query (1–2000 characters) and top_k (3–2000); no extra fields'}, status_code=422)
                job = await asyncio.to_thread(service.submit, spec, request.headers.get('idempotency-key'))
                job['status_url'] = f"/jobs/{job['job_id']}"
                return JSONResponse(job, status_code=202)
            if request.method == 'DELETE':
                return JSONResponse(await asyncio.to_thread(service.cancel_job, 'local', job_id))
            job = await asyncio.to_thread(service.get_job, 'local', job_id)
            artifact = request.path_params.get('artifact')
            if artifact:
                if job['status'] != 'succeeded':
                    return JSONResponse({'error':'Job has no completed artifacts', 'status':job['status']}, status_code=409)
                if artifact not in FILES:
                    return JSONResponse({'error':'Unknown artifact'}, status_code=404)
                path = service.config.root / 'plots' / job_id / FILES[artifact]
                if not path.is_file():
                    return JSONResponse({'error':'Artifact unavailable'}, status_code=404)
                return FileResponse(path, media_type='text/html' if artifact == 'plot' else None,
                                    headers={'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff'})
            if job['status'] == 'succeeded':
                job['summary'] = await asyncio.to_thread(service.store.get, 'local', job['results']['summary_id'], 'query')
                job['artifacts'] = {key:f'/jobs/{job_id}/{key}' for key in FILES}
            return JSONResponse(job)
        except VTError as exc:
            message = str(exc)
            code = 404 if 'unavailable' in message else 429 if 'queue is full' in message else 409
            return JSONResponse({'error':message}, status_code=code)

    return Starlette(lifespan=lifespan, routes=[Route('/health', health),
        Route('/jobs', endpoint, methods=['POST']),
        Route('/jobs/{job_id}', endpoint, methods=['GET','DELETE']),
        Route('/jobs/{job_id}/{artifact}', endpoint, methods=['GET'])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/query-jobs.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8010)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
