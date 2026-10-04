"""Submit an HTTP job, poll without model calls, and download its plot."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid
import httpx


def run(query, top_k=500, server='http://127.0.0.1:8010', output='demo-output/server-job', timeout=660):
    token = os.environ.get('VECTOR_QUERY_TOKEN')
    if not token:
        raise ValueError('Set VECTOR_QUERY_TOKEN')
    with httpx.Client(base_url=server, headers={'Authorization':f'Bearer {token}'}, timeout=30) as client:
        response = client.post('/jobs', json={'query':query,'top_k':top_k}, headers={'Idempotency-Key':uuid.uuid4().hex})
        response.raise_for_status()
        job = response.json()
        job_id = job['job_id']
        print(f'Submitted {job_id}; clustering runs on {server}', flush=True)
        deadline = time.monotonic()+timeout
        while job['status'] not in {'succeeded','failed','cancelled'}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Job still pending; resume GET /jobs/{job_id}')
            time.sleep(1)
            response = client.get(f'/jobs/{job_id}')
            response.raise_for_status()
            job = response.json()
        if job['status'] != 'succeeded':
            raise RuntimeError(job)
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact, filename in {'plot':'clusters.html','report':'report.md','results':'results.json','assignments':'assignments.csv'}.items():
            response = client.get(job['artifacts'][artifact])
            response.raise_for_status()
            (folder/filename).write_bytes(response.content)
        report_file = folder/'report.md'
        report_file.write_text(report_file.read_text().replace('banking77_dbscan_3d.html', 'clusters.html'))
        (folder/'job.json').write_text(json.dumps(job, indent=2))
        return {'job_id':job_id, 'status':job['status'], 'summary':job['summary'], 'plot':str((folder/'clusters.html').resolve())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--top-k', type=int, default=500)
    parser.add_argument('--server', default=os.getenv('VECTOR_QUERY_URL','http://127.0.0.1:8010'))
    parser.add_argument('--output', default='demo-output/server-job')
    args=parser.parse_args()
    print(json.dumps(run(args.query,args.top_k,args.server,args.output),indent=2))


if __name__=='__main__':
    main()
