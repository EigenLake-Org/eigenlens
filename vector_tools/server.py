"""Official MCP SDK server: stdio or authenticated Streamable HTTP."""
from __future__ import annotations
import argparse
import asyncio
import contextvars
import hmac
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Literal
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse
from .config import Config
from .service import Service
from .schemas import Record, Selection, Operation, Execution, VTError

principal = contextvars.ContextVar("principal", default="local")

def build_server(service, host="127.0.0.1", port=8000):
    server = FastMCP("Vector ML", host=host, port=port, stateless_http=True, json_response=True,
                     max_request_body_size=2 * 1024 * 1024,
                     instructions="Inspect capabilities and source first. Create a dataset, run async analysis, poll job status, then inspect summaries and examples. Counts are vector records, not deduplicated tickets. Themes are evidence for investigation, not confirmed causes.",
                     transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                        allowed_hosts=["127.0.0.1:*", "localhost:*", "service:*", "chat-worker:*", "testserver"],
                        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://service:*", "http://chat-worker:*"]))
    server._mcp_server.version = "0.1.0"
    async def invoke(method, *args):
        try:
            return await asyncio.to_thread(getattr(service, method), principal.get(), *args)
        except VTError:
            raise
        except Exception as exc:
            # Driver messages can expose DSNs; never return them verbatim.
            raise VTError(f"{type(exc).__name__}: request failed; check source configuration and service diagnostics") from None
    @server.tool()
    async def get_capabilities() -> dict:
        """List authorized source IDs, algorithms, model availability, compute backends and limits."""
        return await invoke("get_capabilities")
    @server.tool()
    async def describe_source(source_id: str) -> dict:
        """Inspect configured mappings, tenant/namespace, capabilities and consistency before selecting."""
        return await invoke("describe_source", source_id)
    @server.tool()
    async def create_dataset(source_id: str, selection: Selection) -> dict:
        """Create a lazy dataset from existing IDs, AND filters, or deterministic sample. Never send embeddings to the model."""
        return await invoke("create_dataset", source_id, selection.model_dump())
    @server.tool()
    async def register_records(records: list[Record] | None = None, artifact_name: str | None = None, selection_scope: Literal["retrieval_results", "complete_selection"] = "retrieval_results") -> dict:
        """Register bounded caller records OR an application-staged JSONL filename in ingress/<caller>/. External service handles are not automatically resolvable."""
        return await invoke("register_records", [r.model_dump() for r in records] if records is not None else None, artifact_name, selection_scope)
    @server.tool()
    async def run_analysis(dataset_id: str, operations: list[Operation], execution: Execution | None = None, idempotency_key: str | None = None) -> dict:
        """Validate a plan and submit a background local worker job. Reuse a stable idempotency key for retries. Topics require clusters_from; sentiment may aggregate by it."""
        return await invoke("run_analysis", dataset_id, [o.model_dump() for o in operations], execution.model_dump() if execution else None, idempotency_key)
    @server.tool()
    async def get_job(job_id: str) -> dict:
        """Inspect job phase, terminal error, or mapping from operation IDs to result handles. Let application code poll long jobs."""
        return await invoke("get_job", job_id)
    @server.tool()
    async def cancel_job(job_id: str) -> dict:
        """Stop the worker and report actual outcome. A job that already completed retains its results."""
        return await invoke("cancel_job", job_id)
    @server.tool()
    async def get_summary(result_id: str) -> dict:
        """Get compact computed counts, scope, effective parameters and provenance, without full assignments or embeddings."""
        return await invoke("get_summary", result_id)
    @server.tool()
    async def get_examples(result_id: str, cluster_id: int | None = None, limit: int = 5) -> dict:
        """Inspect up to 100 representative source IDs and text excerpts, optionally within one cluster."""
        return await invoke("get_examples", result_id, cluster_id, limit)
    @server.tool()
    async def get_result_page(result_id: str, cursor: str | None = None, limit: int = 100) -> dict:
        """Page record assignments, sentiment scores, topic terms or ranked outliers. Cursor is bound to this immutable result."""
        return await invoke("get_result_page", result_id, cursor, limit)
    @server.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "ok", "server_version": "0.1.0", "supervisor_alive": service.supervisor.thread.is_alive()})
    @server.custom_route("/ingress/{artifact_name}", methods=["POST", "DELETE"])
    async def ingress(request):
        """Stream bounded JSONL into the authenticated caller's staging directory."""
        name=request.path_params['artifact_name']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}\.jsonl',name):
            return JSONResponse({'error':'Expected a simple .jsonl filename'},status_code=400)
        folder=service.config.root/'ingress'/principal.get()
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/name
        if request.method=='DELETE':
            path.unlink(missing_ok=True)
            return JSONResponse({'deleted':True})
        if request.headers.get('content-type','').split(';')[0]!='application/x-ndjson':
            return JSONResponse({'error':'Use application/x-ndjson'},status_code=415)
        cap=service.config.limits.max_bytes
        try:
            length=int(request.headers.get('content-length','0'))
        except ValueError:
            return JSONResponse({'error':'Invalid Content-Length'},status_code=400)
        if length<0 or length>cap:
            return JSONResponse({'error':'Upload exceeds max_bytes'},status_code=413)
        temp_path=None
        try:
            with tempfile.NamedTemporaryFile(dir=folder,prefix='.upload-',delete=False) as target:
                temp_path=Path(target.name)
                size=0
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>cap:
                        return JSONResponse({'error':'Upload exceeds max_bytes'},status_code=413)
                    await asyncio.to_thread(target.write,chunk)
            # Publish only a complete upload; never replace another staged input.
            try:os.link(temp_path,path)
            except FileExistsError:
                return JSONResponse({'error':'Artifact already exists; choose a new filename'},status_code=409)
            return JSONResponse({'artifact_name':name,'bytes':size},status_code=201)
        finally:
            if temp_path is not None:temp_path.unlink(missing_ok=True)
    return server

class AccessMiddleware:
    def __init__(self, app, config, allow_local=False):
        self.app, self.config, self.allow_local = app, config, allow_local
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope.get("path") == "/health":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        owner = None
        for name, cfg in self.config.callers.items():
            token = os.getenv(cfg.get("token_env", "VECTOR_ML_MCP_TOKEN"))
            if token and hmac.compare_digest(auth, "Bearer " + token):
                owner = name; break
        if owner is None and self.allow_local and not auth:
            owner = "local"
        if owner is None:
            return await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
        if scope.get('path','').startswith('/ingress/'):
            token=principal.set(owner)
            try:
                # The upload route enforces max_bytes while streaming, rather than buffering vectors.
                await self.app(scope,receive,send)
            finally:principal.reset(token)
            return
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > 2 * 1024 * 1024:
                return await JSONResponse({"error": "Request exceeds 2 MiB; use staged JSONL registration"}, status_code=413)(scope, receive, send)
            if not event.get("more_body", False):
                break
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        token = principal.set(owner)
        try:
            await self.app(scope, replay, send)
        finally:
            principal.reset(token)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = Config(args.config)
    service = Service(config)
    server = build_server(service, args.host, args.port)
    try:
        if args.transport == "stdio":
            principal.set(os.getenv("VECTOR_ML_CALLER", "local"))
            server.run(transport="stdio")
        else:
            import uvicorn
            allow_local = args.host in {"127.0.0.1", "localhost", "::1"} and not any(os.getenv(c.get("token_env", "VECTOR_ML_MCP_TOKEN")) for c in config.callers.values())
            if not allow_local and not any(os.getenv(c.get("token_env", "VECTOR_ML_MCP_TOKEN")) for c in config.callers.values()):
                raise VTError("Non-loopback HTTP requires configured bearer tokens")
            uvicorn.run(AccessMiddleware(server.streamable_http_app(), config, allow_local), host=args.host, port=args.port, log_level="warning")
    finally:
        service.close()

if __name__ == "__main__":
    main()
