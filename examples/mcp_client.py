"""Native MCP clustering workflow over stdio or authenticated HTTP; no model needed."""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

OPERATIONS = [
    {"id":"groups", "type":"cluster", "algorithm":"dbscan_search"},
    {"id":"themes", "type":"topics", "clusters_from":"groups", "parameters":{"ngram_max":2}},
]

def content(result):
    if result.isError:
        raise RuntimeError(str(result.content))
    if result.structuredContent:
        return result.structuredContent
    return json.loads(result.content[0].text)

async def poll(call, job_id, timeout=300):
    deadline = asyncio.get_running_loop().time() + timeout
    delay = .2
    while True:
        job = await call("get_job", {"job_id": job_id})
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"Job {job_id} remains pending; resume polling this ID")
        await asyncio.sleep(delay)
        delay = min(delay * 1.7, 3)

@asynccontextmanager
async def connection(transport="stdio"):
    if transport == "stdio":
        args = ["-m", "vector_tools.server"]
        if os.getenv("VECTOR_ML_CONFIG"):
            args += ["--config", os.environ["VECTOR_ML_CONFIG"]]
        async with stdio_client(StdioServerParameters(command=sys.executable, args=args, env=dict(os.environ))) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                yield session, initialized
    else:
        import httpx
        token = os.getenv("VECTOR_ML_MCP_TOKEN")
        async with httpx.AsyncClient(headers={"Authorization":f"Bearer {token}"} if token else {}) as client:
            async with streamable_http_client(os.environ["VECTOR_ML_MCP_URL"], http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
                    yield session, initialized

async def main():
    import uuid
    trace = []
    async with connection(os.getenv("VECTOR_ML_TRANSPORT", "stdio")) as (session, initialized):
        async def call(name, args):
            value = content(await session.call_tool(name, args))
            trace.append({"tool":name, "arguments":args, "result":value})
            return value
        discovered = await session.list_tools()
        await call("get_capabilities", {})
        from vector_tools.demo import tickets
        ds = await call("register_records", {"records":[r.model_dump() for r in tickets()], "selection_scope":"complete_selection"})
        job = await call("run_analysis", {"dataset_id":ds["dataset_id"], "operations":OPERATIONS, "idempotency_key":uuid.uuid4().hex})
        job = await poll(call, job["job_id"])
        if job["status"] != "succeeded":
            raise RuntimeError(job)
        for result in job["results"].values():
            await call("get_summary", {"result_id":result})
            await call("get_examples", {"result_id":result})
            cursor = None
            rows = []
            while True:
                page = await call("get_result_page", {"result_id":result, "cursor":cursor, "limit":25})
                rows.extend(page["rows"])
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            assert len(rows) == page["total"]
        print(json.dumps({"protocol_version":initialized.protocolVersion, "server_version": initialized.serverInfo.version,
                          "tools":[t.name for t in discovered.tools], "trace":trace}, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
