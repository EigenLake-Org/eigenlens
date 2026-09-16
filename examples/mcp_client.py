"""Real MCP client workflow for any configured source; no model needed."""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

OPERATIONS = [
    {"id":"groups", "type":"cluster", "parameters":{"n_clusters":4, "random_state":42}},
    {"id":"themes", "type":"topics", "clusters_from":"groups"},
    {"id":"sentiment", "type":"sentiment", "clusters_from":"groups"},
    {"id":"outliers", "type":"outliers"},
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
        source = os.getenv("VECTOR_ML_SOURCE_ID", "demo")
        await call("describe_source", {"source_id":source})
        selection = {"ids":json.loads(os.environ["VECTOR_ML_IDS"])} if os.getenv("VECTOR_ML_IDS") else {"sample_n":62, "seed":42}
        ds = await call("create_dataset", {"source_id":source, "selection":selection})
        job = await call("run_analysis", {"dataset_id":ds["dataset_id"], "operations":OPERATIONS, "idempotency_key":uuid.uuid4().hex})
        job = await poll(call, job["job_id"])
        if job["status"] != "succeeded":
            raise RuntimeError(job)
        for result in job["results"].values():
            await call("get_summary", {"result_id":result})
            await call("get_examples", {"result_id":result})
        print(json.dumps({"protocol_version":initialized.protocolVersion, "server_version": initialized.serverInfo.version,
                          "tools":[t.name for t in discovered.tools], "trace":trace}, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
