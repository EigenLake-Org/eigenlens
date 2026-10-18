"""Real MCP protocol checks for analysis and caller-owned database tools."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from starlette.testclient import TestClient
from vector_tools.config import Config
from vector_tools.service import Service
from vector_tools.server import AccessMiddleware, build_server
from test_density import fixture_data, operation

TOOLS = {"get_capabilities", "register_records", "run_analysis", "get_job", "cancel_job",
         "get_summary", "get_examples", "get_result_page", "connect_database", "list_connections",
         "list_collections", "retrieve_records", "disconnect_database"}
OPS = [operation(), {"id":"themes", "type":"topics", "clusters_from":"groups", "parameters":{"ngram_max":2}}]


def unpack(result):
    assert not result.isError, result.content
    return result.structuredContent or json.loads(result.content[0].text)


async def test_native_stdio_clustering(tmp_path):
    env = {k:v for k,v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG"}}
    env["VECTOR_ML_DATA"] = str(tmp_path / "core")
    async with stdio_client(StdioServerParameters(command=sys.executable, args=["-m", "vector_tools.server"], env=env)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert {tool.name for tool in (await session.list_tools()).tools} == TOOLS
            async def call(name, args):
                return unpack(await session.call_tool(name, args))
            caps = await call("get_capabilities", {})
            assert set(caps["operations"]) == {"cluster", "topics"}
            assert caps["limits"]["max_jobs"] == 1
            ds = await call("register_records", {"records":[r.model_dump() for r in fixture_data().records]})
            job = await call("run_analysis", {"dataset_id":ds["dataset_id"], "operations":OPS})
            deadline = time.monotonic() + 90  # Includes cold BERTopic imports in a spawned worker.
            while job["status"] not in {"succeeded", "failed", "cancelled"}:
                assert time.monotonic() < deadline, job
                await asyncio.sleep(.05)
                job = await call("get_job", {"job_id":job["job_id"]})
            assert job["status"] == "succeeded", job
            assert job['info']['stage']=='completed'
            assert job['progress']=={'phase':'succeeded','fraction':1}
            summary = await call("get_summary", {"result_id":job["results"]["groups"]})
            assert summary["scope"] == "retrieval_results" and summary["processed"] == 33
            assert summary["summary_data"]["noise_count"] == 1
            rows, cursor = [], None
            while True:
                page = await call("get_result_page", {"result_id":job["results"]["groups"], "limit":7, "cursor":cursor})
                rows.extend(page["rows"])
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            assert len(rows) == len({r["id"] for r in rows}) == 33
            assert all(len(r["pca_3d"]) == 3 for r in rows)
            examples = await call("get_examples", {"result_id":job["results"]["groups"], "cluster_id":0})
            assert len(examples["examples"]) == 5
            topic_page = await call("get_result_page", {"result_id":job["results"]["themes"]})
            assert {term["term"] for row in topic_page["rows"] for term in row["terms"]} >= {"card payment", "cash withdrawal"}
            for removed in ("sentiment", "outliers"):
                rejected = await session.call_tool("run_analysis", {"dataset_id":ds["dataset_id"], "operations":[{"id":"old", "type":removed}]})
                assert rejected.isError


def test_authenticated_http_mcp_without_tcp(tmp_path, monkeypatch):
    # Exercise real ASGI/MCP JSON-RPC routing without requiring an external port.
    monkeypatch.setenv("VECTOR_ML_DATA", str(tmp_path))
    monkeypatch.setenv("VECTOR_ML_MCP_TOKEN", "http-test-token")
    service = Service(Config())
    try:
        server = build_server(service)
        with TestClient(AccessMiddleware(server.streamable_http_app(), service.config)) as client:
            headers = {"Authorization":"Bearer http-test-token", "Accept":"application/json, text/event-stream"}
            counter = 0
            def rpc(method, params=None):
                nonlocal counter
                counter += 1
                response = client.post("/mcp", json={"jsonrpc":"2.0", "id":counter, "method":method, "params":params or {}}, headers=headers)
                assert response.status_code == 200, response.text
                payload = response.json()
                assert "error" not in payload, payload
                return payload["result"]
            rpc("initialize", {"protocolVersion":"2025-11-25", "capabilities":{}, "clientInfo":{"name":"clustering-test", "version":"1"}})
            assert {tool["name"] for tool in rpc("tools/list")["tools"]} == TOOLS
            def call(name, args):
                result = rpc("tools/call", {"name":name, "arguments":args})
                assert not result.get("isError"), result
                return result.get("structuredContent") or json.loads(result["content"][0]["text"])
            ds = call("register_records", {"records":[r.model_dump() for r in fixture_data().records], "selection_scope":"complete_selection"})
            job = call("run_analysis", {"dataset_id":ds["dataset_id"], "operations":OPS})
            deadline = time.monotonic() + 90
            while job["status"] not in {"succeeded", "failed", "cancelled"}:
                assert time.monotonic() < deadline, job
                time.sleep(.05)
                job = call("get_job", {"job_id":job["job_id"]})
            assert job["status"] == "succeeded", job
            assert job['info']['stage']=='completed'
            summary = call("get_summary", {"result_id":job["results"]["groups"]})
            assert summary["processed"] == 33 and summary["scope"] == "complete_selection"
            assert client.post("/mcp", json={"jsonrpc":"2.0", "id":1, "method":"tools/list"}).status_code == 401
    finally:
        service.close()
