"""Real SDK discovery + Runner tests. Only the model is a deterministic stub."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import httpx
import pytest
from agents import Agent, RunContextWrapper
from agents.mcp import MCPUtil
from examples.agent_client import connection, workflow
from examples.deterministic_model import ScriptedAnalyst
from examples.mcp_client import content

@pytest.fixture(params=["stdio","http"])
def transport(request, tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_ML_DATA",str(tmp_path))
    monkeypatch.delenv("VECTOR_ML_CONFIG",raising=False)
    monkeypatch.setenv("VECTOR_ML_MCP_TOKEN","test-token")
    process=None
    if request.param=="http":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
        monkeypatch.setenv("VECTOR_ML_MCP_URL",f"http://127.0.0.1:{port}/mcp")
        process=subprocess.Popen([sys.executable,"-m","vector_tools.server","--transport","http","--port",str(port)], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health").status_code==200:
                    break
            except httpx.ConnectError:
                time.sleep(.1)
        else:
            process.terminate(); process.wait(); raise AssertionError("HTTP startup failed")
    yield request.param
    if process:
        process.terminate(); process.wait(timeout=10)

async def test_agent_workflow_real_discovery(transport):
    async with connection(transport) as server:
        agent=Agent(name="test",model=ScriptedAnalyst(),mcp_servers=[server])
        tools=await MCPUtil.get_all_function_tools([server],True,RunContextWrapper(None),agent)
        assert {t.name for t in tools} == {"get_capabilities","describe_source","create_dataset","register_records","run_analysis","get_job","cancel_job","get_summary","get_examples","get_result_page"}
        run=next(t for t in tools if t.name=="run_analysis")
        assert "operations" in run.params_json_schema["properties"]
        result=await workflow(server,ScriptedAnalyst())
        assert result["status"]=="completed",result
        assert "62 vector records" in result["final_answer"]
        assert len(server.jobs)==1
        assert {row["tool"] for row in server.trace} >= {"get_summary","get_examples","get_job"}
        ref=result["jobs"][0]["results"]["groups"]
        page=content(await server.call_tool("get_result_page",{"result_id":ref,"limit":3}))
        wrong=await server.call_tool("get_result_page",{"result_id":result["jobs"][0]["results"]["outliers"],"cursor":page["next_cursor"]})
        assert wrong.isError
        ds=content(await server.call_tool("create_dataset",{"source_id":"demo","selection":{}}))
        bad=await server.call_tool("run_analysis",{"dataset_id":ds["dataset_id"],"operations":[{"id":"c","type":"cluster"}],"execution":{"backend":"spark"}})
        assert bad.isError

async def test_turn_limit_resume_without_duplicate(transport):
    async with connection(transport) as server:
        result=await workflow(server,ScriptedAnalyst(),max_turns=4)
        assert result["initial_turn_limit_hit"]
        assert "62 vector records" in result["final_answer"]
        assert len(server.jobs)==1
        assert result["jobs"][0]["status"]=="succeeded"

async def test_http_auth(transport):
    if transport!="http":
        pytest.skip("HTTP-specific authentication")
    async with httpx.AsyncClient() as client:
        r=await client.post(os.environ["VECTOR_ML_MCP_URL"],json={})
        assert r.status_code==401
        r=await client.post(os.environ["VECTOR_ML_MCP_URL"],headers={"Authorization":"Bearer wrong"},json={})
        assert r.status_code==401

async def test_background_job_responsive_and_terminal_error(transport):
    import time
    from examples.mcp_client import poll
    async with connection(transport) as server:
        ds=content(await server.call_tool("create_dataset",{"source_id":"demo","selection":{}}))
        started=time.monotonic()
        job=content(await server.call_tool("run_analysis",{"dataset_id":ds["dataset_id"],"operations":[{"id":"o","type":"outliers","parameters":{"n_estimators":1000}}]}))
        assert time.monotonic()-started < 3
        assert job["status"] in {"queued","running"}
        started=time.monotonic()
        caps=content(await server.call_tool("get_capabilities",{}))
        assert time.monotonic()-started < 2 and "local" in [b["name"] for b in caps["backends"]]
        stopped=content(await server.call_tool("cancel_job",{"job_id":job["job_id"]}))
        assert stopped["status"] in {"cancelled","succeeded"}
        empty=content(await server.call_tool("register_records",{"records":[]}))
        failed=content(await server.call_tool("run_analysis",{"dataset_id":empty["dataset_id"],"operations":[{"id":"c","type":"cluster"}]}))
        async def call(name,args):
            return content(await server.call_tool(name,args))
        done=await poll(call,failed["job_id"])
        assert done["status"]=="failed" and "Empty selection" in done["error"]

async def test_native_mcp_client(transport):
    from examples.mcp_client import connection as native_connection, poll
    async with native_connection(transport) as (session, initialized):
        assert initialized.serverInfo.version=="0.1.0"
        assert len((await session.list_tools()).tools)==10
        async def call(name,args):
            return content(await session.call_tool(name,args))
        ds=await call("create_dataset",{"source_id":"demo","selection":{"sample_n":10}})
        job=await call("run_analysis",{"dataset_id":ds["dataset_id"],"operations":[{"id":"s","type":"sentiment"}]})
        done=await poll(call,job["job_id"])
        assert done["status"]=="succeeded"
        summary=await call("get_summary",{"result_id":done["results"]["s"]})
        assert summary["processed"]==10
