"""Agent/Runner on real MCP tools. Application polls and resumes within the same context."""
import asyncio
import json
import os
import sys
import uuid
from agents import Agent, Runner, RunConfig
from agents.exceptions import MaxTurnsExceeded
from agents.mcp import MCPServerStdio, MCPServerStreamableHttp
from examples.mcp_client import content, poll

class CaptureCalls:
    def __init__(self, *args, **kwargs):
        self.trace = []
        self.jobs = []
        super().__init__(*args, **kwargs)
    async def call_tool(self, tool_name, arguments, *args, **kwargs):
        result = await super().call_tool(tool_name, arguments, *args, **kwargs)
        value = None if result.isError else content(result)
        self.trace.append({"tool":tool_name, "arguments":arguments, "result":value, "is_error":result.isError})
        if tool_name == "run_analysis" and value and value.get("job_id"):
            if value["job_id"] not in self.jobs:
                self.jobs.append(value["job_id"])
        return result

class Stdio(CaptureCalls, MCPServerStdio):
    pass

class Http(CaptureCalls, MCPServerStreamableHttp):
    pass

def connection(transport):
    common = {"name":"Vector ML", "client_session_timeout_seconds":30, "use_structured_content":True}
    if transport == "stdio":
        args = ["-m", "vector_tools.server"]
        if os.getenv("VECTOR_ML_CONFIG"):
            args += ["--config", os.environ["VECTOR_ML_CONFIG"]]
        return Stdio(params={"command":sys.executable, "args":args, "env":dict(os.environ)}, **common)
    token = os.getenv("VECTOR_ML_MCP_TOKEN")
    return Http(params={"url":os.environ["VECTOR_ML_MCP_URL"], "headers":{"Authorization":f"Bearer {token}"} if token else {}}, **common)

async def workflow(server, model, source_id="demo", max_turns=20, *, sample_n=62, n_clusters=4, subject="records"):
    request_key = uuid.uuid4().hex
    selection_instruction = (f"Create a seeded sample of up to {sample_n} records. " if sample_n is not None else "Create a complete dataset using selection={} with no sampling. ")
    agent = Agent(name="Record Analyst", model=model, mcp_servers=[server], instructions=(
        f"Analyze {subject}. Discover capabilities and inspect the configured source. "
        + selection_instruction +
        f"Then submit one plan: cluster with {n_clusters} groups, ctfidf topics, VADER sentiment aggregated by groups, and outliers. "
        f"Use idempotency_key {request_key} on submission and retries. "
        "Check job status once. If pending, return the job ID and pending status. The host will poll and resume you. "
        "On completion inspect every summary and representative examples. Answer with computed counts, source IDs, "
        "scope and limitations. Counts describe vector records. Themes are evidence for investigation, not confirmed causes. "
        "On resumption reuse the completed job and dataset; never submit another job just to retrieve results."))
    history = [{"role":"user", "content":f"Analyze configured source {source_id}."}]
    config = RunConfig(tracing_disabled=True)
    first = None
    turn_limit_hit = False
    try:
        first = await Runner.run(agent, input=history, max_turns=max_turns, run_config=config)
        history = first.to_input_list()
    except MaxTurnsExceeded as exc:
        turn_limit_hit = True
        # Supported SDK RunState retains actual tool history if a turn limit interrupts the run.
        if getattr(exc, "run_data", None) is not None:
            history = exc.run_data.input + [item.to_input_item() for item in exc.run_data.new_items]
        if not server.jobs:
            raise RuntimeError("Turn limit reached before submission; no job was created") from exc
    async def call(name, args):
        return content(await server.call_tool(name, args))
    completed = []
    for job_id in server.jobs:
        completed.append(await poll(call, job_id))
    if completed:
        history.append({"role":"user", "content":"Completed jobs (reuse these handles): " + json.dumps(completed)})
        try:
            final = await Runner.run(agent, input=history, max_turns=20, run_config=config)
        except MaxTurnsExceeded as exc:
            return {"status":"turn_limit", "jobs":completed, "trace":server.trace, "error":"Inspection exceeded turn limit; durable result handles are retained."}
    else:
        final = first
    return {"status":("completed" if all(j["status"] == "succeeded" for j in completed) else "job_error") if completed else "no_analysis",
            "initial_turn_limit_hit":turn_limit_hit, "initial_answer":first.final_output if first else None,
            "final_answer":final.final_output if final else None, "jobs":completed, "trace":server.trace}

async def main():
    transport = os.getenv("VECTOR_ML_TRANSPORT", "stdio")
    if "--deterministic" in sys.argv:
        from examples.deterministic_model import ScriptedAnalyst
        model = ScriptedAnalyst(os.getenv("VECTOR_ML_SOURCE_ID", "demo"))
        mode = "deterministic model stub; real Agent/Runner/MCP/ML"
    else:
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL"):
            raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL for a live run; --deterministic explicitly uses a model stub.")
        model = os.environ["OPENAI_MODEL"]
        mode = "live model"
    async with connection(transport) as server:
        result = await workflow(server, model, os.getenv("VECTOR_ML_SOURCE_ID", "demo"))
    print(json.dumps({"mode":mode, **result}, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
