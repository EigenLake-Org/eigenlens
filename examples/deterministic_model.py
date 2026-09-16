"""Deterministic MODEL STUB for acceptance tests. MCP, Runner, workers and ML are real.
This does not establish a successful live OpenAI model integration.
"""
import json
import uuid
from agents import Model, ModelResponse, Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from examples.mcp_client import OPERATIONS

class ScriptedAnalyst(Model):
    def __init__(self, source="demo"):
        self.source = source
        self.step = 0
        self.calls = {}
        self.values = {}
        self.queue = []
        self.key = uuid.uuid4().hex
    def stream_response(self, *args, **kwargs):
        raise NotImplementedError("Stub supports non-streaming Runner.run only")
    async def get_response(self, system_instructions, input, *args, **kwargs):
        for item in input if isinstance(input, list) else []:
            if item.get("type") == "function_call_output" and item.get("call_id") in self.calls:
                payload = item["output"]
                if isinstance(payload, list):
                    payload = "".join(p.get("text", "") for p in payload)
                try:
                    self.values[self.calls[item["call_id"]]] = json.loads(payload) if isinstance(payload, str) else payload
                except (ValueError, TypeError):
                    pass
        if self.step < 6 and any(isinstance(i.get("content"), str) and i["content"].startswith("Completed jobs (reuse these handles): ") for i in input if isinstance(i, dict)):
            self.step = 6
        if self.step == 0:
            action = ("get_capabilities", {})
        elif self.step == 1:
            action = ("describe_source", {"source_id":self.source})
        elif self.step == 2:
            action = ("create_dataset", {"source_id":self.source, "selection":{"sample_n":62,"seed":42}})
        elif self.step == 3:
            action = ("run_analysis", {"dataset_id":self.values["create_dataset"]["dataset_id"], "operations":OPERATIONS, "idempotency_key":self.key})
        elif self.step == 4:
            action = ("get_job", {"job_id":self.values["run_analysis"]["job_id"]})
        elif self.step == 5:
            self.step += 1
            job = self.values["get_job"]
            return self.message(f"Job {job['job_id']} is {job['status']}. The host will poll and resume inspection.")
        else:
            if self.step == 6:
                completed = None
                for item in input:
                    text = item.get("content", "")
                    if isinstance(text, str) and text.startswith("Completed jobs (reuse these handles): "):
                        completed = json.loads(text.split(": ", 1)[1])[0]
                if completed is None or completed["status"] != "succeeded":
                    return self.message("Analysis has no completed results; inspect the terminal job error.")
                self.queue = [(name, {"result_id":ref}) for ref in completed["results"].values() for name in ["get_summary", "get_examples"]]
            if not self.queue:
                summaries = [v for k,v in self.values.items() if k.startswith("summary:")]
                groups = next(v for v in summaries if v["operation"] == "cluster")
                sent = next(v for v in summaries if v["operation"] == "sentiment")
                examples = [r for k,v in self.values.items() if k.startswith("examples:") for r in v["examples"]]
                ids = list(dict.fromkeys(r["id"] for r in examples))[:12]
                return self.message(f"Computed analysis covers {groups['processed']} vector records ({groups['scope']}). Cluster sizes: {groups['summary_data']['cluster_sizes']}. Sentiment: {sent['summary_data']['labels']}. Inspected source examples: {', '.join(ids)}. Themes are investigation leads; this synthetic sample does not establish collection-wide failure causes.")
            action = self.queue.pop(0)
        self.step += 1
        name, arguments = action
        id = "call_" + uuid.uuid4().hex
        key = "summary:" + arguments["result_id"] if name == "get_summary" else "examples:" + arguments["result_id"] if name == "get_examples" else name
        self.calls[id] = key
        return ModelResponse(output=[ResponseFunctionToolCall(id=id, call_id=id, name=name, arguments=json.dumps(arguments), type="function_call")], usage=Usage(), response_id=None)
    def message(self, text):
        return ModelResponse(output=[ResponseOutputMessage(id="msg_"+uuid.uuid4().hex, role="assistant", type="message", status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])])], usage=Usage(), response_id=None)
