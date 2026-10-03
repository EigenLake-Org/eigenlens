import json
from examples.agent_client import connection
from examples.mcp_client import content, poll
from test_density import fixture_data, operation


async def test_dbscan_over_real_mcp_stdio(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_ML_DATA", str(tmp_path))
    monkeypatch.delenv("VECTOR_ML_CONFIG", raising=False)
    async with connection("stdio") as server:
        async def call(name, args):
            return content(await server.call_tool(name, args))
        caps = await call("get_capabilities", {})
        assert "dbscan_search" in caps["operations"]["cluster"]["algorithms"]
        ds = await call("register_records", {"records": [r.model_dump() for r in fixture_data().records]})
        job = await call("run_analysis", {"dataset_id": ds["dataset_id"], "operations": [operation()]})
        done = await poll(call, job["job_id"])
        assert done["status"] == "succeeded", done
        ref = done["results"]["groups"]
        summary = await call("get_summary", {"result_id": ref})
        assert summary["processed"] == 33
        assert summary["summary_data"]["noise_count"] == 1
        page = await call("get_result_page", {"result_id": ref, "limit": 100})
        assert len(page["rows"]) == 33
        assert len(page["rows"][0]["pca_3d"]) == 3
        assert (await call("get_examples", {"result_id": ref}))["examples"]
        json.dumps(summary, allow_nan=False)


async def test_agent_runner_uses_registered_dbscan_plan(tmp_path, monkeypatch):
    """The model is a scripted stub; Runner, MCP, job worker and algorithms are real."""
    from agents import Model, ModelResponse, Usage
    from openai.types.responses import ResponseFunctionToolCall
    from examples.deterministic_model import ScriptedAnalyst
    from examples.agent_client import workflow
    import uuid

    class DensityAnalyst(Model):
        def __init__(self, dataset_id):
            self.dataset_id = dataset_id
            self.step = 0
        def stream_response(self, *args, **kwargs):
            raise NotImplementedError
        async def get_response(self, system_instructions, input, *args, **kwargs):
            assert self.dataset_id in system_instructions
            assert 'dbscan_search' in system_instructions
            completed = [i for i in input if isinstance(i.get('content'), str) and i['content'].startswith('Completed jobs (reuse these handles): ')]
            if completed:
                done = json.loads(completed[-1]['content'].split(': ', 1)[1])[0]
                if self.step == 2:
                    action = ('get_summary', {'result_id':done['results']['groups']})
                elif self.step == 3:
                    action = ('get_examples', {'result_id':done['results']['groups']})
                else:
                    return ScriptedAnalyst().message('Inspected completed DBSCAN results.')
            elif self.step == 0:
                action = ('get_capabilities', {})
            elif self.step == 1:
                action = ('run_analysis', {'dataset_id':self.dataset_id, 'operations':[operation()]})
            else:
                return ScriptedAnalyst().message('Pending; host will poll.')
            self.step += 1
            call_id = 'call_' + uuid.uuid4().hex
            return ModelResponse(output=[ResponseFunctionToolCall(id=call_id, call_id=call_id,
                name=action[0], arguments=json.dumps(action[1]), type='function_call')], usage=Usage(), response_id=None)

    monkeypatch.setenv('VECTOR_ML_DATA', str(tmp_path))
    monkeypatch.delenv('VECTOR_ML_CONFIG', raising=False)
    async with connection('stdio') as server:
        ds = content(await server.call_tool('register_records', {'records':[r.model_dump() for r in fixture_data().records]}))
        run = await workflow(server, DensityAnalyst(ds['dataset_id']), dataset_id=ds['dataset_id'], operations=[operation()])
        assert run['status'] == 'completed'
        assert len(server.jobs) == 1
        assert run['final_answer'] == 'Inspected completed DBSCAN results.'
        assert not any(row['tool']=='create_dataset' for row in server.trace)
