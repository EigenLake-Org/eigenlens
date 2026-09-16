# OpenAI Agents SDK integration

`examples/agent_client.py` implements both supported transports using `Agent(mcp_servers=[server])` and `Runner.run`. The optional `agents` extra installs the client SDK; the core and service run without it. Database credentials belong to the service; `OPENAI_API_KEY` and the selected model belong to the agent host.

## Setup

```sh
pip install -e '.[server,sentiment,agents]' -c requirements-tested.txt
export OPENAI_API_KEY='...'
export OPENAI_MODEL='an-available-model-id'
export VECTOR_ML_SOURCE_ID=demo

# Stdio: the example launches the actual vector_tools.server entry point.
python -m examples.agent_client

# HTTP: point the host at an already-running service.
export VECTOR_ML_TRANSPORT=http
export VECTOR_ML_MCP_URL=http://localhost:8000/mcp
export VECTOR_ML_MCP_TOKEN=local-demo-token
python -m examples.agent_client
```

For local source connections, set `VECTOR_ML_CONFIG=config/local.json` and the configured credential variables before launching the stdio example. Docker uses `/app/config/docker.json` with service hostnames. An agent running on the host reaches the published `http://localhost:8000/mcp`; an agent in another container on the Compose network reaches `http://service:8000/mcp`. `localhost` inside an agent container points to that container itself. HTTP uses `MCPServerStreamableHttp`, a bearer header, a 30-second client session timeout and `use_structured_content=True`. Stdio uses `MCPServerStdio` with `python -m vector_tools.server`, carrying environment configuration into the subprocess. Server diagnostics go to stderr.

## Discovery and invocation

The official MCP SDK provides schemas and structured-content responses with text fallback. Tests use the actual Agents SDK `MCPUtil.get_all_function_tools` discovery/conversion path, including nested operations, optional fields, result schemas and real tool invocations. The server exposes these names:

1. `get_capabilities`
2. `describe_source`
3. `create_dataset`
4. `register_records`
5. `run_analysis`
6. `get_job`
7. `cancel_job`
8. `get_summary`
9. `get_examples`
10. `get_result_page`

Some open-ended metadata/result objects are intentionally non-strict JSON objects in MCP. The SDK conversion path is tested rather than assuming every schema is suitable for a strict OpenAI function schema. Transport/schema errors remain explicit tool errors.

## Long jobs and conversation preservation

The example keeps the server's async context open across the initial run, application polling, and resumed run. A capture wrapper records actual tool calls and submitted job IDs. The agent checks status once and can return pending. The host polls `get_job` with bounded exponential backoff from 0.2 seconds to 3 seconds and a 300-second deadline, without repeated model calls merely to wait.

On completion, `RunResult.to_input_list()` preserves the conversation and tool history. The host appends completed job/result handles and invokes `Runner.run` again for evidence inspection. The initial prompt gives a stable idempotency key; retries cannot create duplicate jobs. The host reuses the original job even when the initial agent turn limit is exceeded, preserving `MaxTurnsExceeded.run_data` history when available. Terminal job errors are passed into the resumed run. A polling timeout identifies the retained job ID for later resumption; it does not create another submission or claim completion.

A host process that exits entirely can reconnect over HTTP and poll a saved job ID. The example does not persist the entire agent conversation to disk across host-process restarts; use a supported durable SDK session for that additional application requirement. With stdio, exiting the context shuts down the server and interrupts incomplete jobs, so polling and resumption must remain inside that context.

## Credential-free acceptance demonstration

```sh
python -m examples.agent_client --deterministic
```

This uses an explicitly scripted **model stub**. `Agent`, `Runner`, MCP discovery/conversion, transport, jobs and algorithms remain real. It exercises initial-run completion, host polling, resumption and inspection of actual summaries/examples. It does not verify an OpenAI model's reasoning or live API access.

The completed synthetic fixture run computed 62 records and four cluster sizes of 20, 20, 20 and 2 (cluster labels are arbitrary). VADER returned 42 negative and 20 positive records. IsolationForest ranked the two deliberately unusual records first. The returned answer uses actual counts and inspected source IDs. See [verification](verification.md) and the saved demonstration evidence for actual handles and tool-call ordering.

The same example switches sources using `VECTOR_ML_SOURCE_ID` without changing agent setup. For live model verification, run both stdio and HTTP against each configured source with a valid API key and accessible chosen model; those checks are not counted as passing when credentials are absent.

## Versions and references

The tested release combines Vector Tools 0.1.0, `mcp` 1.30.0 and `openai-agents` 0.22.2. The real initialized client negotiated MCP protocol **2025-11-25**. A package version is not a protocol version; linking newer protocol documentation does not mean that protocol revision was negotiated. The Agents SDK emits a deprecation warning for its internal v1 HTTP client alias; the tested HTTP flow still passes.

Implementation follows the official [Agents SDK MCP guide](https://openai.github.io/openai-agents-python/mcp/), [MCP client reference](https://openai.github.io/openai-agents-python/ref/mcp/server/) and [Runner reference](https://openai.github.io/openai-agents-python/ref/run/). No default model is hardcoded: set `OPENAI_MODEL` to a model available to your account.
