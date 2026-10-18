# Use EigenLens with the OpenAI Agents SDK

Your existing Python agent can use EigenLens alongside its current tools. The Agents SDK discovers and calls MCP tools. The EigenLens server retrieves stored vectors and runs DBSCAN and BERTopic in background worker processes.

## 1. Install and connect

For an already running [EigenLens HTTP worker](worker.md#docker), install only the Agents SDK in your agent application. No repository clone or local EigenLens installation is required:

```bash
pip install 'openai-agents==0.22.2'
export EIGENLENS_MCP_URL=http://127.0.0.1:8020/mcp
```

Supply these values through your application's private environment:

| Variable | Value |
| --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key for model calls |
| `OPENAI_MODEL` | A tool-capable OpenAI model available to your application |
| `EIGENLENS_MCP_URL` | The worker's Streamable HTTP endpoint, including `/mcp` |
| `EIGENLENS_MCP_TOKEN` | The bearer token configured on that worker |
| `EIGENLENS_CONNECTION_ID` | A saved database connection ID owned by the worker caller |
| `EIGENLENS_COLLECTION` | The collection to explore, such as `banking77_minilm` |

The OpenAI API key, MCP bearer token and vector database API key authenticate three different connections. The MCP URL must be reachable from your agent runtime: inside another container, `127.0.0.1` refers to that container. For a client on the worker's Compose network, the endpoint is `http://worker:8000/mcp`. Use your deployment's HTTPS endpoint for remote access.

The examples use SDK-managed MCP: **your Python process connects to EigenLens**. OpenAI does not need network access to your private worker. Tool descriptions and results used by the agent are included in model requests.

## 2. Create the database connection once

The worker needs its database driver and a persistent `EIGENLENS_CONNECTION_KEY`; Docker includes the supported drivers. Configure these on the server as described in [database setup](databases.md#where-settings-live).

If your application already saved a connection, use `list_connections` to find its ID and skip creation. Otherwise, run this setup script from your application backend. Provide `QDRANT_URL` and optional `QDRANT_API_KEY` privately. The URL must be reachable **from the worker**, for example `http://qdrant:6333` for Compose's Qdrant service.

```python
import asyncio
import json
import os

from agents.mcp import MCPServerStreamableHttp


def open_eigenlens():
    return MCPServerStreamableHttp(
        name="EigenLens",
        params={
            "url": os.environ["EIGENLENS_MCP_URL"],
            "headers": {
                "Authorization": f"Bearer {os.environ['EIGENLENS_MCP_TOKEN']}"
            },
            "timeout": 300,
            "sse_read_timeout": 300,
        },
        cache_tools_list=True,
        client_session_timeout_seconds=300,
    )


async def call(server, name, arguments):
    reply = await server.call_tool(name, arguments)
    if reply.isError:
        raise RuntimeError(f"EigenLens tool failed: {name}")
    if reply.structuredContent is not None:
        return reply.structuredContent
    return json.loads(next(item.text for item in reply.content if item.type == "text"))


async def main():
    async with open_eigenlens() as server:
        connection = await call(server, "connect_database", {
            "db_type": "qdrant",
            "url": os.environ["QDRANT_URL"],
            "api_key": os.environ.get("QDRANT_API_KEY"),
            "name": "My database",
        })
        print("Save this as EIGENLENS_CONNECTION_ID:", connection["connection_id"])
        collections = await call(server, "list_collections", {
            "connection_id": connection["connection_id"],
        })
        print(collections)


if __name__ == "__main__":
    asyncio.run(main())
```

Save this as `eigenlens_connection.py`, then run `python eigenlens_connection.py`. Retain the returned connection ID in your application configuration and set `EIGENLENS_CONNECTION_ID` to it. Set `EIGENLENS_COLLECTION` to a returned collection name. Reuse the connection rather than creating one per message.

This setup calls MCP directly without a model call, keeping database credentials out of the model prompt. The server saves the encrypted connection. IDs and jobs are scoped to the authenticated caller; continue using the same worker and caller identity.

## 3. Give the agent EigenLens tools

Save this beside the setup file as `agent_app.py`:

```python
import asyncio
import os

from agents import Agent, Runner
from eigenlens_connection import open_eigenlens


async def main():
    async with open_eigenlens() as server:
        agent = Agent(
            name="My assistant",
            model=os.environ["OPENAI_MODEL"],
            instructions=(
                "Use EigenLens tools for vector database questions. "
                f"Use connection_id={os.environ['EIGENLENS_CONNECTION_ID']} "
                f"and collection={os.environ['EIGENLENS_COLLECTION']}. "
                "List collections when asked; listing does not require analysis. "
                "For clustering, inspect get_capabilities, call retrieve_records, "
                "then call run_analysis with DBSCAN search and BERTopic. "
                "For a text search, report the returned selection count and search_limit. "
                "Analyze every retrieved record. For a whole collection, omit query "
                "and query_vector; do not send the string 'None'. "
                "After submission, return the job_id and status; the application "
                "will poll. Never claim a pending job has finished."
            ),
            mcp_servers=[server],
        )
        result = await Runner.run(agent, "Which collections are available?")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

Run `python agent_app.py`. Change the prompt to, for example:

> Show clusters in ATM-related records from the selected collection. Use DBSCAN search and BERTopic, and return the job ID.

Text retrieval requires the worker to resolve the collection's embedding model. Configure its saved options/model metadata, or generate a compatible `query_vector` in your application. Whole-collection retrieval uses stored vectors without generating a query embedding. See [retrieval settings and limits](databases.md#retrieve_records).

### Add to an existing agent, including a weather agent

If you already have an `agent` with a weather function or other tools, attach the connected server before running it:

```python
async with open_eigenlens() as server:
    agent.mcp_servers = [*agent.mcp_servers, server]
    # Add the database context and EigenLens workflow above to the agent's
    # existing instructions (or its dynamic instructions function).
    result = await Runner.run(agent, user_message)
```

Keep the existing `agent.tools`, model, handoffs and other MCP servers. Run all turns using this server inside its `async with` block. For multiple turns, retain conversation context with the SDK's session support or `result.to_input_list()`; store connection ID, selected collection, job IDs and result IDs in application state too.

There is no core `select_collection` tool. The client keeps the selected collection and passes it to `retrieve_records`. When a user changes collections, update that client state; the core checks the explicit collection on each retrieval.

## 4. Start clustering and display detailed job stages

The agent can call `run_analysis` and return a job ID. For long jobs, let application code poll `get_job` so every progress update does not require another model call.

Add this helper to `eigenlens_connection.py` above its `if __name__ == "__main__":` block:

```python
async def wait_for_job(server, job_id, timeout=1800):
    deadline = asyncio.get_running_loop().time() + timeout
    previous = None
    delay = 1.0
    while True:
        job = await call(server, "get_job", {"job_id": job_id})
        info = job.get("info") or {}
        update = (job["status"], info.get("stage"), info.get("message"),
                  info.get("completed"), info.get("total"))
        if update != previous:
            # Replace print with your UI's status/progress callback.
            print(job["status"], info.get("stage"), info.get("message"),
                  info.get("completed"), info.get("total"))
            previous = update
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Job {job_id} is still pending; retain its ID to resume")
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 5.0)
```

For a deterministic application-controlled workflow, replace the setup script's `main()` with the following after saving the connection ID. This shows the **actual clustering call** and reads its completed results without calling a language model:

```python
async def main():
    async with open_eigenlens() as server:
        await call(server, "get_capabilities", {})
        selection = await call(server, "retrieve_records", {
            "connection_id": os.environ["EIGENLENS_CONNECTION_ID"],
            "collection": os.environ["EIGENLENS_COLLECTION"],
            "query": "ATM card and cash withdrawal problems",
            "search_limit": 1000,
        })
        print("Retrieved selection:", selection)
        if selection["count"] == 0:
            print("No records matched; no analysis submitted.")
            return
        job = await call(server, "run_analysis", {
            "dataset_id": selection["dataset_id"],
            "operations": [
                {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
                {"id": "themes", "type": "topics", "algorithm": "bertopic",
                 "clusters_from": "groups", "parameters": {"ngram_max": 2}},
            ],
        })
        print("Save job ID:", job["job_id"])
        job = await wait_for_job(server, job["job_id"])
        if job["status"] != "succeeded":
            print("Analysis did not succeed:", job["status"], job.get("error"))
            return
        for operation, result_id in job["results"].items():
            summary = await call(server, "get_summary", {"result_id": result_id})
            print(operation, result_id, summary)
        examples = await call(server, "get_examples", {
            "result_id": job["results"]["groups"], "limit": 5,
        })
        print("Representative examples:", examples)
```

For a job submitted earlier by the agent, call `wait_for_job(server, saved_job_id)` directly instead of retrieving and submitting again. A polling timeout does **not** cancel the worker. Use `cancel_job` explicitly to request cancellation. An individual tool call still has the connection's 300-second request timeout; this is separate from the background job's time budget.

`status` controls completion. `info` adds the current stage, message, operation and any measurable counters. Stage counters are not an overall percentage. Short stages may finish between polls. Cache hits may already be `succeeded` when submitted. See [get_job fields](mcp.md#4-get_job).

After success, pass the result IDs and a request to summarize them into `Runner.run`; the agent can call `get_summary` and `get_examples` itself. For plots or complete exports, application code must follow every `get_result_page.next_cursor` and render the returned `pca_3d` coordinates. MCP returns data; the Agents SDK does not automatically create or display a chart.

## Optional: local stdio instead of HTTP

Install EigenLens with [pip](../README.md#installation) on the agent machine, then replace the HTTP connection with:

```python
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="EigenLens",
    params={
        "command": "eigenlens",
        "args": [],
        "env": {**os.environ,
                "EIGENLENS_DATA": os.path.expanduser("~/.local/share/eigenlens")},
    },
    cache_tools_list=True,
    client_session_timeout_seconds=300,
) as server:
    # Create/use your agent with mcp_servers=[server] here.
    capabilities = await call(server, "get_capabilities", {})
```

The SDK launches and stops the local MCP process. Analysis runs in that server's local worker processes. No bearer token is needed for stdio. Configure the database/encryption environment on this launcher, and use one server per data directory. The default local limit is 10,000 rows; use the [BANKING77 configuration](../README.md#try-the-banking77-demo) for the full demo corpus.

## Reference and verification

- [Official OpenAI Agents SDK integrations documentation](https://developers.openai.com/api/docs/guides/agents/integrations-observability) explains SDK-managed MCP and server lifetimes.
- The HTTP constructor, `call_tool`, agent attachment and stdio parameters above were checked against installed `openai-agents==0.22.2`. The MCP connection and read-only tool calls can be verified without a model API key; `Runner.run` requires your configured model and OpenAI API key.
- [All EigenLens tools and exact arguments](mcp.md) and [database adapters](databases.md) describe the server-side contract.
