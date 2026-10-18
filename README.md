# EigenLens

Cluster existing embeddings and model their topics through Python or an MCP job server.

- **Clustering:** PCA + DBSCAN parameter search.
- **Topics:** [BERTopic](https://maartengr.github.io/BERTopic/getting_started/manual/manual.html) keywords and phrases from the discovered clusters.
- **Jobs:** queueing, cancellation and saved results. Default: **1 worker**; configurable up to 16.

## Installation

Requires Python 3.11 or newer. Install the MCP server with pip from this repository's implementation branch:

```bash
pip install 'eigenlens[server,qdrant] @ git+https://github.com/EigenLake-Org/eigenlens.git@feature/banking77-dbscan'
```

This installs the `eigenlens` command, DBSCAN, BERTopic and the Qdrant adapter. Other database extras are `pinecone`, `milvus`, `weaviate` and `pgvector`; use `[server,databases]` to install all five. More backends can provide [adapter plugins](docs/databases.md#add-another-database). The BANKING77 demo below adds dataset dependencies.

## Use your existing Python database client

Pass your application's initialized **QdrantClient** or **WeaviateClient v4** directly. `from_client` detects the SDK; reads start when you call `materialize()`. You keep control of credentials, connection settings, timeouts and closing the client.

```python
import eigenlens as el
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
try:
    data = el.from_client(
        client, collection="banking77_minilm",
        limits=el.Limits(max_rows=15000, max_bytes=256 * 1024 * 1024,
                         max_working_bytes=6 * 1024**3, max_seconds=1800),
    ).materialize()
    results = el.analyze(data, [
        {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
        {"id": "themes", "type": "topics", "clusters_from": "groups",
         "parameters": {"ngram_max": 2}},
    ])
    print(results["groups"].summary())
    print(results["themes"].summary())
finally:
    client.close()
```

For Weaviate, install the `weaviate` extra and pass its connected client:

```python
import weaviate
import eigenlens as el

with weaviate.connect_to_local() as client:
    data = el.from_client(client, collection="Questions", text_field="content").materialize()
    groups = el.cluster(data)
    themes = el.topics(data, groups, ngram_max=2)
```

These examples read the entire collection across pages, or fail at configured limits. They use stored embeddings. To cluster search results, pass `query_vector` and `search_limit`; generate that query vector with the same model used for the collection. Set `vector_field` only when the collection has multiple dense vector fields. See [Python client options and MCP handoff](docs/databases.md#existing-python-sdk-clients).

The BANKING77 example uses the demo's larger data, working-memory and time budgets for its full 13,069 records; your machine must have enough available memory for that analysis.

Python analysis runs in your application's process. To submit work to the MCP job server, register the materialized records and call `run_analysis`. A live Python client stays in your application; MCP receives JSON records or connection settings.

## Connect MCP to a chat agent

MCP exposes tools that an agent can call during a conversation. You connect the server once, then ask questions in plain language. The server runs the analysis in background jobs.

### Local setup with Codex

After installation, register the server:

```bash
codex mcp add eigenlens \
  --env "EIGENLENS_DATA=$HOME/.local/share/eigenlens" \
  -- eigenlens
codex mcp list
```

Start a new Codex session with the server enabled. Codex launches it over stdio; you do not need to run a separate server command. The chat host must be able to find the installed `eigenlens` executable on its PATH. Use one active server per data directory. See the [official Codex MCP commands](https://learn.chatgpt.com/docs/developer-commands#codex-mcp).

Other local MCP hosts can use the same executable, arguments and environment. For a shared Docker server, use the Streamable HTTP URL `http://127.0.0.1:8020/mcp` and its bearer token; see [connection options and uploads](docs/mcp.md#connection-options).

### Ask this in chat

First [connect your vector database through MCP](#connect-your-vector-database) or [stage a JSONL file](docs/mcp.md#connection-options). Then ask:

> Use eigenlens to cluster the dataset I supplied with DBSCAN search and model its topics with BERTopic using phrases up to two words. Wait for the job to finish, then show cluster sizes, unassigned records, topic terms and five representative examples for each cluster you discuss. Report any analysis failure.

The agent registers your input to obtain a dataset ID, starts a job, polls its status, then reads the result IDs. Keep these IDs in the conversation for follow-up questions. For a ready-made dataset, see the [BANKING77 demo](#try-the-banking77-demo) below.

| Ask in chat | Tools the agent uses |
| --- | --- |
| “What can you analyze, and what are the limits?” | `get_capabilities` |
| “Connect to the database configured on this server.” | `connect_database` with no arguments (local default) |
| “Show my saved database connections.” | `list_connections` |
| “List collections in my database.” | `list_collections` |
| “Show clusters in ATM.” | `retrieve_records` with a query → `run_analysis` → `get_job` |
| “Analyze these supplied vector records.” | `register_records` → `run_analysis` → `get_job` |
| “What is the analysis doing?” | `get_job` |
| “Stop this analysis.” | `cancel_job` |
| “Summarize the clusters and their topics.” | `get_summary` for both results |
| “Show five examples from cluster 0.” | `get_examples` with the cluster result ID |
| “Export every record's cluster assignment.” | `get_result_page`, following every page |
| “Plot the clusters in 3D.” | `get_result_page` for cluster coordinates, then the agent's plotting tool |

Plotting requires a chat client or agent with chart/file rendering tools. MCP returns coordinates and assignments; the BANKING77 Python example also creates a standalone HTML plot.

## Connect your vector database

Database connections are managed by this MCP server through read-only adapters. Configure a default on the server, or have your application's connection form call `connect_database` with `db_type`, URL, API key and optional field/model mappings. It returns a caller-owned `connection_id`; your chat keeps that ID and its selected collection. Supply secrets through configuration or the connection form.

### Where to set connection settings

For a local Qdrant default, set these variables in the environment used to launch MCP:

```bash
# Generate once; retain this key in your private server environment across restarts.
export EIGENLENS_CONNECTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
export EIGENLENS_DATABASE_TYPE=qdrant
export EIGENLENS_DATABASE_URL=http://localhost:6333
# Set EIGENLENS_DATABASE_API_KEY in your private environment for an authenticated database.
```

For Codex, register the server with these environment settings (replace the earlier generic registration):

```bash
codex mcp add eigenlens \
  --env "EIGENLENS_DATA=$HOME/.local/share/eigenlens" \
  --env "EIGENLENS_CONNECTION_KEY=$EIGENLENS_CONNECTION_KEY" \
  --env "EIGENLENS_DATABASE_TYPE=$EIGENLENS_DATABASE_TYPE" \
  --env "EIGENLENS_DATABASE_URL=$EIGENLENS_DATABASE_URL" \
  --env "EIGENLENS_DATABASE_API_KEY=${EIGENLENS_DATABASE_API_KEY:-}" \
  -- eigenlens
```

Start a new session and ask: **“Connect to the configured database and list its collections.”** The no-argument default is available to caller `local`. Other authenticated callers supply their own connection settings. For Docker, keep the same settings in a private `.env` file; the worker receives them through Compose. Its local Qdrant URL is `http://qdrant:6333`. See [database setup and exact MCP calls](docs/databases.md).

Saved connections live in `<EIGENLENS_DATA>/connections.sqlite` (Docker: `/data/connections.sqlite` in `chat-worker-data`). URLs, credentials and options are encrypted. Keep `EIGENLENS_CONNECTION_KEY` outside that database and preserve it across restarts. Connections are scoped to the authenticated caller; chats sharing a server caller identity share that caller's access.

For Qdrant, users provide:

| Setting | Example | Purpose |
| --- | --- | --- |
| Database URL | `http://localhost:6333` or your Qdrant cloud URL | Address reachable from the MCP server |
| API key | Your Qdrant API key | Authentication; optional for local instances without authentication |
| Collection | `banking77_minilm` or your collection name | Records to search or analyze |

The database API key authenticates Qdrant access. The MCP worker's bearer token authenticates a separate connection to the analysis server.

### Example: “Show clusters in ATM”

After connecting and selecting a collection, ask:

> Search my selected collection for ATM-related records. Cluster every retrieved record with DBSCAN and BERTopic. Show the selection count and search limit, cluster sizes, topics, examples and a 3D plot.

The agent uses this workflow:

1. Call `connect_database` once and `list_collections`, then keep the connection ID and selected collection.
2. Call `retrieve_records` with `query: "ATM"`. The server resolves a SentenceTransformer model from saved options or stored `embedding_model` metadata and embeds the query in the background. Other embedding providers can supply `query_vector`.
3. The adapter retrieves stored vectors/text and registers an immutable snapshot, returning `dataset_id`.
4. Call `run_analysis` with that dataset ID and the DBSCAN/BERTopic plan.
5. Poll `get_job`, read summaries/examples/result pages, and render the returned coordinates using the chat client's plotting tools.

```text
connect_database → list_collections → retrieve_records
                    → run_analysis → get_job
                    → clusters + BERTopic terms + plot coordinates
```

The server must know the collection's embedding model; matching dimensions alone does not establish that embedding spaces match. It fails with an actionable error when the model is unknown. Model/field mappings are backend configuration, so users can simply ask questions. To analyze the whole collection, omit `query` and `query_vector`; the adapter traverses all pages without generating a query embedding.

Semantic retrieval is a bounded top-k selection (`search_limit` defaults to 1,000) and returns scope `retrieval_results`. Whole-collection retrieval returns `complete_selection` or fails at row/byte/time limits without registering a partial dataset. Analysis processes all returned records. These counts do not imply every semantic match in the database. External clients can still supply records using `register_records` or [staged JSONL uploads](docs/mcp.md#connection-options).

Supported database adapters: **Qdrant, Pinecone, Milvus, Weaviate and PostgreSQL/pgvector**. `get_capabilities` reports which drivers are installed. Pinecone full scans require serverless indexes; pgvector uses vector tables in the configured schema; Weaviate requires its HTTP and gRPC endpoints. The [database guide](docs/databases.md) covers options and backend limitations.

## Reuse completed analysis

`run_analysis` automatically reuses successful results for the same caller when the exact registered snapshot, selection scope/provenance, operations, limits and analysis implementation/dependency versions match. A new dataset ID containing identical records can hit the cache. Changed vectors, text, metadata or settings cause a fresh job. Failed, cancelled and in-flight jobs are not reused; missing result artifacts cause a cache miss.

`run_analysis` and `get_job` return `cache_hit` and `cached_from_job`. A hit creates a completed job record referencing the original owned results without starting another worker. Result timestamps, materialized artifact IDs and execution host/PID describe that original computation. Cache metadata persists in `metadata.sqlite` across server restarts. Registered snapshots and results still require manual retention management.

To request a fresh computation, include `"execution": {"use_cache": false}` in the `run_analysis` arguments. Normal cache reuse is enabled by default. Explicit idempotency keys continue to identify the exact request, independently of result reuse. See [performance and CPU/memory settings](docs/performance.md).

## All MCP tools

These are tool calls made by your agent, not terminal commands. Optional arguments are marked `?`.

| Tool | Arguments | What it returns |
| --- | --- | --- |
| `get_capabilities` | None | Algorithms, supported parameters and server limits |
| `connect_database` | `db_type?`, `url?`, `api_key?`, `options?`, `name?` | Saved `connection_id`; no arguments uses the local server default |
| `list_connections` | None | This caller's saved connection IDs and labels |
| `list_collections` | `connection_id` | Collection/index/table names |
| `retrieve_records` | `connection_id`, `collection`; `query?` **or** `query_vector?`, `search_limit?` | Registered `dataset_id`, exact supplied count and selection scope |
| `disconnect_database` | `connection_id` | Removes the saved connection; existing datasets/results remain |
| `register_records` | `records` **or** `artifact_name`; `selection_scope?` | `dataset_id` and registered count |
| `run_analysis` | `dataset_id`, `operations`; `execution?`, `idempotency_key?` | Background job with `job_id` |
| `get_job` | `job_id` | Status plus stage details/counters in `info`; result IDs on success or an error on failure |
| `cancel_job` | `job_id` | Actual job outcome after cancellation |
| `get_summary` | `result_id` | Compact counts, parameters, scope and provenance |
| `get_examples` | `result_id`; `cluster_id?`, `limit?` (default 5, range 1–100) | Representative record IDs and text excerpts |
| `get_result_page` | `result_id`; `cursor?`, `limit?` (default 100, range 1–1,000) | Assignment/coordinate or topic rows, plus `next_cursor` |

There are **13 MCP tools**. See [analysis calls](docs/mcp.md) and [database calls](docs/databases.md) for exact JSON. Summaries are compact; use result pages for the full output. DBSCAN keeps up to five representative examples per cluster. Cluster `-1` is unassigned data.

### The clustering call

Clustering runs through `run_analysis`. Replace the dataset ID with the one returned by `register_records`:

```json
{
  "name": "run_analysis",
  "arguments": {
    "dataset_id": "ds_example",
    "operations": [
      {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"}
    ]
  }
}
```

Poll `get_job` using the returned `job_id`. Keep using `status` for the job outcome; display `info.message` for the current stage (PCA, DBSCAN search, BERTopic, saving results) and `info.completed` / `info.total` when available. These counters describe the current stage, not an overall completion percentage. See the [detailed response](docs/mcp.md#4-get_job).

On success, use `results.groups` as the result ID for summaries, examples and pages. Add a `topics` operation referencing `groups` to include BERTopic; the [complete plan](docs/mcp.md#3-run_analysis) shows that call.

## Try the BANKING77 demo

[BANKING77](https://huggingface.co/datasets/mteb/banking77) contains 13,069 banking questions across 77 intents. The setup uses the full train and test splits, generates MiniLM embeddings once, and indexes them in Qdrant. No OpenAI API key is needed.

Clone the repository for its example scripts and Docker configuration, then install the demo dependencies with pip:

```bash
git clone --branch feature/banking77-dbscan https://github.com/EigenLake-Org/eigenlens.git
cd eigenlens
pip install -c requirements-tested.txt '.[server,banking]'
docker compose --profile qdrant up -d --wait qdrant
python -m examples.banking77 setup
python -m examples.banking77 analyze
```

If you already have the repository, run these commands from its directory starting with `pip install`.

The example keeps a local embedding cache and records dataset/model revisions and a checksum. Repeating setup reuses that cache and upserts the same point IDs. It rejects a collection containing a different dataset snapshot. Run `python -m examples.banking77 prepare` to rebuild the cache. Dataset files and vectors are not committed to Git.

Analysis saves assignments, topic terms and an interactive 3D plot in `demo-output/banking77/`. It processes the complete cached corpus. Clustering uses existing vectors; BERTopic uses those assignments to model topics without downloading an embedding model. `-1` remains unassigned data.

### Use the banking dataset in chat

After setup, run this from the repository directory to register a dedicated MCP server with the banking limits and prepared records:

```bash
codex mcp add eigenlens-banking \
  --env "EIGENLENS_DATA=$PWD/.eigenlens/banking77" \
  -- eigenlens --config "$PWD/config/banking77.json"
```

Start a new session with that server enabled, then ask:

> Use eigenlens-banking to register banking77.jsonl as a complete selection. Cluster all records with DBSCAN search, then model their topics with BERTopic. Wait for completion and show cluster sizes, unassigned records, topic terms and examples.

The prepared artifact is already in this server's staging directory. The dedicated configuration accepts the full corpus; the generic default server accepts up to 10,000 records.

### Run the MCP example without a chat model

From the repository directory after pip installation:

```bash
python examples/mcp_client.py
```

This starts a local MCP server, discovers its tools, analyzes **62 synthetic records**, polls the job and reads summaries, examples and every result page. It needs no language-model API key. For the Python SDK alone, run `python examples/sdk_demo.py`.

## Documentation

- [Docker worker setup and concurrency](docs/worker.md)
- [MCP connection, chat prompts and exact tool calls](docs/mcp.md)
- [Database connections, storage and adapters](docs/databases.md)
- [Record, analysis and upload contracts](docs/contracts.md)
- [Architecture](docs/architecture.md)

For development: `pip install -e '.[server,banking,databases,test]'` then `pytest -q`.

## Migrating from Vector Tools

The project is now **EigenLens**: install `eigenlens`, use `import eigenlens`, and start the MCP server with `eigenlens`. The GitHub repository is `EigenLake-Org/eigenlens`; update an existing checkout with `git remote set-url origin https://github.com/EigenLake-Org/eigenlens.git`.

Existing `import vector_tools` code, module imports, the `vector-ml` command and `python -m vector_tools.server` remain compatibility aliases. New configuration uses `EIGENLENS_*`; previous `VECTOR_ML_*` variables remain supported, with the new names taking precedence. Both adapter entry-point groups are supported; new plugins should use `eigenlens.database_adapters`.

New local storage defaults to `.eigenlens`. An existing `.vector-ml` directory is reused when no explicit data environment variable is supplied and the corresponding new directory does not exist. To keep an existing explicit storage location, set `EIGENLENS_DATA` to that location. Keep the original connection encryption key. Existing Compose project/network/volume identifiers remain `vector-ml` / `vector-ml_default` so upgrades retain stored jobs, connections, Qdrant data and artifacts. Historical result files keep their original provenance.
