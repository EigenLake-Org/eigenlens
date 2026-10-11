# Vector Tools

Cluster existing embeddings and model their topics through Python or an MCP job server.

- **Clustering:** PCA + DBSCAN parameter search.
- **Topics:** [BERTopic](https://maartengr.github.io/BERTopic/getting_started/manual/manual.html) keywords and phrases from the discovered clusters.
- **Jobs:** queueing, cancellation and saved results. Default: **1 worker**; configurable up to 16.

## Installation

Requires Python 3.11 or newer. Install the MCP server with pip from this repository's implementation branch:

```bash
pip install 'vector-tools[server] @ git+https://github.com/EigenLake-Org/vector-tools.git@feature/banking77-dbscan'
```

This installs the `vector-ml` command, DBSCAN and BERTopic. The BANKING77 demo below adds its dataset and Qdrant dependencies.

## Connect MCP to a chat agent

MCP exposes tools that an agent can call during a conversation. You connect the server once, then ask questions in plain language. The server runs the analysis in background jobs.

### Local setup with Codex

After installation, register the server:

```bash
codex mcp add vector-tools \
  --env "VECTOR_ML_DATA=$HOME/.local/share/vector-tools" \
  -- vector-ml
codex mcp list
```

Start a new Codex session with the server enabled. Codex launches it over stdio; you do not need to run a separate server command. The chat host must be able to find the installed `vector-ml` executable on its PATH. Use one active server per data directory. See the [official Codex MCP commands](https://learn.chatgpt.com/docs/developer-commands#codex-mcp).

Other local MCP hosts can use the same executable, arguments and environment. For a shared Docker server, use the Streamable HTTP URL `http://127.0.0.1:8020/mcp` and its bearer token; see [connection options and uploads](docs/mcp.md#connection-options).

### Ask this in chat

First [connect your vector database through the agent's retrieval integration](#connect-your-vector-database) or [stage a JSONL file](docs/mcp.md#connection-options). Then ask:

> Use vector-tools to cluster the dataset I supplied with DBSCAN search and model its topics with BERTopic using phrases up to two words. Wait for the job to finish, then show cluster sizes, unassigned records, topic terms and five representative examples for each cluster you discuss. Report any analysis failure.

The agent registers your input to obtain a dataset ID, starts a job, polls its status, then reads the result IDs. Keep these IDs in the conversation for follow-up questions. For a ready-made dataset, see the [BANKING77 demo](#try-the-banking77-demo) below.

| Ask in chat | Tools the agent uses |
| --- | --- |
| “What can you analyze, and what are the limits?” | `get_capabilities` |
| “Analyze these supplied vector records.” | `register_records` → `run_analysis` → `get_job` |
| “What is the analysis doing?” | `get_job` |
| “Stop this analysis.” | `cancel_job` |
| “Summarize the clusters and their topics.” | `get_summary` for both results |
| “Show five examples from cluster 0.” | `get_examples` with the cluster result ID |
| “Export every record's cluster assignment.” | `get_result_page`, following every page |
| “Plot the clusters in 3D.” | `get_result_page` for cluster coordinates, then the agent's plotting tool |

Plotting requires a chat client or agent with chart/file rendering tools. MCP returns coordinates and assignments; the BANKING77 Python example also creates a standalone HTML plot.

## Connect your vector database

Your chat app or agent connects to the database through its retrieval integration, such as application code using the database SDK or a separate database MCP connector. It also connects to this clustering MCP. Supply the database connection settings to that retrieval integration.

For Qdrant, users provide:

| Setting | Example | Purpose |
| --- | --- | --- |
| Database URL | `http://localhost:6333` or your Qdrant cloud URL | Address reachable from the retrieval client |
| API key | Your Qdrant API key | Authentication; optional for local instances without authentication |
| Collection | `banking77_minilm` or your collection name | Records to search or analyze |

The database API key authenticates Qdrant access. The MCP worker's bearer token authenticates a separate connection to the analysis server.

### Example: “Show clusters in ATM”

Once both connections are configured, ask:

> Search my selected collection for ATM-related records. Analyze the matching selection with DBSCAN and BERTopic, then show cluster sizes, topics, examples and a 3D plot.

The agent/application performs this workflow:

1. Resolve the collection's embedding model and, for named vectors, the vector field to use.
2. Embed the ATM query using that same model and search Qdrant.
3. Fetch the intended matching selection with IDs, stored vectors and text; paginate as needed. Map the text payload field to `text`.
4. Call `register_records` with those records, then `run_analysis` with the DBSCAN/BERTopic plan.
5. Poll `get_job`, read summaries/examples/result pages, and render the returned coordinates using the chat client's plotting tools.

```text
Your vector database → IDs + stored vectors + text
                    → register_records → run_analysis → get_job
                    → clusters + BERTopic terms + plot coordinates
```

The retrieval integration must know the collection's embedding model; matching vector dimensions alone does not establish that the embedding spaces match. To analyze the whole collection, fetch all stored records directly; no query embedding is needed. For another database, its retrieval integration supplies the same record format.

Register a retrieved subset as `retrieval_results`; use `complete_selection` when the entire intended selection is supplied. Counts describe the records actually analyzed. Respect the worker's capabilities and limits, and report exceeded limits rather than silently truncating the selection. Large input uses [staged JSONL uploads](docs/mcp.md#connection-options).

**The current core MCP has no database connection or search tool.** It does not list collections or generate query embeddings. A connection string passed to `register_records` will not connect a database; that tool accepts vector records or a server-staged JSONL filename. The [MCP guide](docs/mcp.md#natural-language-chat-requests) explains the retrieval prerequisites and exact analysis calls.

## All MCP tools

These are tool calls made by your agent, not terminal commands. Optional arguments are marked `?`.

| Tool | Arguments | What it returns |
| --- | --- | --- |
| `get_capabilities` | None | Algorithms, supported parameters and server limits |
| `register_records` | `records` **or** `artifact_name`; `selection_scope?` | `dataset_id` and registered count |
| `run_analysis` | `dataset_id`, `operations`; `execution?`, `idempotency_key?` | Background job with `job_id` |
| `get_job` | `job_id` | Status/progress; result IDs on success or an error on failure |
| `cancel_job` | `job_id` | Actual job outcome after cancellation |
| `get_summary` | `result_id` | Compact counts, parameters, scope and provenance |
| `get_examples` | `result_id`; `cluster_id?`, `limit?` (default 5, range 1–100) | Representative record IDs and text excerpts |
| `get_result_page` | `result_id`; `cursor?`, `limit?` (default 100, range 1–1,000) | Assignment/coordinate or topic rows, plus `next_cursor` |

See [exact JSON calls for every tool and the complete workflow](docs/mcp.md). Summaries are compact; use result pages for the full output. DBSCAN keeps up to five representative examples per cluster. Cluster `-1` is unassigned data.

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

Poll `get_job` using the returned `job_id`. On success, use `results.groups` as the result ID for summaries, examples and pages. Add a `topics` operation referencing `groups` to include BERTopic; the [complete plan](docs/mcp.md#3-run_analysis) shows that call.

## Try the BANKING77 demo

[BANKING77](https://huggingface.co/datasets/mteb/banking77) contains 13,069 banking questions across 77 intents. The setup uses the full train and test splits, generates MiniLM embeddings once, and indexes them in Qdrant. No OpenAI API key is needed.

Clone the repository for its example scripts and Docker configuration, then install the demo dependencies with pip:

```bash
git clone --branch feature/banking77-dbscan https://github.com/EigenLake-Org/vector-tools.git
cd vector-tools
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
codex mcp add vector-tools-banking \
  --env "VECTOR_ML_DATA=$PWD/.vector-ml/banking77" \
  -- vector-ml --config "$PWD/config/banking77.json"
```

Start a new session with that server enabled, then ask:

> Use vector-tools-banking to register banking77.jsonl as a complete selection. Cluster all records with DBSCAN search, then model their topics with BERTopic. Wait for completion and show cluster sizes, unassigned records, topic terms and examples.

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
- [Record, analysis and upload contracts](docs/contracts.md)
- [Architecture](docs/architecture.md)

For development: `pip install -e '.[server,banking,test]'` then `pytest -q`.
