# Vector Tools

Clustering and topic labels for existing vector records, through a Python SDK or an MCP job server. Vector Chat retrieves records from Qdrant and sends them here for analysis.

## What it does

- **PCA + DBSCAN:** bounded parameter search, explicit noise assignments, cluster summaries, and 3D PCA coordinates for plots.
- **MiniBatch K-means:** clustering with a specified cluster count.
- **Topic labels:** c-TF-IDF terms and phrases within each cluster, plus representative record excerpts.
- **Jobs:** process isolation, bounded concurrency (default **1**), queueing, cancellation, persistent results, and paging.

The caller supplies embeddings and optional text. Database connections, query embedding, retrieval, chat UI, and plots belong to [Vector Chat](https://github.com/EigenLake-Org/vector-chat). This core contains only the analysis worker and its contracts.

## Python quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -c requirements-tested.txt -e '.[server,test]'
python examples/sdk_demo.py
pytest -q
```

The demo uses small synthetic support-ticket vectors; it does not download a dataset or embedding model.

```python
import vector_tools as vt

# records: [{"id": "...", "vector": [...], "text": "...", "metadata": {...}}]
data = vt.from_records(records).materialize()
results = vt.analyze(data, [
    {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
    {"id": "themes", "type": "topics", "clusters_from": "groups",
     "parameters": {"ngram_max": 2}},
])
assignments = results["groups"].rows  # id, cluster_id, pca_3d
labels = results["themes"].rows       # cluster_id, weighted terms
```

DBSCAN requires at least three nonzero vectors. If no grid candidate qualifies, the result reports `no_valid_candidate` and marks every record unassigned (`-1`). See [contracts](docs/contracts.md) for limits and selection rules.

## MCP

```bash
python examples/mcp_client.py  # starts an isolated stdio server
```

The eight tools are `get_capabilities`, `register_records`, `run_analysis`, `get_job`, `cancel_job`, `get_summary`, `get_examples`, and `get_result_page`. An agent registers records, submits a clustering/topic plan, then reads results. Application code polls jobs and renders plots from the returned coordinates. Large inputs use authenticated JSONL upload.

## Docker worker

From adjacent `vector-tools` and `vector-chat` checkouts, the chat startup script starts both containers:

```bash
cd ../vector-chat
./scripts/run-docker-chat.sh                # one worker
./scripts/run-docker-chat.sh --workers 3    # three concurrent analysis jobs
```

For the core worker alone:

```bash
export VECTOR_CHAT_WORKER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export VECTOR_CHAT_WORKERS=1
docker compose --profile chat-worker up -d --build --wait chat-worker
```

The MCP URL is `http://127.0.0.1:8020/mcp`; use the generated token as bearer authentication. `/health` reports worker count. For a local Qdrant database, optionally run `docker compose --profile qdrant up -d qdrant`. The worker itself does not connect to it.

One worker container runs a supervisor and up to 1–16 job processes. Counts above one divide its 8 GiB working-memory estimate across jobs; CPU and container memory stay fixed. Start with one for large selections. [Worker setup](docs/chat-worker.md) covers storage and lifecycle.

## Repository

| Path | Purpose |
| --- | --- |
| `vector_tools/density.py` | PCA, DBSCAN search, noise and plot coordinates |
| `vector_tools/operators.py` | K-means and c-TF-IDF labels |
| `vector_tools/jobs.py` | Supervisor, worker processes, cancellation |
| `vector_tools/service.py`, `server.py` | Durable analysis API and MCP transport |
| `config/chat-worker.json` | Worker limits and authentication config |
| `examples/` | SDK and native MCP clustering demos |
| `tests/` | Algorithms, contracts, jobs and transport checks |

This cleanup removes the former sentiment/outlier operations, source-connector APIs, Spark placeholder, Arrow export, and separate ingestion/query demos. Clients must retrieve their data and call `register_records`. Vector Chat already uses this workflow. Existing completed result artifacts remain readable; no data migration or deletion is performed.
