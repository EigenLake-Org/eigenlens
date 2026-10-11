# Vector Tools

Cluster existing embeddings and model their topics through Python or an MCP job server.

- **Clustering:** PCA + DBSCAN parameter search.
- **Topics:** [BERTopic](https://maartengr.github.io/BERTopic/getting_started/manual/manual.html) keywords and phrases from the discovered clusters.
- **Jobs:** queueing, cancellation and saved results. Default: **1 worker**; configurable up to 16.

## Try the BANKING77 demo

[BANKING77](https://huggingface.co/datasets/mteb/banking77) contains 13,069 banking questions across 77 intents. The setup uses the full train and test splits, generates MiniLM embeddings once, and indexes them in Qdrant. No OpenAI API key is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -c requirements-tested.txt -e '.[server,banking]'
docker compose --profile qdrant up -d --wait qdrant
python -m examples.banking77 setup
python -m examples.banking77 analyze
```

The example keeps a local embedding cache and records dataset/model revisions and a checksum. Repeating setup reuses that cache and upserts the same point IDs. It rejects a collection containing a different dataset snapshot. Run `python -m examples.banking77 prepare` to rebuild the cache. Dataset files and vectors are not committed to Git.

Analysis saves assignments, topic terms and an interactive 3D plot in `demo-output/banking77/`. It processes the complete cached corpus. Clustering uses existing vectors; BERTopic uses those assignments to model topics without downloading an embedding model. `-1` remains unassigned data.

## Use Python or MCP

```python
import vector_tools as vt

data = vt.from_records(records).materialize()  # id, vector, optional text
results = vt.analyze(data, [
    {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
    {"id": "themes", "type": "topics", "clusters_from": "groups",
     "parameters": {"ngram_max": 2}},
])
```

Run `python examples/sdk_demo.py` for a small synthetic example, or `python examples/mcp_client.py` for the native MCP workflow. Agents register records, submit a plan, poll its job, and read summaries, examples and result pages. Callers retrieve their own records; the worker runs PCA/DBSCAN and BERTopic.

## Documentation

- [Docker worker setup and concurrency](docs/worker.md)
- [MCP tools, uploads and analysis contracts](docs/contracts.md)
- [Architecture](docs/architecture.md)

For development: `pip install -e '.[server,banking,test]'` then `pytest -q`.
