# Vector Tools

Cluster existing embeddings and label their topics through Python or an MCP job server. [Vector Chat](https://github.com/EigenLake-Org/vector-chat) adds Qdrant search, chat threads and interactive plots.

- **Clustering:** PCA + DBSCAN search, or MiniBatch K-means.
- **Topics:** c-TF-IDF keywords and phrases, with representative examples.
- **Jobs:** queueing, cancellation and saved results. Default: **1 worker**; configurable up to 16.

## Try the BANKING77 demo

[BANKING77](https://huggingface.co/datasets/mteb/banking77) contains 13,069 banking questions across 77 intents. The setup uses the full train and test splits, generates MiniLM embeddings once, and indexes them in Qdrant. No OpenAI API key is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -c requirements-tested.txt -e '.[server,banking]'
docker compose --profile qdrant up -d --wait qdrant
python -m examples.banking77 setup
```

The example keeps a local embedding cache and records dataset/model revisions and a checksum. Repeating setup reuses that cache and upserts the same point IDs. It rejects a collection containing a different dataset snapshot. Run `python -m examples.banking77 prepare` to rebuild the cache. Dataset files and vectors are not committed to Git.

To explore the data, keep `vector-chat` next to this checkout:

```bash
cd ../vector-chat
./scripts/run-docker-chat.sh             # one worker
# ./scripts/run-docker-chat.sh --workers 3
```

Open [Vector Chat](http://127.0.0.1:7860), create a chat with `http://localhost:6333`, and select **banking77_minilm**. Try `show clusters in ATM`, `show clusters in credit cards`, or `show clusters` for the full collection.

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

Run `python examples/sdk_demo.py` for a small synthetic example, or `python examples/mcp_client.py` for the native MCP workflow. Agents register records, submit a plan, poll its job, and read summaries, examples and result pages. Vector Chat handles database retrieval; the core worker handles analysis.

## Documentation

- [Docker worker setup and concurrency](docs/chat-worker.md)
- [MCP tools, uploads and analysis contracts](docs/contracts.md)
- [Architecture](docs/architecture.md)

For development: `pip install -e '.[server,banking,test]'` then `pytest -q`.
