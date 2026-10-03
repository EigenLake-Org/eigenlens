# BANKING77: DBSCAN and agent usage

This example implements the supplied BANKING77 workflow: MiniLM embeddings → L2 normalization → PCA (up to 50 dimensions) → L2 normalization → DBSCAN grid search → mean TF-IDF terms → intent evaluation → interactive 3D PCA plot and CSV exports.

The original syntax error is corrected. Missing `label_text` falls back to ClassLabel names or numeric labels. A search with no qualifying candidate returns `no_valid_candidate` with all grid diagnostics, rather than crashing or silently choosing an unsuitable result. Those rows are marked unassigned (`-1`); they are not an accepted DBSCAN solution.

## Try it

From the repository root, with Python 3.11+ and internet access for installation and first download:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[banking,server,agents]'

# Quick seeded sample; embeddings are computed locally (no OpenAI key).
python -m examples.banking77 prepare --limit 2000
python -m examples.banking77 analyze

# Or reproduce the train + test workflow on all records:
python -m examples.banking77 prepare
python -m examples.banking77 analyze

# Exercise the same calculation through a real local MCP server:
python -m examples.banking77 mcp
```

Preparation stores records and embeddings once under `.vector-ml/banking77/ingress/local/`. Later analysis reuses that file after checking its SHA-256 checksum. Dataset and model refs resolve to immutable Hugging Face commits, recorded in the manifest. To reproduce a prior download, pass `--dataset-revision <sha>` and `--model-revision <sha>` to `prepare`.

The dedicated `config/banking77.json` allows 15,000 rows, a conservative 6 GiB working-memory budget, one concurrent service job, and a 30-minute service timeout. This is an upper budget, not a reservation; full-data DBSCAN can be expensive. Use `--limit 2000` on smaller machines. The SDK runs synchronously; service jobs provide timeout and cancellation. No existing Qdrant collection is modified.

## Output

Under `demo-output/banking77/`:

- `banking77_dbscan_3d.html`: standalone interactive plot, bundled Plotly JS.
- `banking77_cluster_summary.csv`: sizes, terms, dominant intents and purity.
- `banking77_dbscan_parameter_search.csv`: every tested setting and its diagnostics.
- `banking77_clustered.csv`: record IDs, queries, intents, assignments and coordinates.
- `banking77_cluster_vs_intent.csv`: cluster/intent contingency table.
- `banking77_result.json`: portable result, parameters, evaluation and provenance.
- `banking77_manifest.json`: data/model revisions and prepared-data checksum.

## How the OpenAI agent uses it

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-model-id'
python -m examples.banking77 agent
```

The host stages the prepared embeddings and registers the local artifact through MCP. It passes only the returned dataset ID to the agent, keeping the embedding matrix out of the model context. The agent discovers capabilities and submits this typed plan through `run_analysis`:

```json
{
  "dataset_id": "<registered dataset handle>",
  "operations": [
    {
      "id": "groups",
      "type": "cluster",
      "algorithm": "dbscan_search",
      "parameters": {
        "pca_components": 50,
        "label_field": "label_text",
        "random_state": 42
      }
    }
  ],
  "idempotency_key": "<one stable key per analysis request>"
}
```

The service computes the result in its worker process. The application polls `get_job` without spending model turns, then resumes the agent with completed result handles. The agent calls `get_summary` and `get_examples` to explain actual results. The host pages all assignments using `get_result_page` and creates the local plot/CSVs. The model neither executes the pasted script nor creates the embeddings itself.

The example uses an SDK-managed stdio MCP server, suitable for a local/private service ([official OpenAI guidance](https://developers.openai.com/api/docs/guides/agents/integrations-observability)). Registration reports `retrieval_results` scope: the registered set may be a prepared subset; the manifest records whether train+test or a seeded sample was prepared.

### Use existing Qdrant data

An agent can instead call `create_dataset` with an authorized source ID such as `food_reviews`, then submit the same `dbscan_search` operation. Omit `label_field` when the records have no known intent labels. Qdrant access is resolved by server-owned source configuration, as in the existing demo. The algorithm works on supplied vectors regardless of embedding provider. Do not combine incompatible embedding models in one dataset.

### Python SDK

```python
import vector_tools as vt

# records contain id, vector, text, and optional metadata.label_text
materialized = vt.from_records(records).materialize()
results = vt.analyze(materialized, [{
    "id": "groups", "type": "cluster", "algorithm": "dbscan_search",
    "parameters": {"label_field": "label_text"}
}])
print(results["groups"].summary())
```

Large datasets require explicit `Limits` or a bounded sample. `get_capabilities` advertises algorithm-specific parameter lists. Existing MiniBatchKMeans remains the default.

## Interpretation and limits

- Default grid: eps 0.30–0.80 in steps of 0.05; min_samples 5, 10, 15.
- Select highest non-noise silhouette among candidates with at least two clusters, no more than the configured cluster cap, and strictly less than 70% noise. Ties keep the first grid entry.
- Silhouette uses at most 4,000 seeded non-noise records. A sample containing fewer than two usable clusters has no score. Selection is exploratory, not held-out model validation.
- True intents never influence embeddings, PCA, parameter choice or clustering. ARI/NMI use all rows, including noise, only after selection. Purity describes dominant known intent within each group; it is not a classification accuracy.
- `-1` denotes noise (or unassigned records when no candidate qualifies). It is not a discovered intent.
- TF-IDF terms average document weights within each cluster. This is the supplied script's method, distinct from the existing optional c-TF-IDF `topics` operation.
- Plotting projects normalized embeddings to three dimensions; DBSCAN uses its separate higher-dimensional PCA space.
- The combined train/test run is exploratory. It is not a train/test generalization benchmark.

Sources: [BANKING77 dataset](https://huggingface.co/datasets/mteb/banking77), [MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

## Verification in this implementation session

Available-environment suite: **41 passed, 4 skipped, 6 deselected** (`pytest -q -k "not http"`). The additional Agents SDK test uses a scripted model with real Runner/MCP/worker execution and confirms that a registered dataset is reused for one DBSCAN job.

Algorithm tests cover two separated groups plus noise, label independence, metrics, invalid grids, no valid candidate, empty vocabulary, downstream topics and memory limits. A real stdio MCP test covers discovery, registration, asynchronous execution, summaries, examples and result paging.

The real BANKING77 download, MiniLM inference, Plotly export and live OpenAI call were not executed in this restricted session: additional dependencies could not be downloaded from PyPI. No BANKING77 quality scores are claimed. HTTP tests also require permission to bind a local socket.

## Recommended flow: index → query Qdrant → cluster matches → plot

```bash
pip install -e '.[banking,qdrant]'
./scripts/try-banking-query.sh
# Subsequent queries reuse all indexed vectors:
python -m examples.banking77_qdrant query --query 'general banking support problems' --top-k 2000
```

The script starts the existing Qdrant Compose service, downloads the full BANKING77 train/test corpus and MiniLM model on first use, indexes normalized 384-dimensional embeddings into `banking77_minilm`, runs a general banking query, and clusters the top 2,000 returned vectors with DBSCAN. No OpenAI key is needed. Activate the virtual environment before running the script.

Open `demo-output/banking77-query/banking77_dbscan_3d.html`. The directory also contains `query_report.md` with cluster counts and nearest matches, `query_results.json` with ranked scores and cluster summaries, and all CSV exports. A broad query still returns a relevance-ranked subset; these are not whole-collection clusters.

For an already running Qdrant service:

```bash
python -m examples.banking77_qdrant demo
```

Use `--qdrant-url`, `--collection`, `--query`, `--top-k` (3–2000), and `--output` to customize. The query uses the exact embedding model revision recorded during preparation. IDs are deterministic and upserts are repeatable. A collection holding another dataset/model snapshot is refused; choose a fresh collection name instead of overwriting it.

Agent integration can call the Python `query_and_plot` function from an application-owned tool, or retrieve with `retrieve` and register those returned records through the existing MCP `register_records` tool. The agent then submits `dbscan_search` on that retrieval dataset; the application exports the result. Credentials, Qdrant URLs, and output folders should stay in host configuration. Embedding matrices are passed between Qdrant and the application, not placed in model messages.

### Agent chooses the query and gets clusters plus a plot

After indexing with `examples.banking77_qdrant demo`:

```bash
pip install -e '.[agents]'
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-model-id'
export BANKING_QUERY='Show general banking support clusters'
python -m examples.banking77_query_agent
```

The agent has a typed `search_banking_clusters(query, top_k)` function tool. Host code embeds the query with MiniLM, searches Qdrant, clusters retrieved vectors and writes the plot. The tool returns computed group summaries and the plot path; the agent explains them. No raw embedding vectors or full 2,000-row tables enter the model context. This live-model example requires credentials and is separate from the key-free query CLI.

## Verified Qdrant demo (2026-10-03)

The full end-to-end Qdrant flow has now run successfully: 13,069 BANKING77 records embedded with MiniLM and indexed; the general banking query retrieved 2,000 matches; DBSCAN selected eps=0.55/min_samples=10 and found 25 clusters containing 603 records, with 1,397 noise points (69.85%). Non-noise silhouette was 0.4242. The high noise rate is part of the result, not hidden or forced into clusters. Plotly HTML and all exports were generated. The latest suite passed 48 tests, with five skipped. See [run evidence](evidence/banking77-qdrant.json). Earlier session download restrictions above describe the prior implementation run; they no longer apply to this completed Qdrant run. Live OpenAI model calls remain untested.
