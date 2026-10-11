# Clustering contracts

## Records and scopes

A record contains `id` (unique string), `vector` (finite, nonempty dense array), optional `text`, `metadata`, and `parent_id`. Clustering requires vectors of the same dimension on every selected record. Text is optional; absent text produces empty terms or a missing-text count. Counts refer to vector records; parent IDs do not deduplicate records.

`register_records` accepts exactly one of an inline `records` list or a simple `.jsonl` `artifact_name` staged in `ingress/<caller>/`. Registration rejects duplicate IDs, unknown fields, excessive rows/bytes and invalid scopes. `selection_scope` defaults to `retrieval_results`; use `complete_selection` only when the caller supplies the entire intended selection. The core never infers whether a database retrieval is complete.

The SDK additionally provides bounded in-memory `.ids()`, `.filter()` and seeded `.sample()`. Sampling is explicit and deterministic by record ID. Materialization fails on exceeded limits instead of silently truncating.

## Plans

```json
[
  {"id":"groups","type":"cluster","algorithm":"dbscan_search"},
  {"id":"themes","type":"topics","clusters_from":"groups","parameters":{"ngram_max":2}}
]
```

Operation IDs must be unique. A plan contains 1–16 operations; topic operations must reference a cluster operation in that plan. The registry orders dependencies. Unknown operations, algorithms, fields, parameters and unavailable backends fail validation before queueing. Service execution is asynchronous and local; use the Python SDK for synchronous calls.

| Operation | Algorithm | Parameters |
| --- | --- | --- |
| `cluster` | `minibatch_kmeans` (default) | `n_clusters`, `random_state`, `batch_size`, `n_init`, `max_iter`, `normalize` |
| `cluster` | `dbscan_search` | `pca_components`, `eps_values`, `min_samples_values`, `max_noise_ratio`, `silhouette_sample_size`, `random_state`, `label_field` |
| `topics` | `ctfidf` | `top_n`, `stop_words`, `ngram_max` |

DBSCAN L2-normalizes embeddings before PCA and normalizes the reduced coordinates afterward. Defaults search eps 0.30–0.80 in 0.05 increments and min_samples 5, 10, 15, with at most 64 grid combinations. The winning candidate has at least two clusters, no more than `max_clusters`, noise strictly below the threshold (default 70%), and the highest sampled silhouette among non-noise points. Ties use the first grid entry. No valid candidate returns diagnostics with all rows unassigned (`-1`), not a fabricated clustering. Optional metadata labels are used only for post-hoc evaluation.

The 3D PCA projection is separate from the clustering space and preserves all record IDs. c-TF-IDF weights terms within each cluster; phrases never cross record boundaries. `ngram_max` supports 1–3. Noise may have terms, but does not represent a discovered topic. Terms and clusters are exploratory evidence.

## MCP tools and results

| Tool | Purpose |
| --- | --- |
| `get_capabilities` | Algorithms, parameters, backend and active limits |
| `register_records` | Persist caller-owned input snapshot |
| `run_analysis` | Validate and enqueue; return job handle |
| `get_job` | Status, error or operation-to-result handles |
| `cancel_job` | Stop queued/running job, preserve already completed result |
| `get_summary` | Compact counts, scope, parameters and provenance |
| `get_examples` | Up to 100 representative IDs and excerpts |
| `get_result_page` | Up to 1,000 rows and a result-bound cursor |

Jobs progress through `queued`, `running`, then `succeeded`, `failed` or `cancelled`. Reusing an owner-scoped `idempotency_key` with the identical request returns the same job; a different request fails. Application code should poll with backoff. A full queue rejects a submission before creating a job.

Results contain immutable IDs, dataset fingerprints, effective parameters, library versions, source scope and count semantics. Worker results include execution hostname and PID. Summaries omit centroids and show at most 20 topic groups with 10 terms each; page results to obtain full output. Pages for DBSCAN contain `id`, `cluster_id` and `pca_3d`; topic pages contain weighted terms. Cursors cannot be reused for another result. Dataset, job and result handles are caller-scoped.

## HTTP JSONL upload

MCP uses `/mcp`. Inline RPC requests are capped at 2 MiB. For larger input:

1. POST newline-delimited record JSON to `/ingress/<unique-name>.jsonl` with `Content-Type: application/x-ndjson` and bearer authentication.
2. Call `register_records` with that `artifact_name` and the appropriate scope.
3. DELETE the staging URL after registration, including when registration fails.

Upload names accept letters, digits, underscores and hyphens, followed by `.jsonl`. No external URLs or arbitrary paths are accepted. Uploads enforce `max_bytes` while streaming and publish only complete files. Existing filenames return 409. Different callers have separate staging directories. Registration validates records and row limits; malformed input does not create a dataset. Staging cleanup does not delete the registered snapshot.

HTTP authentication maps bearer tokens to configured callers. Non-loopback startup requires a token. `/health` is public and reports supervisor health and configured worker slots. Loopback HTTP permits token-free local access only when no token is configured. Stdio uses the configured local caller without HTTP authentication.

## Limits and persistence

Default SDK/service limits are 10,000 rows, 64 MiB serialized input, 256 MiB estimated working memory, one concurrent job, 32 pending jobs, and a 300-second timeout. `config/chat-worker.json` raises these to 20,000 rows, 512 MiB input, 8 GiB working memory, 16 pending jobs and 1,800 seconds, retaining one concurrent job by default.

`VECTOR_ML_WORKERS` supports 1–16 slots. Optional `VECTOR_ML_TOTAL_WORKING_BYTES` divides a shared estimate across slots. Compose passes an 8 GiB shared estimate and enforces an 8 GiB container limit. Memory estimates are conservative guards, not exact RSS guarantees; DBSCAN's quadratic worst case can reject large selections. Timeouts and cancellation terminate the process. Two supervisors cannot share a data directory. Restart marks incomplete jobs failed and preserves completed artifacts. No automatic retention policy exists.
