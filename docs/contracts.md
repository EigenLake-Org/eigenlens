# Analysis and result contracts

## Portable requests

Schemas use Pydantic with unknown fields rejected and `schema_version: "1.0"`. Operations accept installed algorithm names and typed JSON parameters; remote requests cannot supply Python estimators, import paths or executable code. A plan has 1–16 uniquely named operations. Dependencies are validated and ordered before launching a worker. Topics require a cluster dependency; sentiment may aggregate by a cluster dependency.

```json
{
  "dataset_id": "ds_HANDLE",
  "operations": [
    {"id": "groups", "type": "cluster", "algorithm": "minibatch_kmeans", "parameters": {"n_clusters": 30, "random_state": 42}},
    {"id": "themes", "type": "topics", "clusters_from": "groups", "representation": "ctfidf"},
    {"id": "mood", "type": "sentiment", "model": "configured-default", "clusters_from": "groups"},
    {"id": "odd", "type": "outliers", "algorithm": "isolation_forest"}
  ],
  "execution": {"mode": "async", "backend": "auto"},
  "idempotency_key": "application-generated-stable-key"
}
```

`auto` chooses local when every operation is supported. Spark and unregistered algorithms fail with an actionable alternative; no algorithm is silently changed. The service supports asynchronous requests; synchronous execution uses the direct SDK. The SQL/example source fixture has only 62 records, so choose cluster counts accordingly.

## Algorithms

### Clustering

MiniBatchKMeans runs on stored dense embeddings. Default cluster count is `min(max_clusters, max(1, min(8, floor(sqrt(N)))))`. Defaults include seed 42 and `n_init=3`; callers can override supported parameters. Requested clusters must not exceed usable records or the configured cluster limit. `normalize=true` applies L2 normalization before Euclidean clustering; otherwise Euclidean distance uses supplied values, even if the source uses another search metric. Effective sklearn parameters and that assumption are recorded.

Results retain assignments, sizes, centroids and inertia. Representatives are the five closest assigned records to each centroid, with ties broken by ID. MCP summaries omit full centroids and return their shape; full centroids remain in the result artifact. Python convenience methods accept exact `MiniBatchKMeans` instances and preserve their configuration. Unregistered custom estimator subclasses are rejected instead of being misrepresented as portable algorithms.

### Topic representation

Text is concatenated by cluster. CountVectorizer extracts up to 20,000 terms, with English stop words by default. c-TF-IDF is:

`L1-normalized class term frequency × log(1 + average class token count / total corpus frequency of the term)`.

This is group representation, not a separate clustering algorithm. Terms are ordered by weight with deterministic lexical tie-breaking. Default `top_n=10`. Empty/stopword-only vocabularies return empty terms and `empty_vocabulary`; missing text is counted. Clustering fingerprints and record-ID coverage must match the materialized dataset. MCP summaries show at most 20 groups and 10 terms per group; use result pages for complete term outputs.

### Sentiment

VADER 3.3.2 is an English lexicon-and-rule sentiment model. The optional `sentiment` extra installs its bundled lexicon; inference requires no API or network download. If the package or lexicon is missing, capability discovery reports it and sentiment requests fail before a worker is launched. The model identity, package version and lexicon-content SHA-256 are recorded.

Text is processed in configurable batches (default 128). Compound scores range from −1 to +1. Positive means compound ≥ 0.05; negative means compound ≤ −0.05; otherwise neutral. Thresholds are configurable. Positive, neutral and negative scores are lexical proportions, **not calibrated probabilities**. Embedding similarity is not used for sentiment. This baseline has English and domain limitations and is not a transformer classifier.

Missing/blank text is skipped. Cluster aggregates report the count of scored vector records as denominator, missing text count, label counts and mean compound score. Empty groups have a null mean. Parent documents are not deduplicated.

### Outliers

IsolationForest preserves `score_samples` as `raw_score` and publishes `anomaly_score = -raw_score`. Higher public scores mean more anomalous. Results also retain `decision_function`; negative values count as flagged by the fitted threshold. Rows sort by descending anomaly score, then record ID. Default seed is 42 and service inference uses one local job; `normalize` is explicit. Exact `IsolationForest` instances are accepted in the direct Python convenience method.

## Result contracts

Each `Result` contains its result ID, run/job ID, operation ID/type, dataset fingerprint, source identity, scope, processed/skipped counts, effective parameters, package versions, timestamp, backend, summary, full rows and selected text examples. Full outputs are plain JSON; no sklearn or Spark classes appear in public schemas.

All counts refer to vector records. Parent IDs are retained and unique non-null parent counts are reported for context, but there is no implicit parent-ticket aggregation. Missing requested IDs are skipped and counted. Invalid or missing vectors required by an operation, incompatible dimensions and non-finite numbers reject execution. Text-only sentiment can operate on records without embeddings. Empty jobs fail explicitly; direct `.materialize()` may return an empty dataset for inspection.

`get_summary` returns compact evidence; `get_examples` returns 1–100 examples with text truncated to 1,000 characters; `get_result_page` returns up to 1,000 output rows. Cursors include immutable result identity and fingerprint, and are rejected for other results. Source record IDs are preserved in assignments, sentiment and outlier rows. Dataset artifacts retain the complete selected text/metadata. A result handle is inaccessible to another principal and unavailable artifacts produce explicit errors.

## Limits

| Limit | Default |
|---|---:|
| Retained selected records | 10,000 |
| Scanned records | 100,000 |
| Selected serialized content | 64 MiB |
| Total serialized scan content | 4 × selected byte budget |
| Vector dimensions | 65,536 maximum contract length |
| Estimated ML working allocation | 256 MiB |
| Source batch | 256 (configurable 1–1,000) |
| Clusters | 256 |
| Concurrent workers | 2 |
| Queued + running jobs | 32 |
| Worker time including startup/source reads | 300 seconds |
| HTTP request body | 2 MiB |

Limits apply before unbounded selection collection. Sampling uses a bounded heap. Dense matrix copies and cluster-distance allocations receive additional estimates; the Docker service also has a 1 GiB container limit. Estimates are not an exact Python RSS guarantee; source SDKs can allocate one response batch before the dataset layer checks it, and sparse text/model overhead varies. Operators should keep batch sizes and container limits appropriate for their record dimensions and text lengths. Large or distributed workloads must use a future installed backend, not disable all limits.

## Artifact registration

Caller applications can invoke `register_records` with bounded records. For larger inputs, stage UTF-8 JSONL under `<data_dir>/ingress/<caller>/<filename>.jsonl`, then invoke `register_records(artifact_name="filename.jsonl")`. The service enforces its row/byte limits and only accepts a simple filename resolved inside that caller's ingress directory. URLs, arbitrary paths and another service's opaque handles are not resolved.

Registration defaults to `selection_scope="retrieval_results"`; applications may explicitly choose `complete_selection` for a full caller-provided selection. Do not claim a top-k result set represents an entire collection. Operators mounting external artifacts are responsible for the provenance of supplied content.

## Idempotency and interruption

A nonempty per-request idempotency key is unique within the caller. An identical request returns the original job. Reusing the key with a changed request raises an error. Retries of a completed, failed or cancelled job do not create another run. To intentionally rerun, use a new key.

Completed result artifacts are immutable and survive restart. Interrupted work is marked failed; restarting the service does not restart ML automatically. Cancellation reports the actual terminal outcome only after the worker has stopped. Artifact retention/expiration and garbage collection are operator responsibilities in this MVP; no automatic expiration is currently enabled.

## Primary implementation references

- [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [IsolationForest score conventions](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [BERTopic c-TF-IDF weighting](https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html)
- [VADER reference implementation](https://github.com/cjhutto/vaderSentiment)
- [Weaviate iteration](https://docs.weaviate.io/weaviate/manage-objects/read-all-objects)
- [pgvector](https://github.com/pgvector/pgvector)
- [Pinecone ID listing](https://docs.pinecone.io/guides/manage-data/list-record-ids)

### HTTP JSONL upload

Authenticated clients can stream `POST /ingress/<filename>.jsonl` with `Content-Type: application/x-ndjson`, using the same bearer token as MCP. The filename must match `[A-Za-z0-9_-]{1,128}.jsonl`. Uploads are limited by `limits.max_bytes` while streaming; this endpoint does not use the smaller 2 MiB MCP RPC limit. It publishes the file atomically only after the complete stream arrives and never replaces an existing artifact. Responses are 201 on success, 400 for invalid names/lengths, 401 for unauthorized requests, 409 for an existing file, 413 for excessive bytes, and 415 for the wrong content type.

Then call `register_records(artifact_name="filename.jsonl", selection_scope="retrieval_results")` or `complete_selection`, as appropriate. Registration validates the JSONL records and copies them into a durable dataset artifact. Finally use authenticated `DELETE /ingress/<filename>.jsonl` to remove the staging file; deletion does not remove the registered dataset. Both routes resolve only inside the authenticated caller's ingress directory. An interrupted upload removes its temporary file. Abandoned completed uploads and registered artifacts require operator cleanup; retention quotas across files are not implemented.

Worker result summaries include `execution_host` and `worker_pid` for diagnosing where computation ran. Container workers still report the `local` compute backend because the algorithms run locally within that container.
