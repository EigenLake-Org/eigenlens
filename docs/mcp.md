# MCP tools and chat usage

Connect an MCP-capable chat agent to this server, then retrieve or supply records and ask it to analyze them. There are 13 tools: five [database connection/retrieval tools](databases.md#exact-database-tool-calls) and eight analysis tools documented below. MCP tool names are not shell commands. The agent discovers them with MCP `tools/list` and invokes them with `tools/call`.

## Connection options

### Local stdio

Follow the [README pip installation and Codex registration](../README.md#installation). The chat host launches the server and manages its stdin/stdout connection. Configure other local hosts with:

- Executable: the installed `vector-ml` command (use its absolute path if the host cannot resolve it on PATH).
- Arguments: none for the default configuration.
- Environment: `VECTOR_ML_DATA=<absolute-path-to-your-server-data-directory>`.
- Caller: `local` (the stdio default).

Use one server per data directory. The default configuration has a 10,000-record limit. For the full banking corpus, follow [BANKING77 setup and chat registration](../README.md#try-the-banking77-demo): pass `--config <absolute-repo-path>/config/banking77.json` and set `VECTOR_ML_DATA=<absolute-repo-path>/.vector-ml/banking77`. Its prepared artifact is `<VECTOR_ML_DATA>/ingress/local/banking77.jsonl`.

### Shared Docker HTTP server

Start the [Docker worker](worker.md#docker). Its MCP endpoint is `http://127.0.0.1:8020/mcp`. Set `VECTOR_ML_MCP_TOKEN` in the chat host's environment to the **same token** configured as `VECTOR_CHAT_WORKER_TOKEN` when starting Docker. Then register it in Codex:

```bash
codex mcp add vector-tools-http \
  --url http://127.0.0.1:8020/mcp \
  --bearer-token-env-var VECTOR_ML_MCP_TOKEN
codex mcp list
```

The chat host must inherit that environment variable. Other clients use Streamable HTTP with `Authorization: Bearer <token>`. The URL must be reachable from the agent's execution environment; localhost refers to the machine running that client. See the [official Codex MCP command reference](https://learn.chatgpt.com/docs/developer-commands#codex-mcp).

The local BANKING77 cache is on your host. Stage it through the upload route before asking the Docker server to register it:

```bash
# Run from the repository directory; use a unique filename per upload.
upload_name="banking77-$(python3 -c 'import uuid; print(uuid.uuid4().hex)').jsonl"
curl --fail-with-body \
  -H "Authorization: Bearer $VECTOR_ML_MCP_TOKEN" \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @.vector-ml/banking77/ingress/local/banking77.jsonl \
  "http://127.0.0.1:8020/ingress/$upload_name"
```

Tell the agent the returned `artifact_name`; use that filename in `register_records`. After registration, including a failed attempt, delete the upload staging file:

```bash
curl --fail-with-body -X DELETE \
  -H "Authorization: Bearer $VECTOR_ML_MCP_TOKEN" \
  "http://127.0.0.1:8020/ingress/$upload_name"
```

This deletes staging input, preserving any successfully registered dataset. Uploading is an HTTP operation, not an MCP tool. For automated clients, put cleanup in a `finally` block. Inline MCP HTTP requests have a 2 MiB limit; streamed uploads use the configured input-byte limit.

## Complete workflow

1. Inspect `get_capabilities` for supported operations and active limits.
2. Connect/list/retrieve with the [database tools](databases.md) to obtain `dataset_id`, or supply inline/staged records with `register_records`.
3. Call `run_analysis` with that dataset and the clustering/topic plan. Save its `job_id`.
4. Poll `get_job` with increasing delays until `succeeded`, `failed` or `cancelled`. Let application code handle polling for long jobs.
5. On success, take `results.groups` and `results.themes` from the job response. These are separate result IDs.
6. Read summaries and examples. Follow `get_result_page.next_cursor` until it is `null` for every result you want to export or plot.

Job states are `queued` → `running` → `succeeded`, `failed` or `cancelled`. Report terminal errors before attempting to read results. Handles are scoped to the caller and server data directory. Keep returned handles in the chat or application state; there is no tool for listing old jobs or datasets.

## Exact calls for the eight analysis tools

The JSON below is the payload for MCP `tools/call`: `name` identifies the tool and `arguments` holds its inputs. Your agent normally sends it automatically. Replace `ds_example`, `job_example`, `res_groups_example`, `res_themes_example` and cursor placeholders with values returned by your server.

### 1. get_capabilities

```json
{"name": "get_capabilities", "arguments": {}}
```

Returns operation schemas, algorithms, supported parameters, compute backends and active limits. Inspect this before making a plan.

### 2. register_records

For the prepared local banking artifact:

```json
{
  "name": "register_records",
  "arguments": {
    "artifact_name": "banking77.jsonl",
    "selection_scope": "complete_selection"
  }
}
```

For a small selection supplied by a retrieval tool, pass `records` instead of `artifact_name`:

```json
{
  "name": "register_records",
  "arguments": {
    "records": [
      {"id": "record-1", "vector": [0.1, 0.2, 0.3], "text": "My ATM withdrawal failed"},
      {"id": "record-2", "vector": [0.2, 0.1, 0.3], "text": "The cash machine kept my card"},
      {"id": "record-3", "vector": [0.3, 0.2, 0.1], "text": "Cash withdrawal was declined"}
    ],
    "selection_scope": "retrieval_results"
  }
}
```

These three records demonstrate the input format. Real clustering needs a sufficiently large selection; this tiny input will not produce two valid clusters with the default search. Vectors must already exist, share a dimension and use the same embedding space. Include text for topic terms and readable examples. Optional record fields are `metadata` and `parent_id`.

Use exactly one of `records` or `artifact_name`. An artifact name is a simple server-staged `.jsonl` filename, not a client file path, Qdrant collection name or URL. Registration returns the dataset ID and registered count. It does not start an analysis.

`selection_scope` defaults to `retrieval_results`. Use `complete_selection` when you supply the entire intended selection; this flag does not retrieve more records or check database completeness.

### 3. run_analysis

```json
{
  "name": "run_analysis",
  "arguments": {
    "dataset_id": "ds_example",
    "operations": [
      {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
      {
        "id": "themes",
        "type": "topics",
        "algorithm": "bertopic",
        "clusters_from": "groups",
        "parameters": {"ngram_max": 2, "top_n": 10}
      }
    ],
    "execution": {"mode": "async", "backend": "local"},
    "idempotency_key": "banking-analysis-request-001"
  }
}
```

Returns a background job with `job_id`. `groups` and `themes` are operation names you choose; they become keys in the completed job's result map. Topics use DBSCAN assignments through BERTopic's manual workflow. The worker uses supplied embeddings.

`execution` is optional; the service supports asynchronous local execution. `idempotency_key` is optional: reuse a stable key for retries of an identical request, and choose a new key for a different request. A reused key with changed arguments fails. A plan accepts 1–16 operations with unique IDs. A topic operation references a cluster operation in the same plan through `clusters_from`.

| Operation | Optional parameters |
| --- | --- |
| `cluster` / `dbscan_search` | `pca_components`, `eps_values`, `min_samples_values`, `max_noise_ratio`, `silhouette_sample_size`, `random_state`, `label_field` |
| `topics` / `bertopic` | `ngram_max` (1–3), `top_n` (1–100), `stop_words` (`"english"`) |

You can omit custom clustering parameters and use the search defaults. See [analysis contracts](contracts.md#plans) for selection rules. `label_field` is metadata used only for post-hoc evaluation, not cluster formation.

### 4. get_job

```json
{"name": "get_job", "arguments": {"job_id": "job_example"}}
```

Returns the existing `status` (`queued`, `running`, `succeeded`, `failed`, `cancelled`) plus detailed `info`:

```json
{
  "job_id": "job_example",
  "status": "running",
  "progress": {"phase": "running", "fraction": null},
  "info": {
    "stage": "dbscan_search",
    "message": "Evaluating DBSCAN parameters and silhouette scores",
    "operation_id": "groups",
    "operation_index": 1,
    "operation_count": 2,
    "completed": 12,
    "total": 42,
    "unit": "trials",
    "eps": 0.5,
    "min_samples": 5,
    "updated_at": "2026-10-18T10:30:00+00:00"
  }
}
```

Stages cover loading/materializing records, normalization, PCA, neighbor graph construction, DBSCAN parameter search, cluster summaries, BERTopic loading/extraction, topic summaries, and saving results. `info` is the latest persisted stage, not an event log; short stages may finish between polls. Counters appear only when measurable and describe that stage, not overall percentage or remaining time. `operation_index` is one-based. Additional numeric details vary by stage.

Poll `get_job` in application code (for example, once per second) and display `info.message` plus any counters. Keep using `status` to decide when to stop polling. A failed/cancelled job retains its last stage and has a separate `error`; a successful job reports `completed`. Cache hits report `reusing_results` with `status: "succeeded"` and `cache_hit: true`, without running worker stages. Existing jobs without detailed information return a status-based message. Stages are stored alongside job metadata and remain owner-scoped.

On success, `results` maps operation IDs to result IDs, for example:

```json
{"groups": "res_groups_example", "themes": "res_themes_example"}
```

Save both IDs; cluster examples/assignments use the groups result, while topic terms use the themes result. Failed jobs include an error.

### 5. cancel_job

```json
{"name": "cancel_job", "arguments": {"job_id": "job_example"}}
```

Stops the queued/running job and returns its actual outcome. A job that already completed keeps its results. Inspect the returned status.

### 6. get_summary

```json
{"name": "get_summary", "arguments": {"result_id": "res_groups_example"}}
```

Call again with `res_themes_example` for the topic summary. Summaries contain processed record counts, scope, effective parameters, provenance and operation-specific details. Topic summaries include at most 20 groups and 10 terms per group; pages provide the complete output.

### 7. get_examples

```json
{
  "name": "get_examples",
  "arguments": {"result_id": "res_groups_example", "cluster_id": 0, "limit": 5}
}
```

Choose a cluster ID present in the summary. Omit `cluster_id` to get representative examples across clusters. `limit` defaults to 5 and accepts 1–100, but DBSCAN stores up to five representative examples per cluster. Responses include source IDs and text excerpts of up to 1,000 characters. They are not every member of the cluster.

### 8. get_result_page

First page:

```json
{
  "name": "get_result_page",
  "arguments": {"result_id": "res_groups_example", "limit": 1000}
}
```

Next page, using the returned cursor:

```json
{
  "name": "get_result_page",
  "arguments": {
    "result_id": "res_groups_example",
    "cursor": "cursor_from_previous_page",
    "limit": 1000
  }
}
```

The response contains `rows`, `total` and `next_cursor`. Continue until `next_cursor` is `null`; do not reuse a cursor for another result. `limit` defaults to 100 and accepts 1–1,000.

- Cluster rows: record `id`, `cluster_id`, and three coordinates in `pca_3d`.
- Topic rows: `cluster_id` and weighted `terms`.

For a plot, the agent/client collects every cluster page, colors points by `cluster_id`, and renders `pca_3d` using its chart tools. The three plotted PCA coordinates are a visualization; clustering uses its own reduced representation. Cluster `-1` is unassigned data. If no valid DBSCAN candidate exists, report that outcome from the summary.

## Natural-language chat requests

After the [BANKING77 setup and dedicated chat connection](../README.md#use-the-banking-dataset-in-chat), start with:

> Use vector-tools-banking to register banking77.jsonl as a complete selection. Cluster all records with DBSCAN search, then use BERTopic to find keywords and two-word phrases for those clusters. Wait for completion and show the cluster counts, unassigned count, topic terms and examples.

Useful follow-ups:

> What is the status of that job?

> Which clusters look related to cash withdrawals? Inspect topic terms and representative source examples before explaining them.

> Show five source examples from cluster 0.

> Export every assignment and render an interactive 3D cluster plot using all returned coordinates.

> Stop the running analysis and report its final status.

For **“show clusters in ATM” over your vector database**, configure a [database connection](databases.md#where-settings-live). The agent calls `connect_database`, `list_collections`, then `retrieve_records` with the selected connection/collection and query. Retrieval registers stored vectors/text and returns the dataset ID used by `run_analysis`. The server resolves a SentenceTransformer model from configuration or stored metadata; externally generated query vectors support other embedding providers. It fails when a text query's model is unknown.

Semantic retrieval is a bounded ranked selection; report its count and `search_limit`. To analyze the complete collection, omit query inputs and retrieve all pages, subject to server limits. Asking about ATM-related clusters in an already analyzed full corpus can inspect its topics/examples; rerunning clustering specifically on ATM results uses a new retrieved snapshot.

## Python MCP client smoke example

```bash
python examples/mcp_client.py
```

This exercises stdio MCP with 62 synthetic records and prints discovery plus the complete call trace. It does not use a chat model or the banking dataset. To exercise an existing Docker server instead, set the matching bearer token in `VECTOR_ML_MCP_TOKEN` and run:

```bash
VECTOR_ML_TRANSPORT=http \
VECTOR_ML_MCP_URL=http://127.0.0.1:8020/mcp \
python examples/mcp_client.py
```
