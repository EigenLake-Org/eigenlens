# Database connections and retrieval

The MCP server provides five database tools alongside its eight analysis tools. Adapters normalize stored records to `id`, `vector`, optional `text` and `metadata`. Retrieval registers a durable dataset; PCA/DBSCAN and BERTopic run in the existing isolated job workers.

## Existing Python SDK clients

`el.from_client(client, collection, *, vector_field=None, text_field="text", query_vector=None, search_limit=1000, limits=None)` accepts an existing synchronous `qdrant_client.QdrantClient` or connected `weaviate.WeaviateClient` v4. Install the corresponding `qdrant` or `weaviate` extra. Client type detection imports optional SDKs on demand; no database request occurs until materialization.

```python
import eigenlens as el

# qdrant_client is the QdrantClient your application already created.
data = el.from_client(qdrant_client, "banking77_minilm",
                      limits=el.Limits(max_rows=15000, max_bytes=256 * 1024 * 1024,
                                       max_working_bytes=6 * 1024**3, max_seconds=1800)).materialize()
groups = el.cluster(data)
themes = el.topics(data, groups, ngram_max=2)

# The same API accepts your connected WeaviateClient.
other_data = el.from_client(weaviate_client, "Questions",
                           text_field="content", vector_field="body").materialize()
```

Omit `vector_field` when there is a sole dense field. Text defaults to the `text` payload/property; choose `text_field` for a different name. Database IDs become record IDs. Payloads/properties become metadata. No new embedding is generated, and the library does not read credentials from the client's internals or save a connection definition.

### Search, then cluster

Supply an externally generated query vector in the collection's embedding space:

```python
# query_vector comes from the same embedding model as the stored records.
data = el.from_client(qdrant_client, "banking77_minilm",
                      query_vector=query_vector, search_limit=500).materialize()
groups = el.cluster(data)
themes = el.topics(data, groups)
print(data.scope)  # retrieval_results
```

This selects up to `search_limit` nearest records with their **stored** vectors, then analyzes every selected record. It does not claim every semantic match. Full scans use scope `complete_selection`. Local `.filter(...)`, `.ids(...)` and seeded `.sample(...)` are also supported; full-scan ID/filter selection still reads all database pages and counts them against `max_scan_rows`. Filtering a semantic selection retains scope `retrieval_results`; sampling has scope `sample`.

`Limits()` defaults to 10,000 selected rows, 100,000 scanned rows, 64 MiB serialized selected content and batches of 256. Row/byte limits, duplicate IDs, dimension mismatches and conflicting embedding metadata fail without returning a partial materialization. `search_limit` must fit `max_rows` and the backend's query limit. Provenance includes collection, backend, vector field, retrieval mode and available model metadata. It excludes the SDK client, connection URL and credentials; query vectors are fingerprinted.

The application owns the client's lifecycle. EigenLens never calls `connect()` or `close()` on an injected client, including after a failure. Keep it connected until materialization finishes; close it in your application's context manager or `finally` block. Reads release their iterators on failure. Configure SDK request timeouts when you create the client. `max_seconds` checks elapsed retrieval time between SDK operations and records; it cannot interrupt an in-flight SDK call. Analysis after materialization is synchronous in the calling process.

### Submit the retrieved data to an MCP worker

A live Python client cannot be passed in an MCP JSON tool call. When your application already owns the client, materialize locally, then use your connected MCP `ClientSession`:

```python
import json

# session is an initialized mcp.ClientSession connected to the job server.
reply = await session.call_tool("register_records", {
    "records": [record.model_dump(mode="json") for record in data.records],
    "selection_scope": "complete_selection" if data.scope == "complete_selection" else "retrieval_results",
})
if reply.isError:
    raise RuntimeError("MCP record registration failed")
registered = reply.structuredContent
if registered is None:
    registered = json.loads(next(item.text for item in reply.content if item.type == "text"))
job = await session.call_tool("run_analysis", {
    "dataset_id": registered["dataset_id"],
    "operations": [
        {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
        {"id": "themes", "type": "topics", "clusters_from": "groups"},
    ],
})
if job.isError:
    raise RuntimeError("MCP analysis submission failed")
```

Poll `get_job` and read result summaries/pages as shown in the [MCP guide](mcp.md). For large data, use [staged JSONL uploads](mcp.md#connection-options). Registration accepts `complete_selection` or `retrieval_results`; the example maps local filters/samples to `retrieval_results`. Database source identity and local filter/sample details are not transferred through `register_records`. Alternatively, give the MCP server its own connection settings using `connect_database` below; it retrieves and preserves database provenance on the server. Client injection currently supports Qdrant and Weaviate; the configuration-based MCP adapters also support Pinecone, Milvus and pgvector.

## Installation

Install the extras your server uses:

| `db_type` | Pip extra | Collection meaning |
| --- | --- | --- |
| `qdrant` | `qdrant` | Collection; automatically selects a sole dense vector field |
| `pinecone` | `pinecone` | Index; connection option `namespace` selects its namespace |
| `milvus` | `milvus` | Collection; detects the primary key and a sole FLOAT_VECTOR field |
| `weaviate` | `weaviate` | Collection; requires HTTP and gRPC connectivity |
| `pgvector` | `pgvector` | PostgreSQL table with a `vector` column in `schema_name` |

For example, use `[server,qdrant]` in the README's pip URL, or `[server,databases]` for all built-ins. An installed driver does not establish connectivity or authenticate to a database; `connect_database` validates access before saving. Docker installs all five drivers. Supported SDK versions are declared in `pyproject.toml`, with tested versions in `requirements-tested.txt`.

## Where settings live

Set the server encryption key once, and keep its value in private server configuration:

```bash
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

Put that value in `EIGENLENS_CONNECTION_KEY`. Generate it once for a data directory. Connections cannot be decrypted with a different key after restart. The key is separate from the SQLite database/data volume. The core does not automatically load a `.env` file for local stdio; the launcher must supply its environment. Compose reads its `.env` settings and passes the documented variables to the container.

Saved definitions are in `<EIGENLENS_DATA>/connections.sqlite`, mode `0600`. The encrypted payload contains the URL/DSN, API key/password and options. Public metadata holds only the connection ID, caller identity, label, backend type and creation time. `list_connections` returns no URL or credentials. All lookup, list and deletion operations enforce the authenticated caller identity; encrypted definitions also bind their owner.

Docker uses `/data/connections.sqlite` in `worker-data`. Back up that volume and preserve its separate encryption key. Deleting a connection prevents future use of that ID; registered input snapshots and completed analysis artifacts remain available.

### Operator-configured local default

Configure these environment variables in the MCP server launcher:

```dotenv
EIGENLENS_DATABASE_TYPE=qdrant
EIGENLENS_DATABASE_URL=http://localhost:6333
EIGENLENS_DATABASE_API_KEY=
EIGENLENS_DATABASE_OPTIONS={}
```

The API key is optional for an unauthenticated local Qdrant. Use the Docker-network URL `http://qdrant:6333` from the worker container. `connect_database` with no arguments reads this default only for caller `local`. It saves a new connection ID; clients should reuse it rather than reconnect for each message. Explicit connection requests do not inherit default credentials.

### Per-user connections

An application's connection form passes `db_type`, URL, credentials and optional mappings to `connect_database` through its MCP client. Give the chat model the returned connection ID and selected collection. Avoid logging the request containing credentials. A chat thread stores the ID and collection; the MCP server stores the encrypted connection definition.

HTTP caller isolation follows the bearer-token mapping in the server config. Configure distinct callers/tokens for users requiring separate access. Multiple chats using the same caller identity can access that caller's connections. Stdio defaults to `local`.

## Exact database tool calls

These JSON objects are MCP `tools/call` payloads. IDs are placeholders to replace with returned values. Local Qdrant examples need no API key; an application backend can include `api_key` for authenticated access.

### connect_database

Use the configured local default:

```json
{"name": "connect_database", "arguments": {}}
```

Or connect explicitly:

```json
{
  "name": "connect_database",
  "arguments": {
    "db_type": "qdrant",
    "url": "http://localhost:6333",
    "name": "My banking database"
  }
}
```

Returns `connection_id`, `db_type`, label, creation time, status and collection count after validating access. Requires `EIGENLENS_CONNECTION_KEY`. At most `max_connections` definitions are saved per caller (default 128). Credentials are not returned. SDK errors are redacted.

### list_connections

```json
{"name": "list_connections", "arguments": {}}
```

Returns this caller's saved connection IDs and labels. Useful after reconnecting a chat host or restarting the server. The list does not revalidate database access; use `list_collections` to do that.

### list_collections

```json
{"name": "list_collections", "arguments": {"connection_id": "conn_example"}}
```

Returns names and their count for the chosen connection. In pgvector, collections are vector tables in its configured schema. For Pinecone, they are indexes; a supplied index host restricts the listing to that host.

### retrieve_records

Retrieve a whole collection:

```json
{
  "name": "retrieve_records",
  "arguments": {"connection_id": "conn_example", "collection": "banking77_minilm"}
}
```

This traverses every page and registers all returned records as `complete_selection`. Row, scan, byte and time limits fail without creating a partial dataset. Database changes during scans can affect consistency; adapters reject duplicate IDs and repeated cursors where available. No universal database snapshot transaction is promised across backends.

Retrieval runs in the MCP process, with database request timeouts and elapsed-budget checks between returned records/calls. It is separate from the cancellable analysis job; model loading and an in-flight SDK request are not forcibly terminated by `cancel_job`.

Semantic search with model resolution in the background:

```json
{
  "name": "retrieve_records",
  "arguments": {
    "connection_id": "conn_example",
    "collection": "banking77_minilm",
    "query": "ATM card and cash withdrawal problems",
    "search_limit": 1000
  }
}
```

Returns `dataset_id`, exact returned count, byte count, scope and selection provenance. Semantic search is a bounded ranked selection, marked `retrieval_results`. Default `search_limit` is 1,000; it must fit the server's `max_rows` and the adapter's search cap. It does not mean every matching record in the database. No further sampling occurs before analysis. Omit both query inputs for a whole-collection scan.

An external embedding provider can supply a vector instead:

```json
{
  "name": "retrieve_records",
  "arguments": {
    "connection_id": "conn_example",
    "collection": "example_collection",
    "query_vector": [0.1, 0.2, 0.3],
    "search_limit": 100
  }
}
```

This three-value vector illustrates the format; supply an actual vector in the collection's embedding space and dimension. Use exactly one of `query` and `query_vector`. The tool requires stored dense vectors; sparse/multivector-only collections are unsupported by the clustering pipeline. Missing vectors, conflicting model metadata, incompatible dimensions and invalid records fail retrieval.

### disconnect_database

```json
{"name": "disconnect_database", "arguments": {"connection_id": "conn_example"}}
```

Deletes the caller's saved definition and returns `status: "disconnected"`. Existing datasets, jobs and results remain. Clients open and close a database client for each tool request; credentials/configuration persist, not an open socket.

## Run clustering after retrieval

Take the returned dataset ID and submit the existing plan:

```json
{
  "name": "run_analysis",
  "arguments": {
    "dataset_id": "ds_example",
    "operations": [
      {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
      {"id": "themes", "type": "topics", "algorithm": "bertopic", "clusters_from": "groups", "parameters": {"ngram_max": 2}}
    ]
  }
}
```

Poll `get_job`, then read the result IDs using `get_summary`, `get_examples` and `get_result_page`. Source provenance includes database type, connection ID, collection, vector field, retrieval mode and available model/revision metadata. Semantic query vectors are fingerprinted; credentials are not copied into analysis artifacts. See [all analysis calls](mcp.md#exact-calls-for-the-eight-analysis-tools).

## Background model and field mappings

Connection `options` are backend configuration rather than choices in the chat composer. Common options:

| Option | Behavior |
| --- | --- |
| `text_field` | Payload/property/column containing text; default `text` |
| `vector_field` | Select a dense field when there are multiple; a sole field is detected |
| `id_field` | pgvector record ID column; default `id` (Milvus uses its declared primary key) |
| `embedding_model`, `embedding_revision` | SentenceTransformer model and optional revision for text-query embedding |
| `normalize_embeddings` | Whether to normalize generated query vectors; default `true` |
| `collections` | Per-collection overrides of the fields/model settings above; inherit connection defaults |
| `timeout` | Per-request database timeout in seconds; default 30, range 1–120 |
| `namespace` | Pinecone namespace; default empty |
| `db_name` | Milvus database; default `default` |
| `schema_name` | PostgreSQL schema; default `public` |
| `distance` | pgvector query operator: `cosine` (default), `l2` or `ip` |
| `grpc_host`, `grpc_port`, `grpc_secure` | Weaviate gRPC connection overrides |
| `adapter_options` | Encrypted database-specific options for custom adapter plugins |

Example operator configuration:

```json
{
  "text_field": "content",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "collections": {"banking77_minilm": {"text_field": "text", "vector_field": "body"}}
}
```

For text queries, saved configuration takes precedence over a sample record's `embedding_model` / `embedding_revision` metadata; conflicting supplied/stored values fail. The BANKING77 demo includes this metadata, so the server can resolve MiniLM automatically. An unknown model requires backend configuration or a caller-supplied `query_vector`. The server does not infer a model from dimensions. Text-query embedding currently supports SentenceTransformer models on CPU, with remote model code disabled and at most two models cached. Models can download on first use. Other embedding providers, including OpenAI embeddings, use externally generated `query_vector` in the same embedding space. Whole-collection scans need no query embedding model.

## Backend limitations and API references

- **Qdrant:** dense unnamed or named vectors; [scroll pagination](https://qdrant.tech/documentation/concepts/points/#scroll-points) and `query_points` return stored vectors with payloads.
- **Pinecone:** dense-vector queries include values/metadata. Full scans list IDs and fetch vectors in pages; [ID listing requires a serverless index](https://docs.pinecone.io/reference/api/2025-04/data-plane/list). Pod indexes can use semantic retrieval but whole scans fail. Namespace is configured per connection. Native response-size limits can reject large queries.
- **Milvus:** FLOAT_VECTOR fields; [query iterators](https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Vector/query_iterator.md) traverse records; search uses the collection's index/metric. No insert, update or delete operation is exposed.
- **Weaviate:** v4 client, HTTP and gRPC endpoints, with [object iteration including stored vectors](https://docs.weaviate.io/weaviate/manage-objects/read-all-objects). Named dense vectors are supported; configure a field if more than one exists. Default gRPC port is 50051 for HTTP/local and 443 for HTTPS/cloud; override for custom deployments.
- **pgvector:** PostgreSQL `vector` columns, quoted table/column identifiers and bound query values. Sessions are read-only with statement timeouts. Tables need a stable unique ID column. Full scans use server-side cursors; vector queries use [pgvector distance operators](https://github.com/pgvector/pgvector). Half/sparse/bit vectors require a different adapter implementation.

Adapters do not create collections or write database records. Use a stable collection during full scans. The existing BANKING77 setup remains the separate indexing demo.

## Add another database

Implement `eigenlens.databases.base.Adapter`: `list_collections`, `describe`, `scan`, `search` and `close`. Its constructor receives the decrypted connection spec; never return or log credentials. `scan` yields every normalized `Record` and releases resources when its generator is closed; `search` yields the requested bounded ranked selection with stored dense vectors. `describe` returns the selected `vector_field` and `dimension` when known. Set `max_search_limit` for the backend.

Publish a plugin entry point:

```toml
[project.entry-points."eigenlens.database_adapters"]
my_database = "my_package:MyDatabaseAdapter"
```

Installing that package on the server exposes its `db_type` through `get_capabilities`. Applications embedding the service can also call `eigenlens.databases.register_adapter(name, factory)` before starting MCP. New adapters reuse the encrypted connection store, bounded snapshot registration and analysis tools.

## Chat examples

> Connect to the server's configured database and list its collections.

> Use that connection and banking77_minilm. Show clusters in ATM cards. Report how many records were retrieved and the search limit, then analyze every retrieved record and show topics and source examples.

> Cluster the whole collection instead. Report an exceeded limit rather than silently analyzing a subset.

> Show my saved connections.

> Disconnect that database connection.
