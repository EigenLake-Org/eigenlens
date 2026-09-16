# Sources and selection

All sources implement the same dense record contract: source identity, string record ID, selected embedding, optional text, metadata and optional parent ID. The service configuration supplies connections; MCP tools receive only authorized source IDs. The core package does not import optional database clients until used.

## Constructors

```python
import vector_tools as vt

q = vt.from_qdrant(qdrant_client, collection="tickets", vector_field="body", text_field="text")
w = vt.from_weaviate(weaviate_client, collection="Tickets", tenant="team-a", vector_field="body", text_field="text")
p = vt.from_pgvector(pg_connection, schema="public", table="tickets", id_column="id",
                     vector_column="embedding", text_column="text", metadata_columns=["product", "status"])
f = vt.from_pinecone(pinecone_index, namespace="tickets", text_field="text",
                     index_description=pinecone_client.describe_index("my-index"))

# Each supports ID retrieval. Scans/sampling depend on the advertised capability.
data = q.ids(retrieved_ids).materialize()
data = q.filter({"field":"status", "op":"eq", "value":"open"}).sample(5000, seed=42).materialize()
```

Pass `source_name` when using custom SDK clients to give distinct installations stable human-readable identities. Service-owned adapters receive their configured source IDs. PostgreSQL retains database/host/schema/table and column mapping. Qdrant retains its endpoint, collection, named vector, configured dimension and distance. Weaviate retains collection/tenant/vector mapping and distance configuration. Pinecone retains host, explicit namespace, verified deployment information, dimension and distance where supplied by index description. Embedding provenance is caller supplied when the source has no model information; no embedding model is guessed.

## Capability matrix

N = native source/client operation; A = adapter implementation; U = unavailable. Capabilities are also exposed through `get_capabilities` and `describe_source`. `describe_source` is authoritative for deployment-dependent Pinecone enumeration.

| Source | ID fetch | Iterate | Filters | Projection | Exact count | Estimated count | Seeded sample | Partition reads |
|---|---|---|---|---|---|---|---|---|
| Memory | A | A | A | A | N | U | A | U |
| Qdrant | N | N | A | N | U | U | A | U |
| Weaviate | N (per ID) | N | A | N | U | U | A | U |
| pgvector | N | N | N | N | U | U | A | U |
| Pinecone, verified serverless dense | N | N (list + fetch) | A | U | U | U | A | U |
| Pinecone, other/unverified index | N | U | A over supplied IDs | U | U | U | U for collection scan | U |

Counts marked U are not claimed even where a database has another count API: these adapters have not implemented it. Before execution, memory selections can report an exact count; other sources report unknown, plus requested ID count when relevant. After materialization, selected/processed/skipped counts are exact for that run. No approximate count is labelled exact.

Projection currently omits vectors for text-only plans; analysis plans retain text and metadata for inspection and provenance. Qdrant retrieves only the selected named vector. Weaviate's iterator explicitly requests stored vectors and selects the configured named/default vector. Pinecone fetch returns full vectors and metadata because its fetch API lacks equivalent projection. No adapter generates new embeddings.

### Source constraints

- **Qdrant:** tested Docker 1.15.5 with client 1.15.1, named dense `body`, paginated scroll and batch retrieval. Multivectors are rejected. Cosine collections normalize stored vectors; analyses operate on the stored values, which can differ in magnitude from inserted values.
- **Weaviate:** tested Docker 1.32.4 with client 4.23.1, multi-tenant collection and named dense vector. Multi-tenant collections require an explicit tenant. UUID cursor iteration uses bounded cache pages; ID retrieval uses per-object calls. No cross-page snapshot guarantee is made.
- **PostgreSQL/pgvector:** tested PostgreSQL 17 Docker image, psycopg 3.3.5 and pgvector client 0.5.0. Uses configured `vector` float32 columns; `halfvec`, sparse vectors and multivectors are rejected. A caller-supplied connection must be idle so the adapter can open its own read-only REPEATABLE READ transaction. Rows stream through a named server cursor ordered by configured primary key. The configured metric/operator are provenance only; this scan does not use a vector index. DDL and index creation are outside the read adapter.
- **Pinecone:** client 8.1.2 contract checks cover namespace-aware ID fetch, paginated listing and batches. A `describe_index` result must establish a serverless dense index before enumeration is enabled. Passing a bare `serverless=True` flag does not bypass this check. Pod/unverified indexes can analyze supplied IDs. Collection sampling never uses top-k nearest-neighbor search. Live tests require a configured existing 4-dimensional test index and explicit test namespace; no cloud index is created by the example.

### Common filters

Filters combine with AND. Predicates: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `exists`. Fields are `id`, `text`, `parent_id`, or a top-level metadata key. Missing and null values never satisfy comparisons, including `ne`; `exists: false` selects missing/null values. `in` requires a scalar list; `exists` requires a boolean. Numeric/string ordering requires compatible types; adapter comparisons of incompatible types return false. PostgreSQL uses typed column comparisons; callers must use values compatible with the configured column type or receive a database validation failure.

PostgreSQL maps only configured columns, quotes identifiers and parameterizes values. Other adapters apply exactly the common evaluator after reading. Client filtering may scan all records in the bounded selection; cost and source read counts are reported. Nested metadata queries, OR, full text search and arbitrary source-native filters are intentionally unavailable.

## Sampling and consistency

Plans remain lazy until explicit `.materialize()` or worker execution. A sample assigns each record a SHA-256 rank from `(seed, record ID)` and keeps the smallest N ranks in a bounded heap. This is reproducible for unchanged content/IDs and independent of page order. It is a pseudorandom uniform selection over unique IDs, not a database random endpoint. Source changes during a scan can change the result; no snapshot is claimed for Qdrant, Weaviate or Pinecone.

Sampling requires a complete bounded read of the candidate selection, O(read records) traffic and O(N) retained records plus a bounded duplicate-ID set. `max_scan_rows` and scan-byte limits apply even when N is small. The adapter aborts rather than returning an unlabelled truncated sample. Duplicate IDs across pages are rejected. ID fetches report missing IDs; missing vectors or invalid dimensions/numeric values reject the run. Materialized records sort by string ID to preserve output alignment.

Scopes distinguish `retrieval_results`, `sample`, `filtered_selection`, and `complete_selection`. A sample of supplied IDs retains its retrieval population in selection metadata; it is not evidence about the whole collection. Registration defaults to `retrieval_results`; callers must explicitly declare `complete_selection` when that is justified. Scope is relative to the configured source and tenant/namespace.

Selected IDs and exact execution-time content are staged, and the fingerprint hashes source identity plus canonical record content. All operations in a run use that materialization. Reproducibility is tied to selected content, algorithm parameters, seed and recorded dependency versions; results are not promised bitwise-identical across platforms.

## Configuration and live tests

`config/local.json` uses loopback databases; `config/docker.json` uses Compose service names. Set `VECTOR_ML_CONFIG` for SDK/client subprocess examples; Docker supplies its config on the command line. PostgreSQL DSNs and source API keys are resolved from named environment variables. The local seeder only upserts the deterministic fixture IDs in configured demo scopes and does not delete existing collections.

```sh
export PGVECTOR_DSN='postgresql://vector:vector@localhost:5432/vector'
python -m examples.seed_sources qdrant weaviate pgvector
VECTOR_ML_LIVE=1 pytest -q tests/test_live_sources.py

# After configuring an existing Pinecone test index and namespace:
export PINECONE_API_KEY='...'
python -m examples.seed_sources pinecone
VECTOR_ML_LIVE=1 pytest -q tests/test_live_sources.py -k pinecone
```

Pinecone writes are eventually consistent: wait for fixture visibility before testing. The live coverage assertion expects the test scope to contain exactly the fixture IDs; use isolated demo scopes. To analyze an existing retrieval result set, set `VECTOR_ML_IDS` to a JSON array when running `examples.source_sdk` or `examples.mcp_client`.
