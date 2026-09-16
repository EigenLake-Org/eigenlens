Act as a senior ML platform architect and Python engineer. Design and implement a working MVP of a portable ML execution layer for vector data. Use the working package name `vector_tools`, imported as `vt`.

The product should let developers and AI agents analyze records retrieved from vector databases: cluster similar records, extract topics, analyze sentiment, and identify outliers. Developers currently have to assemble database retrieval, data preparation, algorithm execution, and result interpretation themselves. This product should provide a consistent workflow across data sources and execution environments.

Build the MVP with working connectors for Weaviate, pgvector on PostgreSQL, Qdrant, and Pinecone, a composable Python SDK, a real local Python execution backend, and an MCP server deployable with Docker. All four data sources are part of the initial implementation scope. Define extension interfaces for further databases and a future Spark backend. Make reasonable implementation choices, document assumptions, and proceed through implementation and verification.

Make the [OpenAI Agents SDK for Python](https://openai.github.io/openai-agents-python/) a required client integration in the initial release. An application using its `Agent` and `Runner` must be able to discover and invoke this service's MCP tools, run analysis against any of the four configured data sources, inspect results, and answer the user's question. Include executable integration examples and acceptance checks as deliverables.

The main user experience is:

“Group these support tickets by recurring failure themes, describe each group, compare sentiment across groups, and show representative tickets and unusual cases.”

The agent determines the relevant records and analyses. The service computes assignments, scores, counts, and representative examples. The agent uses those results to explain the findings and link them to source records. Treat discovered themes as evidence for investigation; confirmed failure causes require additional evidence.

Use these execution models:

| Access mode | Where the algorithms execute |
| --- | --- |
| Python SDK used directly | In the calling application's Python process through the local backend |
| MCP server deployed with Docker | In a Python worker process inside the service container for the MVP |
| OpenAI Agents SDK connected to the MCP service | The agent application orchestrates tool calls; ML algorithms execute in the service's selected compute backend |
| Future Spark backend | On Spark executors, with the service submitting jobs and retrieving result references |

The MCP server exposes tools and handles requests. A worker performs CPU-intensive ML work so tool discovery and job-status calls remain responsive. The vector database supplies selected records and their vectors. Docker packages the service; the execution backend defines how computation runs.

The agent workflow must support the following sequence:

1. Discover available tools, supported operations, configured sources, and backend capabilities.
2. Select a dataset using existing retrieval-result IDs, a filtered collection, or a sample. Existing query results are a first-class input.
3. Receive a dataset handle plus selection metadata, available fields, and an exact or explicitly estimated record count when available.
4. Submit a structured analysis request referencing that handle.
5. Let the service validate the request, resolve the dataset, and execute the operations.
6. For background work, receive a job handle, inspect status, and obtain result handles on completion.
7. Retrieve compact summaries and selected examples; use those to answer the original question or request further analysis.

Bulk embeddings and complete datasets stay in the application and execution infrastructure. The model receives handles, summaries, and bounded examples. A reference from another service must be explicitly resolvable by this service. Otherwise, the calling application must register or upload the records through a documented ingestion path.

Separate the implementation into these boundaries:

| Boundary | Responsibility |
| --- | --- |
| Source adapters | Source inspection, selection, retrieval by ID, filtering, pagination, and batch reads |
| Dataset layer | Lazy selection plans, record identity, field mapping, sampling, materialization, and dataset fingerprints |
| Operation registry | Typed inputs, parameters, dependencies, outputs, and required backend capabilities |
| Planner and execution router | Validation, dependency ordering, resource limits, and supported backend selection |
| Execution backends | Execute operation implementations against datasets without depending on the MCP transport |
| Job and artifact stores | Persist job state, results, selected-data artifacts, and execution metadata |
| Public interfaces | Python SDK and MCP tools calling the same application services |

Keep source adapters independent from algorithms and compute engines. Keep result contracts independent from scikit-learn and Spark classes. Use versioned, serializable schemas for requests, dataset references, and results. Define an Arrow-compatible batch representation so future backends can exchange data without relying on a single NumPy array.

Implement the dataset contract with source identity, record ID, selected embedding field, optional text, and metadata. Preserve an optional parent document or ticket ID when several vector records belong to the same entity. State whether reported counts describe vector records or unique tickets, and make any aggregation policy explicit.

Implement these source mappings behind the same dataset interface:

| Source | Dataset scope | Required mapping |
| --- | --- | --- |
| Weaviate | Collection and optional tenant | Object UUID, selected named/default vector, text property, and metadata properties; explicitly retrieve stored vectors. See [Weaviate object iteration](https://docs.weaviate.io/weaviate/manage-objects/read-all-objects). |
| pgvector on PostgreSQL | Configured database, schema, and table | Primary key, vector column, text column, and metadata columns; retrieve rows through the PostgreSQL client. See the [pgvector project documentation](https://github.com/pgvector/pgvector). |
| Qdrant | Collection and configured selection scope | Point ID, named vector such as `body`, payload text field such as `text`, and payload metadata. |
| Pinecone | Configured vector index and namespace | Record ID, dense vector values, text in a configured metadata field, and remaining metadata. See [Pinecone record fetching](https://docs.pinecone.io/guides/manage-data/fetch-data). |

Fetch only fields required by the requested operations where the source API permits projection. Preserve alignment between IDs, embeddings, text, and metadata across pagination, sampling, and execution. Retain distance configuration and embedding provenance when available; for PostgreSQL, retain the configured query metric and applicable index/operator information. Reject incompatible dimensions, invalid numeric values, and unsupported vector types with actionable errors. Start with dense embeddings and advertise the vector/index types each connector actually supports.

Use optional dependency extras for the four database clients so importing the core package does not require every connector's dependencies. Require an explicit tenant or namespace when appropriate and preserve it in dataset identity, result provenance, and authorization checks. For PostgreSQL, quote configured identifiers correctly, parameterize predicate values, and stream rows with a defined ordering and consistency policy.

Publish per-source capabilities for ID fetches, iteration, filtered reads, projection, exact/estimated counts, reproducible sampling, and partitioned reads. Report each capability as native, implemented by the adapter, or unavailable, with relevant deployment/version constraints. Define a small common filter schema and advertise its supported predicates per connector. Any client-side filtering must preserve predicate semantics and report its read cost and limits.

For Pinecone vector indexes, implement existing-result ID fetching and namespace-aware reads. Use supported paginated metadata fetching or ID listing plus batched fetches when appropriate. Pinecone's documented ID-listing endpoint is limited to serverless indexes, so validate index capabilities before planning a collection scan or sample. An unavailable enumeration method should produce an actionable error or accept an explicitly supplied ID/artifact set. See [Pinecone ID listing](https://docs.pinecone.io/guides/manage-data/list-record-ids). Never use a large top-k similarity query as a substitute for a complete scan or an unbiased sample.

Support retrieval by existing record IDs across all four connectors, plus filtered reads, seeded sampling, batch iteration, and explicit materialization according to each connector's declared capabilities. Keep a dataset selection plan lazy until execution or an explicit materialization call. Apply row and memory limits before collecting a large dataset into local memory. Record which operations are pushed down to the source and which require client-side reads. The same analysis API and result contract must work after data is selected from any supported source.

Document sampling semantics and cost. Seeded sampling should be reproducible for an unchanged input and defined iteration order; use an adapter-level method when needed. A database's random-sampling endpoint must not be assumed to provide deterministic seeded sampling. A truncated scan must be identified as truncated. Record selected IDs and an execution-time content fingerprint; stage selected content where needed so all operations in a run use the same data. Do not claim snapshot isolation that the source cannot provide.

Every result must identify whether it describes a retrieved result set, a sample, or a complete filtered selection. A top-k similarity result set must not be presented as a representative estimate of an entire collection. Make missing-record, missing-text, and invalid-vector handling explicit, with skipped counts where skipping is permitted.

Implement these four analytical capabilities:

1. **Clustering.** Provide a real MiniBatchKMeans implementation over embeddings, configurable parameters and seed, per-record assignments, cluster sizes, centroids where applicable, and representative examples. Record preprocessing and distance assumptions. Validate cluster count against usable records; allow the caller to specify it and provide a documented default policy for omitted values. Return the effective parameters. Use the current [scikit-learn MiniBatchKMeans interface](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).
2. **Topic representation.** Extract weighted topic terms from text grouped by an existing clustering result using c-TF-IDF. Use class-level term frequencies and weighting consistent with the [BERTopic c-TF-IDF description](https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html). Treat this as representation of discovered groups. Validate that the clustering result and text belong to the same materialized dataset. Handle empty text, stop words, and empty vocabularies explicitly. Readable labels may come from terms or optional later agent interpretation.
3. **Sentiment analysis.** Implement one real, configurable sentiment-model adapter with batched inference on text. Return record IDs, labels, model-specific scores, and model identity/version. Document the label mapping and score meaning, including whether scores are calibrated probabilities. Allow configured aggregation by cluster. Explain model installation or caching and return a clear availability error when required weights are absent. Embedding similarity alone is insufficient to assign sentiment labels.
4. **Outlier detection.** Provide IsolationForest with configurable parameters, per-record raw scores, a documented anomaly-score direction, and ranked examples. Define the public anomaly score so higher values mean more anomalous, preserving the original model output and transformation. Scikit-learn's raw score conventions require this explicit mapping. See the [IsolationForest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

Preserve the following Python experience. The package name is a working name, and the example specifies the intended API to implement:

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
import vector_tools as vt

dataset = vt.from_qdrant(
    client,
    collection="tickets",
    vector_field="body",
    text_field="text",
)

data = dataset.sample(n=5_000, seed=42).materialize()

clusters = vt.cluster(
    data,
    estimator=MiniBatchKMeans(n_clusters=30, random_state=42),
)

topics = vt.topics(data, clusters=clusters, representation="ctfidf")
sentiment = vt.sentiment(data, model="configured-default")
outliers = vt.outliers(
    data,
    estimator=IsolationForest(random_state=42),
)

clusters.summary()
clusters.examples(cluster_id=3)
topics.summary()
sentiment.summary()
outliers.top(n=20)
```

Provide corresponding constructors for the other three sources. The following signatures specify the intended SDK API; the application supplies configured clients and connections. After construction, the same supported dataset selection and analysis methods apply:

```python
weaviate_dataset = vt.from_weaviate(
    weaviate_client,
    collection="Tickets",
    tenant=tenant_name,
    vector_field="body",
    text_field="text",
)

pgvector_dataset = vt.from_pgvector(
    pg_connection,
    schema="public",
    table="tickets",
    id_column="id",
    vector_column="embedding",
    text_column="text",
    metadata_columns=["product", "status"],
)

pinecone_dataset = vt.from_pinecone(
    pinecone_index,
    namespace="tickets",
    text_field="text",
)
```

Keep direct estimator objects as a Python convenience with explicitly supported estimator protocols. For MCP and remote execution, use registered algorithm names and JSON parameters. Map recognized estimators to portable specifications only when their semantics can be preserved. Additional locally supplied estimators must declare their input and output requirements. Remote requests must resolve to installed implementations through the registry.

Use the official MCP SDK and document the actual SDK and protocol versions tested. Expose tools with useful descriptions and validated input/output schemas. Support local stdio and a documented HTTP deployment using the transport supported by the chosen SDK. Tool discovery and invocation should follow the [MCP architecture and tool model](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture). Keep application-level job handling usable independently of optional protocol extensions.

Implement the OpenAI Agents SDK integration using its built-in MCP clients and `Agent(mcp_servers=[...])`. Use `MCPServerStreamableHttp` for the Docker/HTTP service and `MCPServerStdio` for a locally launched server. These are documented in the [Agents SDK MCP guide](https://openai.github.io/openai-agents-python/mcp/). Keep connection setup and cleanup inside the client's async context manager, covering the entire agent run. The agent host needs network access to the configured MCP endpoint; ML execution remains in the backend described above.

Provide an optional `agents` dependency extra containing a compatible `openai-agents` version. Record the tested `openai-agents`, `mcp`, and server versions together. Configure `OPENAI_API_KEY` and the chosen model in the agent application. Use application environment variables such as `OPENAI_MODEL`, `VECTOR_ML_MCP_URL`, `VECTOR_ML_MCP_TOKEN`, and `VECTOR_ML_SOURCE_ID` for the example. Keep database credentials in the service's configured connections. Document how the MCP endpoint is reached when the agent runs on the host versus inside another container.

Use this connection skeleton as the intended developer experience. It demonstrates the documented MCP client constructor and structured-content option; validate it against the selected version of the [Agents SDK MCP client reference](https://openai.github.io/openai-agents-python/ref/mcp/server/) during implementation. The service endpoint and source configuration are supplied by this project; implement and document them alongside the example.

```python
import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp


async def main():
    token = os.getenv("VECTOR_ML_MCP_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    async with MCPServerStreamableHttp(
        name="Vector ML",
        params={
            "url": os.environ["VECTOR_ML_MCP_URL"],
            "headers": headers,
        },
        client_session_timeout_seconds=30,
        use_structured_content=True,
    ) as server:
        agent = Agent(
            name="Ticket Analyst",
            model=os.environ["OPENAI_MODEL"],
            instructions=(
                "Use the Vector ML tools to inspect source capabilities, "
                "select the requested records, and run supported analyses. "
                "Base counts and findings on completed tool results. "
                "Include analysis scope and representative source IDs. "
                "If a job is still running after a status check, return its "
                "job ID and pending status so the host can resume later."
            ),
            mcp_servers=[server],
        )

        source_id = os.environ["VECTOR_ML_SOURCE_ID"]
        result = await Runner.run(
            agent,
            input=(
                f"Using configured source {source_id}, analyze a seeded "
                "sample of up to 5,000 support tickets if sampling is supported. "
                "Group them by recurring failure themes, compare sentiment "
                "across groups, and show representative tickets and outliers."
            ),
            max_turns=20,
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

Deliver a matching stdio example that launches the actual server entry point with `MCPServerStdio`. Keep server diagnostics on stderr so stdout carries the protocol correctly. Validate tool names and schemas through the Agents SDK's real discovery and conversion path, including optional fields and nested operation specifications. Return compact MCP structured content with a readable text fallback, and ensure errors and unavailable backend capabilities remain intelligible to the agent.

The connection skeleton covers a single agent run. Implement a companion application flow for long jobs: retain the submitted job ID, poll with bounded backoff in application code, and resume the agent with completed result handles or a terminal error. Reuse the original dataset/job and preserve the conversation through a supported SDK session or input-history mechanism. With stdio, keep the server context open during polling and resumed runs so its subprocess remains alive while local workers execute. Avoid repeated LLM calls solely to wait. Handle the agent's turn limit explicitly; `Runner.run` exposes `max_turns` as documented in the [Runner reference](https://openai.github.io/openai-agents-python/ref/run/). If submission retries are enabled, use a stable per-request idempotency key so a retry does not create a duplicate job.

Capture the actual tool-call sequence and associated dataset, job, and result IDs in the integration demonstration. The final answer must use computed counts and inspected examples. Keep the core analysis package and server usable without the Agents SDK installed; the SDK belongs to the calling application or optional integration extra.

Start with these tool contracts; small naming improvements are acceptable if examples and implementation stay consistent:

| Tool | Purpose |
| --- | --- |
| `get_capabilities` | List authorized configured source IDs, connector capabilities, supported operations, algorithms, model availability, backends, and resource limits |
| `describe_source` | Inspect an authorized source's collection/table/index, tenant/namespace, field mappings, and available read capabilities |
| `create_dataset` | Create a dataset handle from source selection, filters, IDs, or sampling parameters |
| `register_records` | Register bounded caller-provided records or a supported artifact reference |
| `run_analysis` | Validate and run a serializable operation plan |
| `get_job` | Return status, progress, errors, and completed result references |
| `cancel_job` | Request cancellation and report its actual outcome |
| `get_summary` | Return a bounded summary of one result |
| `get_examples` | Return representative source records for a result or cluster |
| `get_result_page` | Page through assignments, sentiment outputs, or ranked outliers |

Configure database connections and credentials in the service. Tools reference connection IDs and handles scoped to the caller. Validate access to source, dataset, job, and result handles. Apply payload limits and support a documented artifact-registration path for larger input. The service reads the selected source data; analysis results are stored in its artifact store.

A simplified `run_analysis` tool request should support this shape. This is an application payload, with the protocol envelope handled by the MCP SDK:

```json
{
  "name": "run_analysis",
  "arguments": {
    "dataset_id": "ds_123",
    "operations": [
      {
        "id": "groups",
        "type": "cluster",
        "algorithm": "minibatch_kmeans",
        "parameters": {
          "n_clusters": 30,
          "random_state": 42
        }
      },
      {
        "id": "themes",
        "type": "topics",
        "clusters_from": "groups",
        "representation": "ctfidf"
      }
    ],
    "execution": {
      "mode": "async",
      "backend": "auto"
    }
  }
}
```

Validate operation IDs and dependencies before execution. For the MVP, execute a whole plan on one capable backend. Automatic backend selection must honor the requested algorithm semantics and resource limits. If no installed backend supports the plan, return supported alternatives without silently changing the request. Return the selected backend and effective configuration in execution metadata.

Support synchronous SDK calls and asynchronous service jobs. Persist job state outside the request process. A simple durable local metadata store and mounted artifact directory are sufficient for the MVP. Include queued, running, succeeded, failed, and cancelled states, with explicit behavior for cancellation, interruption, and worker restart. Never report cancellation while computation continues as though it had stopped. Persist completed results so they remain inspectable after a request disconnects or the service restarts.

Represent completed jobs as a mapping from requested operation IDs to result handles. Each result should contain its result ID, run ID, dataset fingerprint, analysis scope, processed and skipped counts, source references, effective parameters, backend and algorithm/model versions, timestamps, and operation-specific outputs. Seeds and materialized inputs should be retained for reproducibility within a documented environment. Make result serialization and pagination stable, and bind cursors to a particular result.

Keep summaries compact. Store full assignments, vectors where needed, fitted models where appropriate, and per-record scores as artifacts. Return examples with source record IDs and text suitable for inspection. Expired or unavailable handles should produce an explicit error. Sentiment aggregation must declare its denominator and handling of missing text.

Design the future Spark extension around a backend protocol covering capability reporting, plan validation, submission, status, cancellation, and result retrieval. Define how datasets will be read in partitions or staged as distributed-readable artifacts. Define the source partitioning contract so workers do not independently scan and duplicate the full collection. Keep driver memory bounded.

Spark support requires explicit operator implementations and documented parameter differences. Spark's own clustering estimators are described in the [Spark ML clustering documentation](https://spark.apache.org/docs/latest/ml-clustering.html). An ordinary local scikit-learn estimator must not be relabeled as a distributed implementation. Preserve request/result contracts while reporting numerical or algorithmic differences. Deliver the extension interface and a capability matrix now; mark Spark execution as unimplemented until it actually runs and has been verified.

Keep the first release focused on Weaviate, pgvector, Qdrant, and Pinecone, the four analytical operations, the Python SDK, MCP service, OpenAI Agents SDK integration, local workers, durable results, and Docker deployment. All four connectors must implement their declared supported paths. Further databases, distributed execution, incremental model updates, advanced topic models, and dashboards belong in a documented roadmap.

Produce the following deliverables:

1. A concise architecture explanation with a Mermaid component diagram and an agent-interaction sequence diagram, showing exactly where data moves and computation happens.
2. A maintainable Python package with clear modules for schemas, sources, datasets, operators, backends, jobs, artifacts, SDK entry points, and MCP tools.
3. Implemented Weaviate, pgvector, Qdrant, Pinecone, and in-memory adapters, the local backend, the four operations, typed result objects, and the backend extension protocol. Include the source capability matrix and optional dependency extras.
4. A Dockerfile and Docker Compose profiles for the service with local Qdrant, Weaviate, and PostgreSQL with pgvector, using mounted persistence, model configuration, limits, and health checks. Provide Pinecone configuration and a credential-enabled integration example targeting a configured index and test namespace. Explain the worker process lifecycle.
5. Runnable SDK and MCP-client examples for each source, including the analysis request above, job polling, summary retrieval, and example inspection. Demonstrate existing-result IDs for all connectors and collection sampling wherever supported by the documented capability matrix. Provide OpenAI Agents SDK examples for Streamable HTTP and stdio, with source selection driven by configuration.
6. A reproducible support-ticket demonstration fixture with text, embeddings, metadata, known topic structure, and unusual records. State how demo embeddings are generated. Use synthetic data only where identified as such.
7. Documentation covering setup, source/model configuration, effective algorithm parameters, analysis scope, result formats, current limits, and the Spark capability roadmap.
8. An OpenAI Agents SDK integration walkthrough covering installation, environment variables, service startup, `Agent`/`Runner` usage, background-job resumption, and an observed tool-call sequence from a completed demonstration. Include both a small interactive fixture and a long-job status/resumption scenario.

Verify meaningful behavior before declaring the MVP complete:

- Selected IDs remain aligned with text, embeddings, and outputs across paginated source reads for every connector. The same logical fixture yields equivalent record coverage and portable result schemas across sources.
- Weaviate tenant selection, Pinecone namespace selection, Qdrant named-vector selection, and PostgreSQL schema/column mapping are preserved through dataset creation and analysis. Identical record IDs in different scopes remain distinct.
- Source capability checks handle unsupported filters, unavailable scans/sampling, and source-specific limits before execution. Validate PostgreSQL predicate parameterization and Pinecone enumeration constraints.
- Run integration checks against the local database profiles and a configured Pinecone test namespace when credentials are available. Clearly distinguish SDK-mocked contract checks from live integration checks, and report unavailable Pinecone credentials as an unverified integration rather than a passing live check.
- Sampling is reproducible for an unchanged fixture, and the recorded scope distinguishes sample, filtered selection, and retrieval results.
- Clustering and topic extraction produce inspectable results on the demonstration data without depending on specific cluster-label numbers.
- Sentiment executes the configured real implementation and reports model identity and availability correctly.
- Outlier ranking follows the public score convention and surfaces deliberately unusual fixture points.
- Empty selections, missing fields, invalid dimensions, mismatched dataset/results, and oversized local jobs produce defined outcomes.
- A real MCP client can discover tools, create a dataset, submit analysis, inspect a background job, and retrieve evidence through the deployed service.
- An OpenAI Agents SDK `Agent` invoked with `Runner.run` can use the service via Streamable HTTP and stdio, execute the ticket-analysis workflow, and produce an answer grounded in actual result counts and source examples. Verify connector portability by changing configured source selection while retaining the same agent integration.
- A long analysis returns a job handle promptly, survives the initial agent run ending, and resumes from the existing job after application-level polling. Verify failed jobs, tool errors, and turn-limit handling without duplicate submissions.
- Test the real Agents SDK tool-discovery/schema path even when a live model call is unavailable. Label deterministic model stubs and mocked SDK responses as such; report missing OpenAI credentials or model access as an unverified live agent run, not a successful integration demonstration.
- The service remains responsive during ML execution, stored results survive restart, and interrupted jobs receive a truthful state.
- Backend capability checks reject unavailable or unsupported Spark execution before launching work.

Inspect the available repository and instructions first. Choose compatible dependency versions and record the versions actually used. Briefly explain the design, then implement and run the relevant checks. Use real algorithm outputs in demonstrations, and distinguish implemented, tested, and planned features. If an environment dependency prevents a check, complete the accessible work and report the exact remaining check and limitation.
