# Verification report

Verified on 2026-09-15, with the food-review setup verified on 2026-09-16, using Python 3.13.1 on the host and Python 3.13.15 in Docker. Vector Tools server version is 0.1.0. See [exact package versions](evidence/versions.json), [container runtime](evidence/docker-runtime.json), and `requirements-tested.txt` at the repository root.

## Outcome

**39 tests passed; two expected cases are skipped.** The current full suite includes the active-ML cancellation check, native MCP clients, and eight food-dataset validation/cache tests.

- [Current full-suite output](evidence/food-setup-tests.txt): 39 passed, 2 skipped.
- [Earlier full-suite output](evidence/tests.txt): 30 passed, 2 skipped.
- [Active ML responsiveness/cancellation](evidence/active-ml-test.txt): 1 passed.
- [Native MCP regression output](evidence/native-mcp-tests.txt): 2 passed.
- [Final core/connector output](evidence/final-contract-tests.txt).
- [Final live database output](evidence/final-live-tests.txt).
- Python compilation and `pip check` passed.
- The image built successfully and the Docker service reached healthy status.

One skip is live Pinecone because `PINECONE_API_KEY` is absent. The other is an HTTP-specific auth check when the fixture is using stdio. The installed Agents SDK emits three internal HTTP-client alias deprecation warnings; these did not prevent discovery, invocation or resumption.

## Implemented and verified

| Area | Evidence |
|---|---|
| Real clustering, c-TF-IDF, VADER and IsolationForest | Core tests, explicit sklearn-estimator-object regression, and [computed SDK demonstration](evidence/sdk-demo.json) |
| Sampling, IDs/text/vector alignment, limits, null predicates, mismatched fingerprints and empty selections | Core and connector tests |
| VADER availability failure, model identity, cluster sentiment denominator | Core tests and result artifacts |
| Qdrant named-vector retrieval, ID fetch, paginated read, filters and sampling | Actual in-process Qdrant client and live Docker Qdrant 1.15.5 |
| Weaviate tenant/named-vector retrieval and paginated ID alignment | Connector mocks and live Docker Weaviate 1.32.4 |
| PostgreSQL quoted identifiers, parameterized predicates, pgvector typed values, ordered streaming | SQL contract check and live PostgreSQL 17/pgvector Docker service |
| Pinecone namespace isolation, ID fetch, serverless scan gate and batched pagination | Connector mocks with installed SDK response dataclasses; **not live Pinecone** |
| Worker responsiveness, cancellation, terminal errors, queue idempotency and access isolation | Service and real transport tests |
| Stored results across service/container restart | Service tests and [Docker restart evidence](evidence/docker-restart.json) |
| Official MCP initialization, discovery and tool requests over stdio and Docker HTTP | [stdio trace](evidence/mcp-stdio.json), [HTTP trace](evidence/mcp-http-docker.json) |
| Agents SDK discovery/conversion, optional/nested schemas, actual Agent/Runner invocations | Both transports, explicitly deterministic model stub |
| Long-job host polling, preserved history, turn-limit resumption and no duplicate submission | Agent tests and observed initial queued status in all eight demonstrations |
| Core/service independent of Agents SDK | Docker runtime reports `agents_installed: false` while the client host connects successfully |
| Unsupported Spark rejected before launch | Planner/service/MCP tests |

The protocol negotiated by the real MCP clients was **2025-11-25**, with service version **0.1.0**. Tested packages include MCP 1.30.0, OpenAI Agents 0.22.2, scikit-learn 1.9.1 and VADER 3.3.2.

## Computed synthetic fixture findings

The fixture has 62 vector records: three sets of 20 themed tickets plus two intentionally unusual vectors. Four-cluster runs produce sizes 20, 20, 20 and 2 on the in-memory fixture (cluster label numbers are arbitrary). Weighted terms expose login/authentication, billing/payment and sync/upload text. VADER computes 42 negative and 20 positive records. The two unusual IDs rank first in the SDK IsolationForest result. Examples and full result metadata are saved in the SDK evidence JSON.

Qdrant cosine storage normalizes vectors, so inserted and retrieved vector magnitudes differ. Cross-source tests compare direction and record coverage and verify portable schemas; they do not claim identical numerical clustering across differently normalized source storage.

All fixture content and embeddings are explicitly synthetic. These outputs verify software behavior, not performance or accuracy on a real support corpus.

## Observed agent sequence

This is a deterministic **model stub**, with real Agent/Runner, MCP calls, database reads, worker jobs and ML. No OpenAI API call was made.

- Dataset: `ds_960722fdffca46c89b482e1711c3f061`
- Job: `job_d3dde28ad3f9494a8d518a928e630241`
- Cluster result: `res_541a7d405b1d439f9e527db995b35aa8`
- Topics result: `res_60d29c86374b47e699a4078841bb0109`
- Sentiment result: `res_cfa23e6525ad4effb6334b6bc62ceec7`
- Outliers result: `res_d654d4a226644246a4aa26cb8019240b`

Observed call sequence:

get_capabilities → describe_source → create_dataset → run_analysis → get_job → get_job → get_job → get_job → get_job → get_job → get_job → get_summary → get_examples → get_summary → get_examples → get_summary → get_examples → get_summary → get_examples

The initial run returned queued status, the host polled the existing job without LLM calls, and the resumed run inspected summaries and examples. The same job and cluster result were then retrieved unchanged after a Docker service restart. The returned answer is stored in the trace and uses computed counts and inspected source IDs.

### Source portability demonstrations

Only the configured source/transport changes; all runs use `examples/agent_client.py` and the same Agent integration. These traces contain real tool arguments, result handles, summaries and examples.

| Trace | Outcome | Job |
|---|---|---|
| [agent-http-demo](evidence/agent-http-demo.json) | completed | `job_d3dde28ad3f9494a8d518a928e630241` |
| [agent-http-pgvector](evidence/agent-http-pgvector.json) | completed | `job_976700032f974e4681d81cd837626db0` |
| [agent-http-qdrant](evidence/agent-http-qdrant.json) | completed | `job_8c0c2be457c346b1856901148f74c346` |
| [agent-http-weaviate](evidence/agent-http-weaviate.json) | completed | `job_21a4c1296e214c69b8e5cb27449f3be4` |
| [agent-stdio-pgvector](evidence/agent-stdio-pgvector.json) | completed | `job_9c9935991e8e4cc19054aec397b82383` |
| [agent-stdio-qdrant](evidence/agent-stdio-qdrant.json) | completed | `job_8acbe9814316415a8bdba58bae5f603a` |
| [agent-stdio-weaviate](evidence/agent-stdio-weaviate.json) | completed | `job_f5a17299e22247bf9153ab5c13abbeb5` |
| [agent-stdio](evidence/agent-stdio.json) | completed | `job_45a6ac2772ca4bd3b7905a9830439bba` |

## Unverified external checks

- **Live OpenAI model calls:** `OPENAI_API_KEY` and a selected accessible model were not available. Run the documented agent example without `--deterministic` over both transports and each source. The deterministic model stub is not a successful live-model test.
- **Live Pinecone:** no credentials were present. Configure an existing four-dimensional dense test index and isolated namespace, seed the fixture, wait for visibility, and run the live source test plus both agent transports. Mocked response checks are not a passing cloud integration.
- **Distributed Spark execution:** intentionally unimplemented; only the extension contract, partition contract, capability matrix and early rejection are delivered.
- **Additional deployment coverage:** older Python/dependency combinations, large-scale workloads, remote TLS/OAuth deployment and multi-replica execution were not verified. The current service is one supervisor with a local durable store.

The implementation, tests, source profiles, SDK/MCP/agent examples and documentation are delivered. Fully credentialed end-to-end acceptance remains pending the two external integrations above.


## Reproducible food-review setup

The [Docker-only setup](food-demo.md) downloads a commit-pinned CSV with a verified SHA-256, indexes 1,000 existing 1,536-dimensional OpenAI vectors into the separate `food_reviews` collection, and verifies every stored ID, text, rating and cosine-normalized vector. No embeddings are regenerated. The pinned CSV does not explicitly record the exact embedding model, and provenance reports that limitation.

A real MCP client analyzed all 1,000 records into eight clusters and inspected representative reviews and outliers. VADER computed 903 positive, 78 negative and 19 neutral labels. All 1,000 sentiment outputs were joined to original star ratings by deterministic point ID. These model labels differ from the dataset's star ratings, as expected; the report includes both and does not equate them.

The public demo has no-key `analyze` mode and optional live `agent` mode. Its report comes from actual ML and MCP calls, not the scripted model stub. Live agent mode still requires the user's OpenAI credentials.
