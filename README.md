# Vector Tools

A working local ML execution layer for vector records, imported as `vector_tools` (`vt`). Analyze selected records from **Weaviate, PostgreSQL/pgvector, Qdrant, Pinecone, or memory** using the same Python SDK and MCP tool contracts.

The implementation includes MiniBatchKMeans, c-TF-IDF topic representation, English VADER sentiment, IsolationForest, background worker processes, durable results, Docker deployment, and OpenAI Agents SDK examples.

**Verification:** See [the verification report](docs/verification.md) for measured checks and remaining external integrations. Live OpenAI model calls and live Pinecone reads require your credentials. The deterministic agent example uses a model stub with the real Agents SDK, MCP transport, service and algorithms. Spark execution is an extension interface, not an implemented backend.

## Try real data with one command

With Docker running, execute:

```bash
./scripts/try-food-demo.sh
```

This downloads and verifies **1,000 food reviews with precomputed OpenAI embeddings**, indexes them into Qdrant, and produces a report using all four ML operations through MCP. **No API key or host Python installation is needed.** Open `demo-output/food-review-report.md` afterward.

For a live OpenAI agent, set `OPENAI_API_KEY` and `OPENAI_MODEL`, then run `./scripts/try-food-demo.sh agent`.

[Full food-review demo guide](docs/food-demo.md) · [Qdrant dashboard](http://localhost:6333/dashboard)

## Run the local SDK demo

Use Python 3.13 for the reproducible constrained setup; the tested host uses Python 3.13.1. Package metadata permits older Python 3.11+ with compatible dependency resolution, but those combinations are unverified. macOS and Linux are supported for the service process supervisor.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[sentiment,server,agents,test,qdrant,weaviate,pgvector,pinecone]' -c requirements-tested.txt
python -m examples.sdk_demo
pytest -q
```

`requirements-tested.txt` records the exact tested dependencies and can be used as constraints. Core imports do not require database clients, MCP, the Agents SDK, or VADER.

```python
from sklearn.cluster import MiniBatchKMeans
import vector_tools as vt
from vector_tools.demo import tickets

data = vt.from_records(tickets()).sample(n=50, seed=42).materialize()
clusters = vt.cluster(data, estimator=MiniBatchKMeans(n_clusters=4, random_state=42))
print(clusters.summary())
print(vt.topics(data, clusters=clusters).summary())
print(vt.sentiment(data, clusters=clusters).summary())
print(vt.outliers(data).top(n=5))
```

The demonstration has 62 explicitly synthetic tickets: three known themes plus two unusual embedding points. Its embeddings are seeded Gaussian vectors around hand-selected centers, not embeddings from a language model. Counts refer to vector records; parent ticket IDs are retained but not deduplicated.

## Run MCP and agent examples

```sh
# A real stdio MCP client launches the service, submits a job and inspects results.
python -m examples.mcp_client

# A deterministic model stub drives the REAL Agent/Runner/MCP/ML path.
python -m examples.agent_client --deterministic

# A live OpenAI model needs these application-side variables.
export OPENAI_API_KEY='...'
export OPENAI_MODEL='your-available-model-id'
python -m examples.agent_client
```

The agent host polls pending jobs in application code and resumes with the original conversation and completed result handles. Keep the stdio context open for that entire flow.

## Docker and local databases

```sh
docker compose --profile local up -d --build
export PGVECTOR_DSN='postgresql://vector:vector@localhost:5432/vector'
python -m examples.seed_sources qdrant weaviate pgvector
export VECTOR_ML_TRANSPORT=http
export VECTOR_ML_MCP_URL=http://localhost:8000/mcp
export VECTOR_ML_MCP_TOKEN=local-demo-token
export VECTOR_ML_SOURCE_ID=qdrant
python -m examples.mcp_client
python -m examples.agent_client --deterministic
VECTOR_ML_LIVE=1 pytest -q tests/test_live_sources.py
```

The demo Compose configuration binds published ports to loopback. Change the demo token and configure a TLS reverse proxy for remote deployment. The `qdrant`, `weaviate`, and `pgvector` profiles start individual databases; `local` starts all three. The default service also works against the built-in `demo` source without database profiles.

## Documentation

- [Architecture, data movement and worker lifecycle](docs/architecture.md)
- [Sources, selection semantics and capability matrix](docs/sources.md)
- [Analysis contracts, limits and result handling](docs/contracts.md)
- [Agents SDK integration and observed workflow](docs/agents.md)
- [Verification evidence and remaining checks](docs/verification.md)
- [Original build specification](docs/build-specification.md)

## Repository map

`vector_tools/schemas.py` defines portable contracts; `sources/` contains adapters; `datasets.py` handles lazy plans and materialization; `registry.py` validates operation plans; `operators.py` implements ML; `backends.py` defines execution and extension contracts; `artifacts.py` and `jobs.py` persist artifacts and supervise workers; `service.py` owns application services; `server.py` exposes MCP; `examples/` and `tests/` exercise the workflows.

## BANKING77 DBSCAN demo

PCA/DBSCAN parameter search, intent evaluation, interactive 3D plots, and SDK/MCP agent access are available on this branch. See [BANKING77 setup and agent guide](docs/banking77.md).

### Query BANKING77 in Qdrant and plot clusters

```bash
pip install -e '.[banking,qdrant]'
./scripts/try-banking-query.sh
```

Indexes the full corpus with MiniLM, queries Qdrant, and clusters the top 2,000 matches. Open `demo-output/banking77-query/banking77_dbscan_3d.html`. Use `python -m examples.banking77_qdrant query --query "your banking question"` for another query. An optional agent tool is in `examples/banking77_query_agent.py`.

## Run clustering on a job server

Start `python -m examples.query_job_server` with `VECTOR_QUERY_TOKEN` set. Clients submit to `POST /jobs`, poll the job ID, and download the plot and results. The agent client also uses this HTTP server. See [job server setup, API and deployment](docs/query-job-server.md).
