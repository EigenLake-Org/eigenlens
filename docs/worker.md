# Analysis worker

The MCP server runs PCA/DBSCAN and BERTopic inside isolated job processes. Callers supply existing vectors and optional text, and receive assignments, coordinates, topics and representative examples.

## Docker

### Prebuilt image

Use the public Docker Hub image `eigenlake/eigenlens:0.1.0`. It supports `linux/amd64` and `linux/arm64`; Docker selects the matching architecture. No GitHub account, registry login or repository clone is needed.

For a **new installation**, create a private runtime configuration and start the MCP worker:

```bash
mkdir -p eigenlens-deploy
cd eigenlens-deploy
# Generate once. Retain .env across restarts.
if [ ! -f .env ]; then
  (umask 077
   printf 'EIGENLENS_MCP_TOKEN=%s\nEIGENLENS_CONNECTION_KEY=%s\nEIGENLENS_WORKERS=1\n' \
     "$(openssl rand -hex 32)" "$(openssl rand -base64 32 | tr '+/' '-_')" > .env)
fi
docker pull eigenlake/eigenlens:0.1.0
docker run -d --name eigenlens-worker --init --restart unless-stopped \
  --publish 127.0.0.1:8020:8000 --memory 8g --cpus 2 \
  --env-file .env --env EIGENLENS_CPU_BUDGET=2 \
  --volume eigenlens_worker-data:/data \
  eigenlake/eigenlens:0.1.0
```

Connect your MCP client to `http://127.0.0.1:8020/mcp` using the token from `.env`. The server still requires this token even though its Docker image is public. No host Python or compiler is needed. For database tools, add `EIGENLENS_DATABASE_TYPE`, `EIGENLENS_DATABASE_URL`, and any credentials/options to the private runtime environment, or pass connection settings from your application's backend. Database URLs must be reachable from inside the container. See [database setup](databases.md).

Stop the container with `docker stop eigenlens-worker`; its named data volume remains. Preserve the separate encryption key when reusing that volume. Do not generate a replacement key for an existing connections database.

Existing Compose installations can use `compose.release.yaml`, which retains their project, service, network and named volume. The repository release assets still require GitHub repository access; downloading the Docker image does not. Override `EIGENLENS_IMAGE` to choose another released tag or digest. Allow active jobs to finish before recreating the worker. The `latest` tag tracks releases, but use a version or digest for reproducible deployments.

### Build from source

```bash
export EIGENLENS_MCP_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export EIGENLENS_WORKERS=1
docker compose --profile worker up -d --build --wait worker
```

The variables select the bearer token and concurrent worker count. The endpoint is `http://127.0.0.1:8020/mcp`. MCP and upload routes require the same token; `/health` is public. Large selections use [streamed JSONL uploads](contracts.md#http-jsonl-upload). Use TLS for remote deployments.

The image installs CPU PyTorch and all five database adapter SDKs. Topic modeling uses precomputed embeddings and existing DBSCAN assignments through BERTopic's manual workflow; no embedding model is downloaded by analysis jobs. UMAP and HDBSCAN are dependencies of BERTopic but are not executed by this pipeline. Text-query retrieval may resolve/download a query embedding model separately in the MCP process.

For database tools, configure `EIGENLENS_CONNECTION_KEY` in the private Compose `.env` file. Keep this key across restarts, separate from the data volume. Optional `EIGENLENS_DATABASE_TYPE`, `EIGENLENS_DATABASE_URL`, `EIGENLENS_DATABASE_API_KEY` and `EIGENLENS_DATABASE_OPTIONS` provide a default for caller `local`. Per-user clients can call `connect_database` with their own settings. See [database setup](databases.md).

Query models cache under `/data/models/huggingface` in the data volume, so restarts reuse downloaded weights. Analysis jobs continue using stored vectors and require no query model.

## Workers and limits

Default concurrency is **1**, configurable from 1–16 with `EIGENLENS_WORKERS`. Each job is a process in the same worker container. More workers allow concurrent jobs; they do not split one analysis into smaller jobs. Core startup also supports `EIGENLENS_WORKERS` directly.

Compose divides an 8 GiB working-memory estimate among slots. The container retains two CPUs and an 8 GiB memory limit. Capabilities report per-job limits; `/health` reports worker slots. The worker config allows 20,000 rows, 512 MiB serialized input and 30 minutes per job. Oversized selections fail without silent sampling. Memory estimates do not include every allocation made by imported libraries.

Restart interrupts unfinished jobs. Change worker count after active jobs finish. Per-job cancellation stops only the selected process.

## Persistence

The `eigenlens` Compose project stores worker artifacts in `worker-data` and Qdrant data in `qdrant`. The worker image is `eigenlens-worker`. Stopping the worker preserves data:

```bash
docker compose --profile worker stop worker
```

Completed results and encrypted connections remain available after restart. Connection definitions live in `/data/connections.sqlite` and require their original encryption key. Interrupted jobs are marked failed. Use one supervisor per data volume. Automatic retention is not implemented.
