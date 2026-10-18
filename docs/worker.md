# Analysis worker

The MCP server runs PCA/DBSCAN and BERTopic inside isolated job processes. Callers supply existing vectors and optional text, and receive assignments, coordinates, topics and representative examples.

## Docker

### Prebuilt image

Use `ghcr.io/eigenlake-org/eigenlens:0.1.0`. The image supports `linux/amd64` and `linux/arm64`; Docker selects the matching architecture.

Download the deployment file from the release, without cloning the repository:

```bash
mkdir -p eigenlens-deploy
cd eigenlens-deploy
gh release download v0.1.0 --repo EigenLake-Org/eigenlens \
  --pattern compose.release.yaml
# Generate these once for a new deployment; retain .env across restarts.
if [ ! -f .env ]; then
  (umask 077
   printf 'EIGENLENS_MCP_TOKEN=%s\nEIGENLENS_CONNECTION_KEY=%s\nEIGENLENS_WORKERS=1\n' \
     "$(openssl rand -hex 32)" "$(openssl rand -base64 32 | tr '+/' '-_')" > .env)
fi
docker compose --env-file .env -f compose.release.yaml --profile worker pull worker
docker compose --env-file .env -f compose.release.yaml --profile worker \
  up -d --wait worker
```

The MCP endpoint is `http://127.0.0.1:8020/mcp`. Configure clients with the token in the private `.env` file. No host Python or compiler is needed. The release file has no `build` section and uses the same project, service, network and named data volume as the source-based setup. For an existing installation, reuse its token and encryption key; do not generate a replacement key for an existing connections database.

The repositories currently require GitHub access to download their release assets. Public GHCR packages can be pulled anonymously; restricted packages require `docker login ghcr.io` with a GitHub classic token carrying `read:packages` and access to the package. The release file can also be copied from the repository.

To start the optional local Qdrant service, use the same file with `--profile qdrant up -d --wait qdrant`. Its stored points are in a persistent volume; a new deployment starts empty. Connect an existing database or prepare the [BANKING77 dataset](../README.md#try-the-banking77-demo).

Set `EIGENLENS_IMAGE` in `.env` to a different released version or digest when upgrading. `latest` is available, but use a version or digest for a deployment you need to reproduce. Pull first and allow active jobs to finish before recreating the worker.

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
