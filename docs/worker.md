# Analysis worker

The MCP server runs PCA/DBSCAN and BERTopic inside isolated job processes. Callers supply existing vectors and optional text, and receive assignments, coordinates, topics and representative examples.

## Docker

```bash
export VECTOR_CHAT_WORKER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export VECTOR_CHAT_WORKERS=1
docker compose --profile chat-worker up -d --build --wait chat-worker
```

These Compose variables and service names remain for deployment compatibility. They select the bearer token and concurrent worker count. The endpoint is `http://127.0.0.1:8020/mcp`. MCP and upload routes require the same token; `/health` is public. Large selections use [streamed JSONL uploads](contracts.md#http-jsonl-upload). Use TLS for remote deployments.

The image installs CPU PyTorch as a BERTopic dependency. Topic modeling uses precomputed embeddings and existing DBSCAN assignments through BERTopic's manual workflow; no embedding model is downloaded by the worker. UMAP and HDBSCAN are dependencies of BERTopic but are not executed by this pipeline.

## Workers and limits

Default concurrency is **1**, configurable from 1–16 with `VECTOR_CHAT_WORKERS`. Each job is a process in the same worker container. More workers allow concurrent jobs; they do not split one analysis into smaller jobs. Core startup also supports `VECTOR_ML_WORKERS` directly.

Compose divides an 8 GiB working-memory estimate among slots. The container retains two CPUs and an 8 GiB memory limit. Capabilities report per-job limits; `/health` reports worker slots. The worker config allows 20,000 rows, 512 MiB serialized input and 30 minutes per job. Oversized selections fail without silent sampling. Memory estimates do not include every allocation made by imported libraries.

Restart interrupts unfinished jobs. Change worker count after active jobs finish. Per-job cancellation stops only the selected process.

## Persistence

The `vector-ml` Compose project keeps job state and artifacts in `chat-worker-data`; optional Qdrant uses its own `qdrant` volume. Stopping the worker preserves data:

```bash
docker compose --profile chat-worker stop chat-worker
```

Completed results remain available after restart. Interrupted jobs are marked failed. Use one supervisor per data volume. Automatic retention is not implemented.
