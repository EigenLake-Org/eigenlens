# Chat analysis worker

Vector Chat connects to Qdrant, retrieves the full intended selection, and uploads its records to this worker. The worker runs PCA/DBSCAN and c-TF-IDF in isolated job processes and returns assignments, coordinates and labels. The chat renders results and plots.

## Recommended startup

Keep compatible `vector-tools` and `vector-chat` checkouts next to each other. From Vector Chat:

```bash
./scripts/run-docker-chat.sh             # default one worker
./scripts/run-docker-chat.sh --workers 3 # three concurrent jobs
```

The script creates a private token once and starts the chat UI and core worker. Optional Qdrant is a separate container. Each job is a process inside the worker container, not a new container. Qdrant credentials stay in the chat service.

## Manual startup

```bash
export VECTOR_CHAT_WORKER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export VECTOR_CHAT_WORKERS=1
docker compose --profile chat-worker up -d --build --wait chat-worker
# Set in the chat process environment:
export VECTOR_CORE_MCP_URL=http://127.0.0.1:8020/mcp
export VECTOR_CORE_MCP_TOKEN="$VECTOR_CHAT_WORKER_TOKEN"
```

The token must match in both processes. Published ports bind to loopback. MCP and upload routes require bearer authentication; `/health` is public. Use a TLS reverse proxy for remote deployment. Large selections use [streamed JSONL uploads](contracts.md#http-jsonl-upload); no shared filesystem is required.

## Workers and limits

`VECTOR_CHAT_WORKERS=N` controls core Compose concurrency, 1–16, default 1. Compose passes `VECTOR_ML_WORKERS` and an 8 GiB shared working-memory estimate. Each job receives `8 GiB / N` as its estimate, advertised in capabilities. The container retains two CPUs and an 8 GiB memory limit. More workers allow concurrent independent jobs; they do not parallelize a single PCA/DBSCAN job. `/health` reports the slot count.

The worker config allows 20,000 rows, 512 MiB serialized input and 30 minutes per job. Oversized selections fail without silent sampling. Change slot counts after active jobs finish: restart interrupts active work. Per-job cancellation stops only that job.

## Persistence

The Compose project remains `vector-ml`. Its `chat-worker-data` named volume contains registered datasets, job state, intermediate data and results. The optional `qdrant` volume contains database records. Stop preserves both:

```bash
docker compose --profile chat-worker stop chat-worker
```

Completed jobs survive restart. Interrupted jobs are marked failed. Use one supervisor per data volume. Automatic artifact retention is not implemented. The simplified Compose file defines only the worker and optional Qdrant; historical containers and volumes are not removed by this source cleanup.
