# Separate chat analysis worker

This worker uses the existing core HTTP MCP server and supervisor. Each analysis runs in a spawned process inside the container, with a persistent `/data` volume. It is separate from the generic service on port 8000 and the older BANKING77 query demo on port 8010.

## Recommended setup

Clone compatible `vector-tools` and `vector-chat` checkouts next to each other. Install Vector Chat in a virtual environment and run its `scripts/run-worker-chat.sh`. The script generates a private token once, starts this service, and passes the authenticated MCP URL to the chat. The chat performs Qdrant retrieval and uploads the complete selection; this worker performs PCA, DBSCAN search and c-TF-IDF. Qdrant credentials remain on the chat host.

## Manual setup

```bash
export VECTOR_CHAT_WORKER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
docker compose --profile chat-worker up -d --build --wait chat-worker
# In the chat process environment:
export VECTOR_CORE_MCP_URL=http://127.0.0.1:8020/mcp
export VECTOR_CORE_MCP_TOKEN="$VECTOR_CHAT_WORKER_TOKEN"
```

The token must match in both environments. Published ports bind to loopback. `/health` is unauthenticated; MCP and upload routes require bearer authentication. Use a TLS reverse proxy for any remote network deployment.

## Limits and lifecycle

`config/chat-worker.json` allows 20,000 rows, 512 MiB of serialized input, an 8 GiB working-memory estimate, one concurrent analysis and a 30-minute timeout. Compose also enforces an 8 GiB container memory limit and two CPUs. The memory estimate is not an exact RSS guarantee. Increasing the row limit requires reviewing both algorithm estimates and container memory. Oversized selections fail; the chat never samples them.

The `chat-worker-data` named volume contains jobs, registered datasets, intermediate data and results. Stop preserves it:

```bash
docker compose --profile chat-worker stop chat-worker
```

Completed jobs survive restart. Jobs interrupted by service restart are marked failed. `cancel_job` terminates a pending/running worker and retains completed results if completion wins the race. Do not run two supervisors against the same data volume. Retention and automatic artifact cleanup are not implemented.

For upload details see [the HTTP JSONL contract](contracts.md#http-jsonl-upload). Stream upload, registration and cleanup let a remote client use the worker without a shared filesystem. Analysis result summaries include the execution hostname and process ID so callers can verify container execution.
