# Query job server

All query embedding, Qdrant retrieval, PCA/DBSCAN and plot rendering now run on the machine hosting this server. Clients submit small JSON requests and receive durable job IDs. The server uses the existing SQLite job store and a separate spawned process for each job, with queue limits, timeouts and cancellation.

## Run locally

Run from the repository root after indexing BANKING77 with `python -m examples.banking77_qdrant demo`:

```bash
source .venv/bin/activate
pip install -e '.[server,banking,qdrant]'
export VECTOR_QUERY_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export HF_HOME="$PWD/work/huggingface"
python -m examples.query_job_server --port 8010
```

Keep the token and give it to authorized clients. All job and artifact endpoints require `Authorization: Bearer <token>`. Health is available at `GET /health`. The current implementation has one shared principal: token holders can access all jobs. Use it for a trusted team, not isolated customer accounts.

The server owns the Qdrant endpoint, collection, model manifest, output folder and resource limits in `config/query-jobs.json`. Clients can only supply a query and top_k. Run one server process/replica per data directory; a filesystem lock enforces this. Defaults: one active job, 16 pending/active jobs, 600-second worker timeout, at most 2,000 retrieved records.

## Submit and retrieve

In another terminal, set the same `VECTOR_QUERY_TOKEN`:

```bash
curl -X POST http://127.0.0.1:8010/jobs \
  -H "Authorization: Bearer $VECTOR_QUERY_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: atm-demo-1' \
  -d '{"query":"ATM card swallowed or declined", "top_k":500}'
```

The response is HTTP 202 with a `job_id` and `status_url`. Repeating an idempotency key with the same request returns the same job. A changed request with that key returns 409.

| Endpoint | Purpose |
|---|---|
| `POST /jobs` | Submit query and top_k |
| `GET /jobs/{job_id}` | Read queued/running/succeeded/failed/cancelled status; completed summaries and artifact URLs |
| `DELETE /jobs/{job_id}` | Cancel queued work or terminate a running worker |
| `GET /jobs/{job_id}/plot` | Download standalone HTML plot |
| `GET /jobs/{job_id}/report` | Download Markdown report |
| `GET /jobs/{job_id}/results` | Download JSON including ranked matches |
| `GET /jobs/{job_id}/assignments` | Download CSV assignments |

Use the included client to submit, poll and download everything:

```bash
python -m examples.query_job_client 'ATM card swallowed or declined' \
  --server http://127.0.0.1:8010 --output demo-output/server-job
```

Open the downloaded `clusters.html`. Artifact URLs require the authorization header; they are not anonymous browser links. Download first or serve them through your authenticated application.

Job status and artifacts survive restart. Jobs interrupted by restart are marked failed; they are not silently retried. On graceful shutdown, running jobs are cancelled and queued jobs are marked failed. Submit a new request key to intentionally retry an interrupted job.

## Agent client

```bash
export VECTOR_QUERY_URL=http://127.0.0.1:8010
export VECTOR_QUERY_TOKEN='the-server-token'
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-model-id'
export BANKING_QUERY='Show clusters of ATM card problems'
python -m examples.banking77_query_agent
```

The agent's `search_banking_clusters` tool calls the HTTP job client. The host polls without spending model turns and downloads the plot, then returns the summary and plot path to the agent. Computation stays on the job server. The query API itself requires no OpenAI key.

## Docker / deployment

```bash
export VECTOR_QUERY_TOKEN='a-long-random-token'
docker compose --profile query-jobs up -d --build qdrant query-jobs
```

The dedicated image installs CPU PyTorch, MiniLM dependencies and the plotting libraries. It mounts the prepared manifest read-only, uses a persistent model cache and a separate job/artifact volume, and connects to Qdrant on the Compose network. The BANKING77 Qdrant collection must already be indexed; indexing is a setup operation, not a job API action.

For another machine, deploy the repository, Qdrant data (or index there), and the matching prepared manifest, then update the server-owned configuration. Put the service behind an HTTPS reverse proxy for remote clients; Compose binds to loopback by default. Set the client's `VECTOR_QUERY_URL` to that host. No remote cloud deployment has been performed by this task.

Resource budgeting is conservative for DBSCAN's neighborhood graph; this is a single-host job server, not a distributed scheduler. Configure `max_jobs` against available RAM. Results remain on disk until an operator removes them; automatic retention cleanup is not implemented.

## Verified locally

51 tests passed (five skipped). A real authenticated HTTP job queried Qdrant for 500 ATM-card matches and completed with five DBSCAN clusters, 188 clustered records and 312 noise points. The client downloaded HTML, JSON, Markdown and CSV artifacts. Auth, validation, idempotency, process cancellation, timeout and persisted results after restart are covered by tests. Docker Compose configuration validates; the dedicated image has not been built in this session. See [verification](evidence/query-job-server.json).

The currently running development server uses a generated token saved locally in ignored `work/query-job-token`. From the repository root, use `export VECTOR_QUERY_TOKEN="$(cat work/query-job-token)"` to connect with the example client. Do not commit that file.
