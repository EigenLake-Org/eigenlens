# Try the Food Reviews Demo

Analyze **1,000 real food reviews with existing OpenAI embeddings** using Qdrant, the MCP service, and four local ML operations. The default demo needs **Docker only—no OpenAI API key or host Python installation**.

## 1. Start everything

Download/unzip the project, open Terminal in its directory, and run:

```bash
./scripts/try-food-demo.sh
```

Requirements: Docker Desktop (or Docker Engine with Compose v2), Bash on macOS/Linux/WSL, internet access for the first download/build, and available localhost ports 6333 and 8000. Allow several minutes for the first image build. Reruns reuse cached Docker layers and the verified dataset download.

The script:

1. Builds and starts Qdrant plus the MCP service.
2. Downloads a pinned 35 MB CSV from the official OpenAI Cookbook repository.
3. Verifies its SHA-256, expected row count, unique IDs, text, ratings and every 1,536-dimensional vector.
4. Upserts stable IDs into a separate `food_reviews` collection, then checks all stored records.
5. Calls the real MCP tools to run clustering, c-TF-IDF topics, VADER sentiment and IsolationForest.
6. Writes a readable report and a JSON trace under `demo-output/`.

**No new embeddings are generated. No LLM is called in this mode.** The report uses real computed results and representative source records; it is not a model-stub demonstration.

## 2. Inspect the results

Open these files after the command completes:

- `demo-output/food-review-report.md`: cluster sizes, top terms, sentiment, representative reviews, unusual reviews and a comparison with star ratings.
- `demo-output/food-review-report.json`: dataset/job/result handles, summaries, source provenance and actual MCP calls.

Open the [Qdrant dashboard](http://localhost:6333/dashboard) and select the `food_reviews` collection to inspect stored points, text, star ratings and metadata. The [MCP endpoint](http://localhost:8000/mcp) is a protocol endpoint, not a webpage.

The default run analyzes the complete 1,000-record collection with **8 clusters**, seed **42**, and the four operations. This collection itself is a sample of the original review corpus; findings do not represent all Amazon reviews.

## 3. Try a live OpenAI agent

Set your application credentials and rerun in agent mode:

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_MODEL='your-available-model-id'
./scripts/try-food-demo.sh agent
```

The agent discovers the `food_reviews` source through MCP, submits analysis, returns a pending job if needed, and resumes after host-side polling. It then inspects summaries/examples and explains the findings. The transcript is saved to `demo-output/food-review-agent.json` and its answer prints in the terminal.

This mode makes live model API calls. The API key is passed to the demo client container; the MCP service does not need it. The embeddings are still reused. Live model execution is unverified in the development environment because no OpenAI credentials were available.

For an already-running setup, skip loading/building and just rerun the client:

```bash
mkdir -p demo-output
docker compose run --rm -T --interactive=false --no-deps --user "$(id -u):$(id -g)" food-client \
  python -m examples.food_reviews analyze --cache-dir /cache --output /output
```

Replace `analyze` with `agent` to run the live agent. Use `--clusters 12` to change the number of groups in either mode (within service limits).

## What data is included?

The source is [fine_food_reviews_with_embeddings_1k.csv](https://github.com/openai/openai-cookbook/blob/main/examples/data/fine_food_reviews_with_embeddings_1k.csv) from OpenAI's Cookbook. [The lock file](../config/food_reviews.lock.json) records the immutable commit URL, checksum, dimensions and expected count. The demo fetches the dataset at setup time; the source archive does not bundle the CSV.

| Property | Value |
|---|---:|
| Review records | 1,000 |
| Embedding dimensions | 1,536 |
| CSV bytes | 35,046,315 |
| 1-star reviews | 87 |
| 2-star reviews | 49 |
| 3-star reviews | 75 |
| 4-star reviews | 138 |
| 5-star reviews | 651 |

The CSV contains precomputed OpenAI vectors but does not explicitly declare their exact model identity. The loader records that uncertainty rather than guessing a model from vector dimensions. This demo does not need query embeddings; clustering and outlier detection use the stored vectors directly. Do not mix embeddings from an assumed model into this collection.

The original `combined` text is stored as payload `text`; vectors are named `body`; star ratings are `score`; `product_id`, original `review_id`, `parent_id`, source URL and dataset checksum are retained. Public reviewer user IDs are not needed by the demo and are not indexed. Point UUIDs are deterministic from dataset checksum and original review ID.

VADER labels are not star ratings. The report joins each scored review to its original star rating using its point ID and reports mean compound scores by rating. The positive-heavy rating distribution and limitations of English lexicon sentiment remain visible.

## How MCP finds the collection

`food_reviews` is a configured and authorized source in both `config/local.json` and `config/docker.json`. The Docker service connects to `http://qdrant:6333`; a host process connects to `http://localhost:6333`. Existing ticket/demo source mappings remain available.

The source's collection mapping is:

```json
{
  "collection": "food_reviews",
  "vector_field": "body",
  "text_field": "text",
  "parent_field": "parent_id"
}
```

The demo submits `create_dataset(source_id="food_reviews", selection={})` and then a background analysis plan. Full embeddings remain in Qdrant and the worker. MCP returns handles, compact summaries, paged outputs and bounded examples.

## Reruns and persistence

- Qdrant data, ML results and the dataset download use named Docker volumes.
- Stable IDs make reruns upsert the same 1,000 reviews.
- The loader refuses incompatible vector settings or an existing collection containing unrelated data. It never deletes a collection to make setup succeed.
- A failed partial load can be retried: already-written review IDs are updated.
- Each analysis creates a new durable job; the report files are replaced with that latest run.
- The default token `local-demo-token` and published loopback ports are intended for local demos. Set `VECTOR_ML_MCP_TOKEN` consistently before running the script if you want a different token.

Stop the demo services while retaining their data:

```bash
docker compose stop service qdrant
```

## Troubleshooting

- **Docker not running:** launch Docker Desktop and rerun the script.
- **Port already in use:** check for an existing service on 6333/8000 before starting another copy.
- **Checksum mismatch:** the loader stops before indexing; retain the pinned lock file and retry the download. Do not bypass verification to use unknown content.
- **Unauthorized:** use the same `VECTOR_ML_MCP_TOKEN` for the service and client. The script handles this when run from a consistent environment.
- **Source unavailable:** rebuild the service so its copied configuration includes `food_reviews`: `docker compose up -d --build service`.
- **Collection contains other data:** choose another collection and update the corresponding source mapping; do not delete unrelated records.
- **Agent credentials missing:** use the default no-key mode or set both `OPENAI_API_KEY` and `OPENAI_MODEL`.

## Host Python alternative

For development using the existing virtual environment:

```bash
source .venv/bin/activate
python -m examples.food_reviews setup
export VECTOR_ML_MCP_URL=http://localhost:8000/mcp
export VECTOR_ML_MCP_TOKEN=local-demo-token
python -m examples.food_reviews analyze
```

The first command indexes into host-accessible Qdrant; the second analyzes through the running Docker MCP service. See the module's `--help` for collection, source, cache, output and cluster options.
