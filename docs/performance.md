# Clustering performance

The search still processes every selected vector and evaluates the entire requested DBSCAN grid. It does not sample records for clustering or change epsilon/min-samples choices. Silhouette evaluation keeps its existing deterministic sample limit.

## Distance and PCA reuse

The worker builds a sparse Euclidean neighbor graph for the normalized PCA vectors, keeping all pairs within the largest requested epsilon. Every DBSCAN trial reuses that graph. Explicit zero-distance edges preserve duplicate vectors. Construction runs in bounded row chunks and checks a conservative 48-byte allowance per edge plus row indices before retaining the graph. If the allowance does not fit the working-memory budget, DBSCAN uses the original vector-based path for every record.

Silhouette evaluation retains at most one distance matrix for its existing deterministic subset (up to `silhouette_sample_size` records). Consecutive trials using the same subset reuse those distances even if their labels differ. A changed subset replaces the matrix. The additional matrix must fit the memory estimate; otherwise scoring uses the original vector-based path. No full collection distance matrix is retained. Results report `distance_cache` as `silhouette_subset` or `disabled_memory_budget`, and `neighborhood_cache` as `radius_graph` or `disabled_memory_budget`.

One full PCA fit supplies both the requested clustering components and the three display components. There is no second SVD solely for the plot. If clustering uses fewer than three components, the fit retains enough display components while the clustering input still uses exactly the requested number.

## Numerical CPU budget

`EIGENLENS_CPU_BUDGET` sets a server-wide numerical-thread budget (1–64). The Docker Compose worker uses 2, matching its two-CPU container limit. Each job gets `max(1, budget // worker_count)` threads. With one worker that is two threads; with two workers it is one per job. More worker slots increase concurrent jobs; each job still evaluates its grid sequentially. Memory is also divided between worker slots as described in the worker guide.

The standalone SDK defaults to one numerical thread. `get_capabilities.execution.numeric_threads_per_job` reports the service setting; clustering results record the actual numerical-thread budget. The Docker job continues to execute inside the worker container.

## Completed-result cache

The MCP service keys completed results on the exact serialized dataset snapshot, scope/selection/source, ordered operation plan, execution backend, limits, numerical threads, analysis source code and numerical/topic library versions. Cache lookup is restricted to the authenticated caller. Successful jobs are reused across new dataset IDs and server restarts. Result files must still be present and valid. Failed, cancelled and still-running jobs do not qualify.

On a hit the service creates an immediately completed job record with `cache_hit: true` and `cached_from_job` pointing to the computation whose result IDs are reused. No child process starts. The result's original execution metadata and creation time remain intact. Normal idempotency checks apply to this new request, including retries and rejection of a reused key with different arguments.

To force a fresh computation, pass `execution: {"use_cache": false}` to `run_analysis`. That computation can populate the cache for subsequent normal requests. Cache lookup does not avoid fetching current database records: the application must register a current snapshot first. A collection name or unchanged point count alone cannot prove the content is unchanged.

Caching is provided by the durable MCP job service; direct synchronous `vt.analyze` calls run their operations in the caller's process. No automatic data/result expiration is implemented.

## BANKING77 verification

A sequential comparison inside the same Docker container processed all 13,069 records and all 42 candidates. The previous clustering code took **261.650 seconds**; the optimized code took **31.455 seconds** (8.32× faster). Process CPU time fell from 261.040 to 43.784 seconds. This measures clustering, scoring, terms/examples and display coordinates; retrieval, registration, BERTopic and chat rendering add time. Timings vary with hardware and load.

The full HTTP MCP analysis preserved all 13,069 assignments, the 146 clusters, 8,487 noise records and every candidate's counts and silhouette score (within 1e-10). A repeated `run_analysis` returned cached result IDs in 6.971 seconds, including snapshot loading, hashing and artifact validation; database retrieval/upload and UI rendering are outside that measurement. The host request timer and Docker wall timestamps disagreed, so the performance comparison above uses only the same-process monotonic timer.
