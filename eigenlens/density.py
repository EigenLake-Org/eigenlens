"""Bounded PCA/DBSCAN search; labels are used only for post-hoc evaluation."""
from collections import Counter
from scipy.sparse import vstack
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .operators import base, example, matrix
from .schemas import Parameters, VTError
from .execution_context import numeric_threads, report_stage


def dbscan_search(data, op_id="clusters", run_id=None, pca_components=50,
                  eps_values=None, min_samples_values=None, max_noise_ratio=.7,
                  silhouette_sample_size=4000, random_state=42, label_field=None):
    eps_values = [round(.3 + .05*i, 2) for i in range(11)] if eps_values is None else eps_values
    min_samples_values = [5, 10, 15] if min_samples_values is None else min_samples_values
    params = dict(pca_components=pca_components, eps_values=eps_values,
                  min_samples_values=min_samples_values, max_noise_ratio=max_noise_ratio,
                  silhouette_sample_size=silhouette_sample_size, random_state=random_state,
                  label_field=label_field)
    Parameters(**params)  # SDK calls get the same validation as MCP calls.
    if len(eps_values) * len(min_samples_values) > 64:
        raise VTError("DBSCAN search is limited to 64 parameter combinations")
    report_stage('validating_vectors', 'Validating vectors and checking the memory budget', selected_records=len(data.records))
    x = matrix(data)
    n, dim = x.shape
    if n < 3 or np.any(np.linalg.norm(x, axis=1) == 0):
        raise VTError("DBSCAN requires at least three nonzero vectors")
    # sklearn DBSCAN can retain a dense neighborhood graph in the worst case.
    reserve = x.nbytes * 8 + n*n*24 + min(n, silhouette_sample_size)**2*8
    if reserve > data.limits.max_working_bytes:
        raise VTError("DBSCAN worst-case memory exceeds max_working_bytes; sample fewer records or explicitly raise the budget")
    # Retain at most one silhouette subset matrix, never all n-by-n distances.
    sample_bytes = min(n, silhouette_sample_size)**2*8
    cache_distances = reserve + sample_bytes <= data.limits.max_working_bytes
    graph_budget = data.limits.max_working_bytes - reserve - (sample_bytes if cache_distances else 0)
    with threadpool_limits(limits=numeric_threads()):
        report_stage('normalizing_vectors', 'Normalizing vectors before PCA', records=n, dimensions=dim)
        x = normalize(x)
        if not np.any(np.var(x, axis=0) > 0):
            raise VTError("DBSCAN requires variation between normalized vectors")
        effective_components = min(pca_components, n-1, dim)
        report_stage('pca', 'Reducing vector dimensions with PCA', records=n, input_dimensions=dim,
                     clustering_dimensions=effective_components)
        pca = PCA(n_components=max(effective_components, min(3, n-1, dim)), random_state=random_state, svd_solver="full")
        scores = pca.fit_transform(x)
        reduced = normalize(scores[:, :effective_components])
        projection = scores[:, :min(3, n-1, dim)].copy()
        neighborhoods = None
        chunk_rows = min(128, graph_budget // (n*48))
        if chunk_rows:
            report_stage('neighbor_graph', 'Building reusable DBSCAN neighborhoods', completed=0, total=n, unit='records')
            neighbors = NearestNeighbors(metric="euclidean", algorithm="brute", n_jobs=1).fit(reduced)
            parts, edges = [], 0
            for start in range(0, n, chunk_rows):
                part = neighbors.radius_neighbors_graph(reduced[start:start+chunk_rows],
                    radius=max(eps_values), mode="distance", sort_results=True)
                edges += part.nnz
                # Allow for graph assembly, DBSCAN copies and neighborhood arrays.
                if edges*48 + (n+1)*16 > graph_budget:
                    break
                parts.append(part)
                report_stage('neighbor_graph', 'Building reusable DBSCAN neighborhoods',
                             completed=min(start+chunk_rows, n), total=n, unit='records')
            else:
                neighborhoods = vstack(parts, format="csr")
            del parts, part, neighbors
        distances, distance_indices = None, None
        search_input = neighborhoods if neighborhoods is not None else reduced
        search, best, best_labels = [], None, None
        trials = len(eps_values) * len(min_samples_values)
        for eps in eps_values:
            for min_samples in min_samples_values:
                report_stage('dbscan_search', 'Evaluating DBSCAN parameters and silhouette scores',
                             completed=len(search), total=trials, unit='trials',
                             eps=float(eps), min_samples=int(min_samples),
                             neighborhood_cache=neighborhoods is not None)
                labels = DBSCAN(eps=eps, min_samples=min_samples,
                    metric="precomputed" if neighborhoods is not None else "euclidean", n_jobs=1).fit_predict(search_input)
                count = len(set(labels) - {-1})
                indices = np.flatnonzero(labels != -1)
                noise = 1 - len(indices)/n
                if len(indices) > silhouette_sample_size:
                    indices = np.sort(np.random.default_rng(random_state).choice(indices, silhouette_sample_size, replace=False))
                unique = len(set(labels[indices]))
                score = None
                if 2 <= unique < len(indices):
                    if cache_distances:
                        if distance_indices is None or not np.array_equal(indices, distance_indices):
                            distances = None  # Release the previous subset before allocating.
                            distances = pairwise_distances(reduced[indices], metric="euclidean", n_jobs=1)
                            np.fill_diagonal(distances, 0.)
                            distance_indices = indices.copy()
                        score = float(silhouette_score(distances, labels[indices], metric="precomputed"))
                    else:
                        score = float(silhouette_score(reduced[indices], labels[indices]))
                candidate = dict(eps=float(eps), min_samples=int(min_samples), clusters=count,
                                 noise_percentage=100*noise, silhouette=score,
                                 silhouette_records=len(indices))
                search.append(candidate)
                if (2 <= count <= data.limits.max_clusters and noise < max_noise_ratio
                        and score is not None and (best is None or score > best["silhouette"])):
                    best, best_labels = candidate, labels.copy()
        report_stage('dbscan_search', 'Finished evaluating DBSCAN parameters', completed=len(search), total=trials, unit='trials')
        if best is None:
            # Preserve search diagnostics; do not silently choose an invalid candidate.
            labels = np.full(n, -1, dtype=int)
            status = "no_valid_candidate"
        else:
            labels = best_labels
            status = "selected"
        neighborhoods_cached = neighborhoods is not None
        del search_input, neighborhoods, distances
    report_stage('summarizing_clusters', 'Collecting cluster terms, representative examples and plot coordinates',
                 records=n, clusters=len(set(labels) - {-1}))
    projection = np.pad(projection, ((0, 0), (0, 3-projection.shape[1])))
    terms = {int(cid): [] for cid in set(labels)}
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=5000, ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([r.text or "" for r in data.records])
        names = vectorizer.get_feature_names_out()
        for cid in terms:
            if cid == -1:
                continue
            weights = np.asarray(tfidf[labels == cid].mean(axis=0)).ravel()
            terms[cid] = [str(names[i]) for i in np.argsort(-weights, kind="stable")[:6] if weights[i] > 0]
        vocabulary_status = "ok"
    except ValueError as exc:
        if not any(s in str(exc) for s in ("empty vocabulary", "After pruning", "max_df corresponds")):
            raise
        vocabulary_status = "empty_vocabulary"
    truth = [r.metadata.get(label_field) for r in data.records] if label_field else []
    has_truth = bool(truth) and all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in truth)
    truth = [str(v) for v in truth] if has_truth else []
    rows, examples, groups = [], [], []
    for cid in sorted(terms):
        idx = np.flatnonzero(labels == cid)
        group = dict(cluster_id=cid, queries=len(idx), top_terms=terms[cid], is_noise=cid == -1)
        if truth:
            counts = Counter(truth[i] for i in idx)
            dominant, amount = sorted(counts.items(), key=lambda v: (-v[1], v[0]))[0]
            group.update(dominant_intent=dominant, purity_percent=100*amount/len(idx), intent_counts=dict(counts))
        groups.append(group)
        center = reduced[idx].mean(axis=0)
        chosen = sorted(idx, key=lambda i: (float(np.linalg.norm(reduced[i]-center)), data.records[i].id))[:5]
        examples.extend(example(data.records[i], cluster_id=cid) for i in chosen)
    for i, r in enumerate(data.records):
        rows.append(dict(id=r.id, cluster_id=int(labels[i]), pca_3d=projection[i].tolist()))
    evaluation = {"status": "unavailable", "reason": "Supply label_field with a string or integer label on every record"}
    if truth and best is not None:
        evaluation = dict(status="evaluated", adjusted_rand_index=float(adjusted_rand_score(truth, labels)),
                          normalized_mutual_information=float(normalized_mutual_info_score(truth, labels)),
                          includes_noise=True, labels_used_for_selection=False)
    return base(data, "cluster", op_id, {**params, "algorithm": "dbscan_search", "effective_pca_components": effective_components,
                "normalization": "L2 before PCA and after PCA", "metric": "euclidean", "best_parameters": best,
                "distance_cache": "silhouette_subset" if cache_distances else "disabled_memory_budget",
                "neighborhood_cache": "radius_graph" if neighborhoods_cached else "disabled_memory_budget",
                "numeric_threads": numeric_threads()},
                dict(status=status, parameter_search=search, groups=groups,
                     cluster_sizes=dict(Counter(map(str, labels))), noise_count=int(np.sum(labels == -1)),
                     noise_percentage=100*float(np.mean(labels == -1)), evaluation=evaluation,
                     explained_variance=float(pca.explained_variance_ratio_[:effective_components].sum()), vocabulary_status=vocabulary_status,
                     selection_rule="Highest non-noise sampled silhouette with >=2 clusters and noise below threshold; first grid entry breaks ties",
                     caveat="3D PCA is a visualization, not the clustering space. Noise -1 is not an intent. If no candidate qualifies, all rows are marked unassigned (-1)."),
                rows, examples, run_id)
