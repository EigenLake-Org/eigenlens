"""Real local algorithms returning portable result contracts."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import math
import uuid
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as unit_normalize
from threadpoolctl import threadpool_limits
from .schemas import Result, VTError

def versions():
    return {"vector-tools": "0.1.0", "numpy": version("numpy"), "scikit-learn": version("scikit-learn")}

def example(r, **extra):
    return {"id": r.id, "parent_id": r.parent_id, "text": (r.text or "")[:1000], **extra}

def base(data, kind, op_id, params, summary, rows, examples, run_id=None, skipped=None):
    return Result(result_id="res_" + uuid.uuid4().hex, run_id=run_id or "run_" + uuid.uuid4().hex,
                  operation_id=op_id, operation=kind, dataset_fingerprint=data.fingerprint,
                  scope=data.scope, source=data.source, processed=len(rows) if kind != "topics" else len(data.records),
                  skipped={"missing_ids": len(data.missing_ids), "invalid_vectors": 0, **(skipped or {})},
                  effective_parameters=params, versions=versions(), created_at=datetime.now(timezone.utc).isoformat(),
                  summary_data={"selected_records": len(data.records), "unique_parent_ids": len({r.parent_id for r in data.records if r.parent_id is not None}),
                                "source_read_count": data.read_count, "selection": data.selection, **summary},
                  rows=rows, example_rows=examples)

def matrix(data, normalize=False):
    x = data.vectors()
    # Conservative matrix-copy reserve. Specific operator allocations checked below.
    if x.nbytes * 8 > data.limits.max_working_bytes:
        raise VTError("Estimated working memory exceeds max_working_bytes")
    return unit_normalize(x) if normalize else x

def cluster(data, estimator=None, op_id="clusters", run_id=None, algorithm="minibatch_kmeans", **parameters):
    if algorithm == "dbscan_search":
        if estimator is not None:
            raise VTError("dbscan_search does not accept an estimator")
        from .density import dbscan_search
        return dbscan_search(data, op_id=op_id, run_id=run_id, **parameters)
    if algorithm != "minibatch_kmeans":
        raise VTError("Unknown clustering algorithm")
    params = dict(parameters)
    normal = params.pop("normalize", False)
    x = matrix(data, normal)
    if estimator is not None and type(estimator) is not MiniBatchKMeans:
        raise VTError("Supported estimator protocol: exact sklearn MiniBatchKMeans; custom estimators require a registered implementation")
    if estimator is None:
        params = {"n_clusters": min(data.limits.max_clusters, max(1, min(8, int(math.sqrt(len(x)))))), "random_state": 42, "n_init": 3, **params}
        estimator = MiniBatchKMeans(**params)
    n = estimator.n_clusters
    if not 1 <= n <= min(len(x), data.limits.max_clusters):
        raise VTError("n_clusters must be <= usable records and max_clusters")
    if x.nbytes * 8 + len(x) * n * 8 > data.limits.max_working_bytes:
        raise VTError("Cluster distance matrix exceeds working memory budget")
    with threadpool_limits(limits=1):
        labels = estimator.fit_predict(x)
        distances = estimator.transform(x)
    rows = [{"id": r.id, "cluster_id": int(label)} for r, label in zip(data.records, labels)]
    examples = []
    for cid in range(n):
        indices = np.flatnonzero(labels == cid)
        chosen = sorted(indices, key=lambda i: (distances[i, cid], data.records[i].id))[:5]
        examples.extend(example(data.records[i], cluster_id=cid, distance=float(distances[i, cid])) for i in chosen)
    effective = {**estimator.get_params(), "normalize": normal, "distance_assumption": "Euclidean on L2-normalized embeddings" if normal else "Euclidean on supplied embeddings"}
    return base(data, "cluster", op_id, effective,
                {"cluster_sizes": dict(sorted(Counter(map(str, labels)).items())), "centroids": estimator.cluster_centers_.tolist(), "inertia": float(estimator.inertia_)}, rows, examples, run_id)

def validate_clusters(data, clusters):
    if clusters.operation != "cluster" or clusters.dataset_fingerprint != data.fingerprint:
        raise VTError("Clustering result must belong to the same materialized dataset")
    mapping = {r["id"]: r["cluster_id"] for r in clusters.rows}
    if set(mapping) != {r.id for r in data.records}:
        raise VTError("Cluster assignment IDs do not match this dataset")
    return mapping

def topics(data, clusters, representation="ctfidf", op_id="topics", run_id=None, top_n=10, stop_words="english"):
    if representation != "ctfidf":
        raise VTError("Supported topic representation: ctfidf")
    if not 1 <= top_n <= 100:
        raise VTError("top_n must be 1..100")
    mapping = validate_clusters(data, clusters)
    ids = sorted(set(mapping.values()))
    docs = [" ".join(r.text or "" for r in data.records if mapping[r.id] == cid) for cid in ids]
    empty = sum(not (r.text or "").strip() for r in data.records)
    try:
        v = CountVectorizer(stop_words=stop_words, max_features=20000)
        counts = v.fit_transform(docs).astype(float)
        tf = unit_normalize(counts, norm="l1", axis=1)
        average = float(np.asarray(counts.sum(axis=1)).mean())
        idf = np.log1p(average / np.asarray(counts.sum(axis=0)).ravel())
        weighted = tf.multiply(idf).tocsr()
        names = v.get_feature_names_out()
        rows = []
        for i, cid in enumerate(ids):
            row = weighted.getrow(i)
            terms = sorted(zip(row.indices, row.data), key=lambda x: (-x[1], names[x[0]]))[:top_n]
            rows.append({"cluster_id": cid, "terms": [{"term": str(names[j]), "weight": float(w)} for j, w in terms]})
        vocabulary_status = "ok"
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        rows = [{"cluster_id": cid, "terms": []} for cid in ids]
        vocabulary_status = "empty_vocabulary"
    result = base(data, "topics", op_id, {"representation": "ctfidf", "top_n": top_n, "stop_words": stop_words, "max_features": 20000,
                                         "weighting": "L1 class term frequency * log(1 + mean class token count / total term frequency)"},
                  {"topics": rows, "vocabulary_status": vocabulary_status, "clustering_result_id": clusters.result_id}, rows,
                  clusters.example_rows, run_id, {"missing_text": empty})
    result.processed = len(data.records) - empty
    return result

class VaderModel:
    """English lexicon/rule model; weights are shipped by vaderSentiment, no network downloads."""
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.model = SentimentIntensityAnalyzer()
        except (ImportError, OSError) as exc:
            raise VTError("VADER weights unavailable. Install vector-tools[sentiment]; bundled lexicons must be readable.") from exc
        self.version = version("vaderSentiment")
        self.fingerprint = hashlib.sha256(self.model.lexicon_full_filepath.encode()).hexdigest()
    def infer(self, texts):
        return [self.model.polarity_scores(t) for t in texts]

def sentiment(data, model="configured-default", clusters=None, op_id="sentiment", run_id=None,
              batch_size=128, positive_threshold=0.05, negative_threshold=-0.05):
    if model not in {"configured-default", "vader"}:
        raise VTError("Configured sentiment model is vader; requested model unavailable")
    if not 1 <= batch_size <= 10000 or not -1 <= negative_threshold < positive_threshold <= 1:
        raise VTError("Invalid batch size or sentiment thresholds")
    adapter = VaderModel()
    mapping = validate_clusters(data, clusters) if clusters else None
    valid = [r for r in data.records if (r.text or "").strip()]
    rows = []
    for start in range(0, len(valid), batch_size):
        batch = valid[start:start+batch_size]
        for r, scores in zip(batch, adapter.infer([r.text for r in batch])):
            label = "positive" if scores["compound"] >= positive_threshold else "negative" if scores["compound"] <= negative_threshold else "neutral"
            rows.append({"id": r.id, "label": label, "scores": scores, **({"cluster_id": mapping[r.id]} if mapping is not None else {})})
    groups = {}
    if mapping is not None:
        for cid in sorted(set(mapping.values())):
            selected = [r for r in rows if r["cluster_id"] == cid]
            groups[str(cid)] = {"denominator": len(selected), "missing_text": sum(mapping[r.id] == cid and not (r.text or "").strip() for r in data.records),
                                "labels": dict(Counter(r["label"] for r in selected)),
                                "mean_compound": float(np.mean([r["scores"]["compound"] for r in selected])) if selected else None}
    result = base(data, "sentiment", op_id,
                  {"model": "vader", "batch_size": batch_size, "positive_threshold": positive_threshold, "negative_threshold": negative_threshold,
                   "model_version": adapter.version, "lexicon_sha256": adapter.fingerprint,
                   "score_meaning": "compound in [-1,1]; pos/neu/neg lexicon proportions, not calibrated probabilities", "language": "English"},
                  {"labels": dict(Counter(r["label"] for r in rows)), "denominator": len(rows), "by_cluster": groups}, rows,
                  [example(r, **{k:v for k,v in score.items() if k != "id"}) for r, score in list(zip(valid, rows))[:20]], run_id,
                  {"missing_text": len(data.records) - len(valid)})
    result.versions["vaderSentiment"] = adapter.version
    return result

def outliers(data, estimator=None, op_id="outliers", run_id=None, **parameters):
    params = dict(parameters)
    normal = params.pop("normalize", False)
    x = matrix(data, normal)
    if estimator is not None and type(estimator) is not IsolationForest:
        raise VTError("Supported estimator protocol: exact sklearn IsolationForest")
    if estimator is None:
        estimator = IsolationForest(**{"random_state": 42, "n_jobs": 1, **params})
    if estimator.n_estimators > 1000:
        raise VTError("n_estimators exceeds 1000")
    with threadpool_limits(limits=1):
        estimator.fit(x)
        raw = estimator.score_samples(x)
        decision = estimator.decision_function(x)
    rows = sorted([{"id": r.id, "raw_score": float(s), "anomaly_score": float(-s), "decision_function": float(d)}
                   for r, s, d in zip(data.records, raw, decision)], key=lambda r: (-r["anomaly_score"], r["id"]))
    by_id = {r.id: r for r in data.records}
    return base(data, "outliers", op_id, {**estimator.get_params(), "normalize": normal, "score_transform": "anomaly_score = -score_samples; higher is more anomalous"},
                {"flagged_count": int(np.sum(decision < 0)), "score_direction": "higher is more anomalous"}, rows,
                [example(by_id[r["id"]], **{k:v for k,v in r.items() if k != "id"}) for r in rows[:100]], run_id)
