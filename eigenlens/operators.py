"""Real local algorithms returning portable result contracts."""
from __future__ import annotations
from datetime import datetime, timezone
from importlib.metadata import version
import uuid
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from threadpoolctl import threadpool_limits
from .schemas import Result, VTError
from .execution_context import report_stage

def versions():
    return {"eigenlens": "0.1.0", "numpy": version("numpy"), "scikit-learn": version("scikit-learn")}

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

def matrix(data):
    x = data.vectors()
    if x.nbytes * 8 > data.limits.max_working_bytes:
        raise VTError("Estimated working memory exceeds max_working_bytes")
    return x

def cluster(data, op_id="clusters", run_id=None, algorithm="dbscan_search", **parameters):
    if algorithm != "dbscan_search":
        raise VTError("Supported clustering algorithm: dbscan_search")
    from .registry import plan
    plan([{"id":op_id, "type":"cluster", "algorithm":algorithm, "parameters":parameters}], limits=data.limits)
    from .density import dbscan_search
    return dbscan_search(data, op_id=op_id, run_id=run_id, **parameters)

def validate_clusters(data, clusters):
    if clusters.operation != "cluster" or clusters.dataset_fingerprint != data.fingerprint:
        raise VTError("Clustering result must belong to the same materialized dataset")
    mapping = {r["id"]: r["cluster_id"] for r in clusters.rows}
    if set(mapping) != {r.id for r in data.records}:
        raise VTError("Cluster assignment IDs do not match this dataset")
    return mapping

class RecordVectorizer(CountVectorizer):
    """Keep BERTopic's group documents, but extract phrases within each source record."""
    def __init__(self, boundary, stop_words="english", ngram_range=(1, 1), max_features=20000):
        self.boundary = boundary
        super().__init__(stop_words=stop_words, ngram_range=ngram_range, max_features=max_features)

    def build_analyzer(self):
        analyze = super().build_analyzer()
        def per_record(document):
            return [term for chunk in document.split(self.boundary) for term in analyze(chunk)]
        return per_record


def topics(data, clusters, representation="bertopic", op_id="topics", run_id=None,
           top_n=10, stop_words="english", ngram_max=1):
    if representation != "bertopic":
        raise VTError("Supported topic representation: bertopic")
    if not 1 <= top_n <= 100:
        raise VTError("top_n must be 1..100")
    if type(ngram_max) is not int or not 1 <= ngram_max <= 3:
        raise VTError("ngram_max must be 1..3")
    mapping = validate_clusters(data, clusters)
    ids = sorted(set(mapping.values()))
    empty = sum(not (r.text or "").strip() for r in data.records)
    report_stage('loading_bertopic', 'Loading BERTopic and preparing topic extraction', records=len(data.records))
    from bertopic import BERTopic
    from bertopic.backend import BaseEmbedder
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer
    from scipy.sparse import diags

    class ScalarSafeCtfidf(ClassTfidfTransformer):
        def fit(self, X, multiplier=None):
            if X.shape[1] != 1:
                return super().fit(X, multiplier)
            # BERTopic 0.17.4 squeezes a one-word vocabulary to a scalar, which
            # scipy.diags rejects. Keep its weighting and transform, but retain 1D.
            frequency = np.asarray(X.sum(axis=0)).reshape(-1)
            average = int(X.sum(axis=1).mean())
            idf = np.log1p(average / frequency)
            if multiplier is not None:
                idf *= multiplier
            self._idf_diag = diags(idf, offsets=0, shape=(1, 1), format="csr")
            return self
    boundary = "vectorrecordboundary" + data.fingerprint
    while any(boundary in (r.text or "") for r in data.records):
        boundary += "x"
    vectorizer = RecordVectorizer(boundary, stop_words=stop_words, ngram_range=(1, ngram_max))
    # Manual BERTopic uses existing DBSCAN labels and supplied embeddings. No model download,
    # UMAP fit, HDBSCAN fit or topic reassignment occurs in this operation.
    model = BERTopic(embedding_model=BaseEmbedder(), umap_model=BaseDimensionalityReduction(),
                     hdbscan_model=BaseCluster(), vectorizer_model=vectorizer,
                     ctfidf_model=ScalarSafeCtfidf(), top_n_words=top_n,
                     calculate_probabilities=False, verbose=False)
    labels = [mapping[r.id] for r in data.records]
    documents = [(r.text or "") + " " + boundary for r in data.records]
    topic_ids = {}
    try:
        report_stage('bertopic', 'Extracting topic phrases with BERTopic c-TF-IDF from existing clusters',
                     records=len(data.records), groups=len(ids))
        with threadpool_limits(limits=1):
            assigned, _ = model.fit_transform(documents, embeddings=matrix(data), y=labels)
        report_stage('summarizing_topics', 'Mapping topic phrases back to the original clusters', groups=len(ids))
        # BERTopic renumbers topics by size; export against the original DBSCAN IDs.
        for cid, topic_id in zip(labels, assigned):
            if cid in topic_ids and topic_ids[cid] != int(topic_id):
                raise VTError("BERTopic unexpectedly split an existing cluster")
            topic_ids[cid] = int(topic_id)
        rows = [{"cluster_id":cid, "terms":[{"term":str(term), "weight":float(weight)}
                 for term, weight in (model.get_topic(topic_ids[cid]) or []) if term and weight > 0][:top_n]}
                for cid in ids]
        vocabulary_status = "ok"
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        rows = [{"cluster_id":cid, "terms":[]} for cid in ids]
        vocabulary_status = "empty_vocabulary"
    result = base(data, "topics", op_id, {"representation":"bertopic", "top_n":top_n,
                  "stop_words":stop_words, "max_features":20000, "ngram_range":[1, ngram_max],
                  "embedding_source":"caller supplied", "clustering_source":"existing DBSCAN assignments",
                  "topic_id_map":{str(k):v for k,v in topic_ids.items()}, "weighting":"BERTopic ClassTfidfTransformer"},
                  {"topics":rows, "vocabulary_status":vocabulary_status, "clustering_result_id":clusters.result_id},
                  rows, clusters.example_rows, run_id, {"missing_text":empty})
    result.versions["bertopic"] = version("bertopic")
    result.processed = len(data.records) - empty
    return result
