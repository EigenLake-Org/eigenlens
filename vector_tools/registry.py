"""Portable operation validation and dependency ordering."""
from .schemas import Operation, Limits, VTError

REGISTRY = {
    "cluster": {"algorithm": "minibatch_kmeans", "fields": {"vector"}, "params": {"n_clusters", "random_state", "batch_size", "n_init", "max_iter", "normalize"}},
    "topics": {"algorithm": "ctfidf", "fields": {"text"}, "params": {"top_n", "stop_words"}},
    "sentiment": {"algorithm": "vader", "fields": {"text"}, "params": {"batch_size", "positive_threshold", "negative_threshold"}},
    "outliers": {"algorithm": "isolation_forest", "fields": {"vector"}, "params": {"n_estimators", "random_state", "max_samples", "contamination", "normalize"}},
}

DBSCAN_PARAMS = {"pca_components", "eps_values", "min_samples_values", "max_noise_ratio", "silhouette_sample_size", "random_state", "label_field"}
REGISTRY["cluster"]["algorithms"] = {
    "minibatch_kmeans": REGISTRY["cluster"]["params"],
    "dbscan_search": DBSCAN_PARAMS,
}

def plan(operations, backend="auto", limits=None):
    limits = limits or Limits()
    if backend not in {"auto", "local"}:
        raise VTError("Spark execution is unimplemented. Supported backend: local; no algorithm substitution is performed.")
    ops = [o if isinstance(o, Operation) else Operation.model_validate(o) for o in operations]
    if not 1 <= len(ops) <= 16:
        raise VTError("A plan must contain 1..16 operations")
    by_id = {o.id: o for o in ops}
    if len(by_id) != len(ops):
        raise VTError("Operation IDs must be unique")
    for o in ops:
        spec = REGISTRY[o.type]
        algorithms = spec.get("algorithms", {spec["algorithm"]: spec["params"]})
        algorithm = o.algorithm or spec["algorithm"]
        if algorithm not in algorithms:
            raise VTError(f"{o.type} supports algorithms {sorted(algorithms)}")
        invalid = o.parameters.supplied().keys() - algorithms[algorithm]
        if invalid:
            raise VTError(f"Unsupported parameters for {o.type}: {sorted(invalid)}")
        if o.type == "topics" and not o.clusters_from:
            raise VTError("Topics require clusters_from referencing a cluster operation")
        if o.clusters_from:
            if o.type not in {"topics", "sentiment"}:
                raise VTError("Only topics and sentiment accept clusters_from")
            if o.clusters_from not in by_id or by_id[o.clusters_from].type != "cluster":
                raise VTError("clusters_from must reference a cluster operation in this plan")
        if o.representation and o.type != "topics":
            raise VTError("representation applies only to topics")
        if o.model and (o.type != "sentiment" or o.model not in {"vader", "configured-default"}):
            raise VTError("Available sentiment model: vader (configured-default)")
        if (o.parameters.n_clusters or 0) > limits.max_clusters:
            raise VTError("n_clusters exceeds configured max_clusters")
    ordered, pending = [], list(ops)
    while pending:
        ready = [o for o in pending if not o.clusters_from or o.clusters_from in {v.id for v in ordered}]
        if not ready:
            raise VTError("Cyclic operation dependencies")
        for o in ready:
            ordered.append(o); pending.remove(o)
    return ordered
