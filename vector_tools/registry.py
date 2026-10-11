"""Portable operation validation and dependency ordering."""
from .schemas import Operation, Limits, VTError

DBSCAN_PARAMS = {"pca_components", "eps_values", "min_samples_values", "max_noise_ratio", "silhouette_sample_size", "random_state", "label_field"}
REGISTRY = {
    "cluster": {"algorithm":"dbscan_search", "fields":{"vector"}, "params":DBSCAN_PARAMS,
                "algorithms":{"dbscan_search":DBSCAN_PARAMS}},
    "topics": {"algorithm":"bertopic", "fields":{"vector", "text"}, "params":{"top_n", "stop_words", "ngram_max"}},
}

def plan(operations, backend="auto", limits=None):
    limits = limits or Limits()
    if backend not in {"auto", "local"}:
        raise VTError("Supported execution backend: local.")
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
            if o.type != "topics":
                raise VTError("Only topics accept clusters_from")
            if o.clusters_from not in by_id or by_id[o.clusters_from].type != "cluster":
                raise VTError("clusters_from must reference a cluster operation in this plan")
        if o.representation and o.type != "topics":
            raise VTError("representation applies only to topics")
    ordered, pending = [], list(ops)
    while pending:
        ready = [o for o in pending if not o.clusters_from or o.clusters_from in {v.id for v in ordered}]
        if not ready:
            raise VTError("Cyclic operation dependencies")
        for o in ready:
            ordered.append(o); pending.remove(o)
    return ordered
