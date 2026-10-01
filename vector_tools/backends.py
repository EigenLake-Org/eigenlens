from __future__ import annotations
from typing import Protocol
from dataclasses import dataclass
import uuid
from .registry import plan, REGISTRY
from .temporal import get_frame
from . import operators

@dataclass(frozen=True)
class Partition:
    """One disjoint, bounded partition. Each record belongs to exactly one partition."""
    partition_id: str
    artifact_uri: str
    format: str = "arrow"
    checksum: str = ""
    rows: int | None = None

class Backend(Protocol):
    def capabilities(self) -> dict: ...
    def validate(self, operations, dataset_ref) -> None: ...
    def submit(self, operations, partitions: list[Partition]) -> str: ...
    def status(self, job_id: str) -> dict: ...
    def cancel(self, job_id: str) -> dict: ...
    def results(self, job_id: str) -> dict: ...

class LocalBackend:
    def capabilities(self):
        return {"name": "local", "implemented": True, "operations": list(REGISTRY),
                "execution": "calling process for SDK; dedicated child process for service"}
    def execute(self, data, operations, run_id=None):
        ordered = plan(operations, limits=data.limits)
        if not data.records:
            from .schemas import VTError
            raise VTError("Empty selection: no records to analyze")
        run_id = run_id or "run_" + uuid.uuid4().hex
        results = {}
        frames = {}
        for op in ordered:
            kwargs = {"op_id": op.id, "run_id": run_id, **op.parameters.supplied()}
            if op.clusters_from:
                kwargs["clusters"] = results[op.clusters_from]
            spec = REGISTRY[op.type]
            if spec.get("temporal"):
                # Every temporal op in one plan shares a single TemporalFrame per
                # bucketing: identical bucket assignment, one embedding matrix.
                normalize = op.parameters.normalize
                if "normalize" not in spec.get("frame_params", ()):
                    normalize = False
                key = (op.parameters.time_field or "published_at", op.parameters.bucket or "month",
                       op.parameters.min_bucket_n or 5, True if normalize is None else bool(normalize))
                if key not in frames:
                    frames[key] = get_frame(data, key[0], key[1], key[2], key[3])
                kwargs["frame"] = frames[key]
            if op.type == "sentiment":
                kwargs["model"] = op.model or "configured-default"
            results[op.id] = getattr(operators, op.type)(data, **kwargs)
        return results

SPARK_CAPABILITIES = {"name": "spark", "implemented": False, "operations": [],
                      "requirements": "Disjoint Arrow artifact partitions on distributed-readable storage, installed Spark operators, remote job lifecycle; never wrap a local sklearn estimator as distributed."}
