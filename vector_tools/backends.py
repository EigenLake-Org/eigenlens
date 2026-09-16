from __future__ import annotations
from typing import Protocol
from dataclasses import dataclass
import uuid
from .registry import plan, REGISTRY
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
        for op in ordered:
            kwargs = {"op_id": op.id, "run_id": run_id, **op.parameters.supplied()}
            if op.clusters_from:
                kwargs["clusters"] = results[op.clusters_from]
            if op.type == "sentiment":
                kwargs["model"] = op.model or "configured-default"
            results[op.id] = getattr(operators, op.type)(data, **kwargs)
        return results

SPARK_CAPABILITIES = {"name": "spark", "implemented": False, "operations": [],
                      "requirements": "Disjoint Arrow artifact partitions on distributed-readable storage, installed Spark operators, remote job lifecycle; never wrap a local sklearn estimator as distributed."}
