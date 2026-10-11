"""Lazy plans and bounded materialization with content-addressed provenance."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import numpy as np
from .schemas import Record, Selection, Limits, Predicate, VTError
from .records import matches, validate_selection

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

@dataclass
class Materialized:
    records: list[Record]
    source: dict
    scope: str
    fingerprint: str
    selection: dict
    read_count: int
    missing_ids: list[str]
    limits: Limits
    def vectors(self):
        if not self.records:
            raise VTError("Empty selection: no records to analyze")
        if any(r.vector is None for r in self.records):
            raise VTError("This operation requires dense vectors; materialize with vector projection")
        return np.asarray([r.vector for r in self.records], dtype=np.float64)
    def serializable(self):
        return {"schema_version": "1.0", "records": [r.model_dump(mode="json") for r in self.records],
                "source": self.source, "scope": self.scope, "fingerprint": self.fingerprint,
                "selection": self.selection, "read_count": self.read_count,
                "missing_ids": self.missing_ids, "limits": self.limits.model_dump()}

@dataclass(frozen=True)
class Dataset:
    source: object
    selection: Selection | None = None
    limits: Limits | None = None
    def __post_init__(self):
        object.__setattr__(self, "selection", self.selection or Selection())
        object.__setattr__(self, "limits", self.limits or Limits())
    def ids(self, ids):
        return replace(self, selection=self.selection.model_copy(update={"ids": [str(i) for i in ids]}))
    def filter(self, *predicates, **equals):
        ps = [p if isinstance(p, Predicate) else Predicate.model_validate(p) for p in predicates]
        ps += [Predicate(field=k, value=v) for k, v in equals.items()]
        return replace(self, selection=self.selection.model_copy(update={"filters": self.selection.filters + ps}))
    def sample(self, n, seed=42):
        return replace(self, selection=Selection(**{**self.selection.model_dump(), "sample_n": n, "seed": seed}))
    @property
    def scope(self):
        if self.selection.sample_n is not None:
            return "sample"
        if self.selection.ids is not None:
            return "retrieval_results"
        return "filtered_selection" if self.selection.filters else "complete_selection"
    def materialize(self, fields=None):
        selection, limits = self.selection, self.limits
        validate_selection(selection)
        self.source.validate(selection)
        if selection.sample_n and selection.sample_n > limits.max_rows:
            raise VTError("Sample exceeds max_rows; lower sample_n or explicitly increase limits")
        if selection.ids is not None and len(selection.ids) > limits.max_scan_rows:
            raise VTError("Supplied ID list exceeds max_scan_rows")
        fields = set(fields or {"vector", "text", "metadata"})
        records, heap, seen = [], [], set()
        read_count, byte_count, scan_bytes = 0, 0, 0
        dimension = None
        for batch in self.source.batches(selection, limits.batch_size, fields):
            for r in batch:
                read_count += 1
                if read_count > limits.max_scan_rows:
                    raise VTError("Scan exceeds max_scan_rows; narrow selection or supply IDs. No truncated result produced.")
                if r.id in seen:
                    raise VTError(f"Duplicate ID {r.id} across pages; source changed or adapter ordering is invalid")
                seen.add(r.id)
                if r.vector is not None:
                    if dimension is None:
                        dimension = len(r.vector)
                    elif len(r.vector) != dimension:
                        raise VTError(f"Incompatible vector dimension at {r.id}: expected {dimension}, got {len(r.vector)}")
                size = len(canonical(r.model_dump(mode="json")).encode())
                # Limits bound serialized scan traffic as well as selected data.
                scan_bytes += size
                if size > limits.max_bytes or scan_bytes > limits.max_bytes * 4:
                    raise VTError("Source read exceeds byte budget; narrow selection")
                if not getattr(self.source, "filter_pushdown", False) and not matches(r, selection.filters):
                    continue
                if selection.sample_n:
                    priority = int.from_bytes(hashlib.sha256(f"{selection.seed}\0{r.id}".encode()).digest(), "big")
                    item = (-priority, r.id, size, r)
                    if len(heap) < selection.sample_n:
                        heapq.heappush(heap, item); byte_count += size
                    elif item[:2] > heap[0][:2]:
                        old = heapq.heapreplace(heap, item); byte_count += size - old[2]
                else:
                    if len(records) >= limits.max_rows:
                        raise VTError("Selection exceeds max_rows; use a seeded sample or narrower selection")
                    records.append(r); byte_count += size
                if byte_count > limits.max_bytes:
                    raise VTError("Selected content exceeds max_bytes")
        if selection.sample_n:
            records = [item[3] for item in heap]
        records.sort(key=lambda r: r.id)
        source = dict(self.source.identity)
        digest = hashlib.sha256()
        digest.update(canonical(source).encode())
        for r in records:
            digest.update(canonical(r.model_dump(mode="json")).encode())
        missing = sorted(set(selection.ids or []) - seen)
        return Materialized(records, source, self.scope, digest.hexdigest(), selection.model_dump(), read_count, missing, limits)
