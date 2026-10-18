"""Copied in-memory records and bounded SDK selection predicates."""
from __future__ import annotations
from .schemas import Record, Selection, Predicate, VTError

MISSING = object()

def matches(record: Record, predicates: list[Predicate]) -> bool:
    for p in predicates:
        v = ({"id": record.id, "text": record.text, "parent_id": record.parent_id}.get(p.field, MISSING)
             if p.field in {"id", "text", "parent_id"} else record.metadata.get(p.field, MISSING))
        exists = v is not MISSING and v is not None
        if p.op == "exists":
            if not isinstance(p.value, bool):
                raise VTError("exists requires a boolean value")
            ok = exists == p.value
        elif not exists:
            ok = False  # Null/missing never satisfies comparisons, including ne.
        else:
            try:
                if p.op == "in":
                    if not isinstance(p.value, list):
                        raise VTError("in requires a list")
                    ok = v in p.value
                else:
                    ok = {"eq": lambda: v == p.value, "ne": lambda: v != p.value,
                          "gt": lambda: v > p.value, "gte": lambda: v >= p.value,
                          "lt": lambda: v < p.value, "lte": lambda: v <= p.value}[p.op]()
            except TypeError:
                ok = False
        if not ok:
            return False
    return True

def validate_selection(selection: Selection):
    for p in selection.filters:
        if p.op == "exists" and not isinstance(p.value, bool):
            raise VTError("exists requires a boolean value")
        if p.op == "in" and not isinstance(p.value, list):
            raise VTError("in requires a list")
        if p.op not in {"in", "exists"} and (p.value is None or isinstance(p.value, list)):
            raise VTError("Comparison predicates require a non-null scalar")
    if selection.ids is not None and len(selection.ids) != len(set(selection.ids)):
        raise VTError("Duplicate retrieval IDs are not allowed")

class InMemory:
    filter_pushdown = False
    def validate(self, selection):
        validate_selection(selection)
    def __init__(self, records, name="memory"):
        self.records = [r.model_copy(deep=True) if isinstance(r, Record) else Record.model_validate(r) for r in records]
        self.identity = {"kind": "memory", "name": name, "embedding_provenance": "caller supplied"}
        if len({r.id for r in self.records}) != len(self.records):
            raise VTError("Duplicate record IDs")
    def batches(self, selection, batch_size, fields):
        self.validate(selection)
        ids = set(selection.ids) if selection.ids is not None else None
        batch = []
        for r in sorted(self.records, key=lambda r: r.id):
            if ids is None or r.id in ids:
                batch.append(r.model_copy(deep=True))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
