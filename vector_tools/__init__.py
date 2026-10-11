"""Clustering SDK for caller-supplied vector records; MCP remains optional."""
from .datasets import Dataset, Materialized
from .schemas import Record, Predicate, Limits, Operation, Result, VTError
from .records import InMemory
from .operators import cluster, topics
from .backends import LocalBackend

__version__ = "0.1.0"

def from_records(records, name="memory", limits=None):
    return Dataset(InMemory(records, name), limits=limits)

def analyze(data, operations, backend="auto"):
    from .registry import plan
    return LocalBackend().execute(data, plan(operations, backend, data.limits))
