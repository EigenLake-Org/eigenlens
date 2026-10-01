"""Composable synchronous SDK; connector clients, MCP and Agents remain optional."""
from .datasets import Dataset, Materialized
from .schemas import Record, Predicate, Limits, Operation, Result, VTError
from .sources import InMemory, Qdrant, Weaviate, Pgvector, Pinecone
from .operators import cluster, topics, sentiment, outliers, drift, novelty, flow
from .temporal import TemporalFrame, clear_frame_cache, get_frame
from .evidence import EVIDENCE_VERSION, render_evidence, render_result
from .backends import LocalBackend

__version__ = "0.1.0"

def from_records(records, name="memory", limits=None):
    return Dataset(InMemory(records, name), limits=limits)

def from_qdrant(client, **kwargs):
    return Dataset(Qdrant(client, **kwargs))

def from_weaviate(client, **kwargs):
    return Dataset(Weaviate(client, **kwargs))

def from_pgvector(connection, **kwargs):
    return Dataset(Pgvector(connection, **kwargs))

def from_pinecone(index, **kwargs):
    return Dataset(Pinecone(index, **kwargs))

def analyze(data, operations, backend="auto"):
    from .registry import plan
    return LocalBackend().execute(data, plan(operations, backend, data.limits))
