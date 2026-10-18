"""Clustering SDK for stored vectors and caller-supplied records; MCP is optional."""
from .datasets import Dataset, Materialized
from .schemas import Record, Predicate, Limits, Operation, Result, VTError
from .records import InMemory
from .operators import cluster, topics
from .backends import LocalBackend

__version__ = "0.1.0"

def from_records(records, name="memory", limits=None):
    return Dataset(InMemory(records, name), limits=limits)

def from_client(client, collection, *, vector_field=None, text_field="text",
                query_vector=None, search_limit=1000, limits=None):
    """Read stored vectors using a caller-owned QdrantClient or WeaviateClient.

    Reads start on materialize(). Without query_vector, traverse all pages;
    with it, select up to search_limit nearest records. The caller owns the
    client's connection, credentials, timeouts and close().
    """
    from .client_source import ClientSource
    limits = limits or Limits()
    return Dataset(ClientSource(client, collection, vector_field=vector_field,
        text_field=text_field, query_vector=query_vector, search_limit=search_limit,
        limits=limits), limits=limits)

def analyze(data, operations, backend="auto"):
    from .registry import plan
    return LocalBackend().execute(data, plan(operations, backend, data.limits))
