"""Extensible, read-only database adapters; optional SDKs are imported on use."""
from contextlib import contextmanager
from importlib import import_module, metadata, util
import re

from ..schemas import VTError

BUILTINS = {
    "qdrant": (".qdrant", "QdrantAdapter", ["qdrant_client"]),
    "pinecone": (".pinecone", "PineconeAdapter", ["pinecone"]),
    "milvus": (".milvus", "MilvusAdapter", ["pymilvus"]),
    "weaviate": (".weaviate", "WeaviateAdapter", ["weaviate"]),
    "pgvector": (".pgvector", "PgvectorAdapter", ["psycopg"]),
}
CUSTOM = {}


def register_adapter(name, factory):
    """Register a factory(spec) -> adapter in application code before serving."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name in BUILTINS:
        raise VTError("Custom adapter name must be unique and use lowercase letters, digits or underscores")
    if not callable(factory):
        raise VTError("Adapter factory must be callable")
    CUSTOM[name] = factory


def plugins():
    return {entry.name: entry for entry in metadata.entry_points(group="vector_tools.database_adapters")}


def capabilities():
    result = {name: {"installed": all(util.find_spec(module) is not None for module in item[2]),
                     "install_extra": name} for name, item in BUILTINS.items()}
    result.update({name: {"installed": True, "plugin": True} for name in {*CUSTOM, *plugins()}})
    return result


def create_adapter(spec):
    name = spec["db_type"]
    if name in CUSTOM:
        factory = CUSTOM[name]
    elif name in BUILTINS:
        module, klass, deps = BUILTINS[name]
        if not all(util.find_spec(dep) is not None for dep in deps):
            raise VTError(f"Database driver unavailable; install the '{name}' extra using this project's pip installation URL")
        factory = getattr(import_module(module, __name__), klass)
    elif name in plugins():
        factory = plugins()[name].load()
    else:
        raise VTError("Unsupported database adapter; inspect get_capabilities for supported db_types")
    return factory(spec)


@contextmanager
def open_adapter(spec):
    adapter = None
    try:
        adapter = create_adapter(spec)
        yield adapter
    except VTError:
        raise
    except Exception as exc:
        # SDK exceptions can contain tokens, DSNs, queries or payloads.
        raise VTError(f"Database request failed ({type(exc).__name__}); check connection settings, SDK compatibility and database access") from None
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass
