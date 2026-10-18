"""Adapter contract and portable record mapping."""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import Record, VTError


class CollectionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vector_field: str | None = Field(default=None, min_length=1, max_length=128)
    text_field: str = Field(default="text", min_length=1, max_length=128)
    id_field: str = Field(default="id", min_length=1, max_length=128)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=256)
    embedding_revision: str | None = Field(default=None, min_length=1, max_length=128)
    normalize_embeddings: bool = True


class ConnectionOptions(CollectionOptions):
    collections: dict[str, CollectionOptions] = Field(default_factory=dict)
    timeout: int = Field(default=30, ge=1, le=120)
    namespace: str = Field(default="", max_length=256)
    db_name: str = Field(default="default", min_length=1, max_length=128)
    schema_name: str = Field(default="public", min_length=1, max_length=128)
    distance: str = Field(default="cosine", pattern=r"^(cosine|l2|ip)$")
    grpc_host: str | None = Field(default=None, max_length=256)
    grpc_port: int | None = Field(default=None, ge=1, le=65535)
    grpc_secure: bool | None = None
    adapter_options: dict[str, Any] = Field(default_factory=dict)


class Adapter(ABC):
    """Read-only operations. scan/search yield normalized Record instances.

    scan must traverse every page and release resources when closed. search
    returns a bounded ranked selection with stored vectors. describe resolves
    the selected dense vector field and dimension, when known. Factories and
    methods must clean up partially constructed clients on failure.
    """
    max_search_limit = 10000

    def __init__(self, spec):
        self.spec = spec
        self.options = ConnectionOptions.model_validate(spec.get("options", {}))

    def collection_options(self, collection):
        values = self.options.model_dump(include=set(CollectionOptions.model_fields))
        if collection in self.options.collections:
            values.update(self.options.collections[collection].model_dump(exclude_unset=True))
        return CollectionOptions.model_validate(values)

    @abstractmethod
    def list_collections(self) -> list[str]: ...

    @abstractmethod
    def describe(self, collection) -> dict: ...

    @abstractmethod
    def scan(self, collection, batch_size): ...

    @abstractmethod
    def search(self, collection, vector, limit): ...

    @abstractmethod
    def close(self): ...

    def sample(self, collection):
        stream = iter(self.scan(collection, 1))
        try:
            return next(stream, None)
        finally:
            if hasattr(stream, "close"):
                stream.close()


def value(obj: Any, field, default=None):
    return obj.get(field, default) if isinstance(obj, dict) else getattr(obj, field, default)


def choose_vector(fields, configured=None):
    fields = list(fields)
    if configured is not None:
        if configured not in fields:
            raise VTError("Configured dense vector field is unavailable")
        return configured
    if len(fields) != 1:
        raise VTError("Collection has multiple or no dense vector fields; configure vector_field on the connection")
    return fields[0]


def record(identifier, vector, payload, text_field="text"):
    if identifier is None or not str(identifier):
        raise VTError("Database did not return a record ID")
    if vector is None:
        raise VTError("Database did not return the stored dense vector")
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    payload = dict(payload or {})
    text = payload.get(text_field)
    if text is not None and not isinstance(text, str):
        raise VTError("Configured text field must contain a string or null")
    try:
        return Record(id=str(identifier), vector=vector, text=text, metadata=payload)
    except Exception:
        raise VTError("Database returned an invalid record; expected an ID, finite dense vector and JSON metadata") from None
