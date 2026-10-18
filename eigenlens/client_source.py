"""Lazy, bounded reads through SDK clients owned by the calling application."""
from importlib import import_module, util
from contextlib import suppress
import hashlib
import time

from pydantic import ValidationError

from .databases.base import ConnectionOptions
from .datasets import canonical
from .records import validate_selection
from .schemas import Limits, Record, VTError


def client_adapter(client, options):
    # Import only installed optional drivers; credentials stay inside the client.
    for db_type, module_name, class_name, adapter_name in (
        ("qdrant", "qdrant_client", "QdrantClient", "QdrantAdapter"),
        ("weaviate", "weaviate", "WeaviateClient", "WeaviateAdapter"),
    ):
        if util.find_spec(module_name) is None:
            continue
        client_class = getattr(import_module(module_name), class_name)
        if isinstance(client, client_class):
            adapter_class = getattr(import_module(f".databases.{db_type}", __package__), adapter_name)
            return db_type, adapter_class({"db_type": db_type, "options": options}, client=client)
    raise VTError("from_client requires a synchronous QdrantClient or WeaviateClient v4; pass an initialized SDK client")


class ClientSource:
    filter_pushdown = False
    id_pushdown = False

    def __init__(self, client, collection, *, vector_field=None, text_field="text",
                 query_vector=None, search_limit=1000, limits=None):
        if not isinstance(collection, str) or not collection.strip() or len(collection) > 256:
            raise VTError("Collection name must contain 1..256 characters")
        try:
            options = ConnectionOptions(vector_field=vector_field, text_field=text_field)
        except ValidationError:
            raise VTError("Invalid vector_field or text_field") from None
        self.limits = limits or Limits()
        if type(search_limit) is not int or search_limit < 1:
            raise VTError("search_limit must be a positive integer")
        if query_vector is not None:
            try:
                if hasattr(query_vector, "tolist"):
                    query_vector = query_vector.tolist()
                query_vector = Record(id="query", vector=query_vector).vector
                if not query_vector or all(value == 0 for value in query_vector):
                    raise ValueError()
            except Exception:
                raise VTError("Query vector must be a finite, nonzero dense vector") from None
            if search_limit > self.limits.max_rows:
                raise VTError("search_limit exceeds max_rows; lower it or explicitly increase limits")
        self.db_type, self.adapter = client_adapter(client, options.model_dump(exclude_unset=True))
        if query_vector is not None and search_limit > self.adapter.max_search_limit:
            raise VTError("search_limit exceeds this database adapter's search limit")
        self.collection = collection
        self.query_vector = query_vector
        self.search_limit = search_limit
        self.scope = "retrieval_results" if query_vector is not None else "complete_selection"
        self.identity = self.provenance(vector_field)

    def provenance(self, field, model=None, revision=None):
        identity = {"kind": "vector_database", "db_type": self.db_type,
                    "collection": self.collection, "client_ownership": "caller",
                    "mode": "top_k" if self.query_vector is not None else "all",
                    "vector_field": field, "embedding_model": model,
                    "embedding_revision": revision, "embedding_provenance": "stored database vectors"}
        if self.query_vector is not None:
            identity["search_limit"] = self.search_limit
            identity["query_fingerprint"] = hashlib.sha256(canonical(self.query_vector).encode()).hexdigest()
        return identity

    def validate(self, selection):
        validate_selection(selection)

    def batches(self, selection, batch_size, fields):
        self.validate(selection)
        deadline = time.monotonic() + self.limits.max_seconds
        stream = None

        def check_deadline():
            if time.monotonic() > deadline:
                raise VTError("Retrieval exceeded max_seconds; no partial dataset produced")

        try:
            info = self.adapter.describe(self.collection)
            check_deadline()
            expected = info.get("dimension")
            if self.query_vector is not None:
                if expected is not None and len(self.query_vector) != expected:
                    raise VTError("Query vector dimension does not match the selected vector field")
                expected = expected or len(self.query_vector)
                stream = iter(self.adapter.search(self.collection, self.query_vector, self.search_limit))
            else:
                stream = iter(self.adapter.scan(self.collection, batch_size))
            batch, count, model, revision = [], 0, None, None
            for item in stream:
                check_deadline()
                count += 1
                if self.query_vector is not None and count > self.search_limit:
                    raise VTError("Adapter returned more than the requested search_limit")
                if not item.vector:
                    raise VTError("Database did not return a stored dense vector")
                if expected is not None and len(item.vector) != expected:
                    raise VTError("Stored vector dimension conflicts with collection schema or query")
                expected = expected or len(item.vector)
                stored_model = item.metadata.get("embedding_model")
                stored_revision = item.metadata.get("embedding_revision")
                if model and stored_model and model != stored_model:
                    raise VTError("Selection contains conflicting embedding models")
                if revision and stored_revision and revision != stored_revision:
                    raise VTError("Selection contains conflicting embedding revisions")
                model, revision = model or stored_model, revision or stored_revision
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            check_deadline()
            if batch:
                yield batch
            self.identity = self.provenance(info.get("vector_field"), model, revision)
        except VTError:
            raise
        except Exception as exc:
            # SDK exception messages may include application credentials/URLs.
            raise VTError(f"Database request failed ({type(exc).__name__}); check the supplied SDK client, collection and field mappings") from None
        finally:
            if stream is not None and hasattr(stream, "close"):
                with suppress(Exception):
                    stream.close()
            # No adapter.close(): this source borrows the application's client.
