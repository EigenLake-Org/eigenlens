"""Connection lifecycle and bounded retrieval into immutable analysis snapshots."""
from collections import OrderedDict
import hashlib
import json
import math
import os
import re
import threading
import time

from pydantic import ValidationError

from .connections import ConnectionStore
from .databases import capabilities, open_adapter
from .databases.base import ConnectionOptions
from .datasets import canonical
from .schemas import Record, VTError

_MODELS = OrderedDict()
_MODEL_LOCK = threading.Lock()


def embed_query(text, model, revision, normalize):
    # Serialize model loading/inference; keep at most two CPU models resident.
    from sentence_transformers import SentenceTransformer
    with _MODEL_LOCK:
        key = (model, revision)
        if key not in _MODELS:
            _MODELS[key] = SentenceTransformer(model, revision=revision, device="cpu", trust_remote_code=False)
            while len(_MODELS) > 2:
                _MODELS.popitem(last=False)
        _MODELS.move_to_end(key)
        return _MODELS[key].encode(text, normalize_embeddings=normalize, convert_to_numpy=True).tolist()


class DatabaseManager:
    def __init__(self, service):
        self.service = service
        self.store = ConnectionStore(service.config.root)

    def authorize(self, owner):
        if owner not in self.service.config.callers:
            raise VTError("Caller unauthorized")

    def connect(self, owner, db_type=None, url=None, api_key=None, options=None, name=None):
        self.authorize(owner)
        if db_type is None:
            if owner != "local" or any(item is not None for item in (url, api_key, options)):
                raise VTError("Supply db_type and database connection settings")
            db_type = os.getenv("VECTOR_ML_DATABASE_TYPE")
            url = os.getenv("VECTOR_ML_DATABASE_URL") or None
            api_key = os.getenv("VECTOR_ML_DATABASE_API_KEY") or None
            try:
                options = json.loads(os.getenv("VECTOR_ML_DATABASE_OPTIONS") or "{}")
            except ValueError:
                raise VTError("VECTOR_ML_DATABASE_OPTIONS must be a JSON object") from None
            name = name or "default"
        if not isinstance(db_type, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", db_type):
            raise VTError("Supply a supported db_type, or configure VECTOR_ML_DATABASE_TYPE for the local default")
        if db_type not in capabilities():
            raise VTError("Unsupported database adapter; inspect get_capabilities")
        if url is not None and (not isinstance(url, str) or not url.strip() or len(url) > 4096):
            raise VTError("Database url must contain 1..4096 characters")
        if api_key is not None and (not isinstance(api_key, str) or not api_key or len(api_key) > 8192):
            raise VTError("Database api_key must contain 1..8192 characters")
        try:
            validated = ConnectionOptions.model_validate(options or {})
        except ValidationError:
            raise VTError("Invalid database options; inspect the database connection guide") from None
        if name is not None and (not name.strip() or len(name) > 128):
            raise VTError("Connection name must contain 1..128 characters")
        self.store.cipher()  # Fail before opening a database if encryption isn't configured.
        spec = {"db_type": db_type, "url": url, "api_key": api_key,
                "options": validated.model_dump(exclude_unset=True)}
        with open_adapter(spec) as adapter:
            collections = adapter.list_collections()  # Authenticate and validate before saving.
            self.validate_collections(collections)
        result = self.store.put(owner, spec, name, self.service.config.limits.max_connections)
        return {**result, "status": "connected", "collection_count": len(collections)}

    def validate_collections(self, collections):
        if len(collections) > self.service.config.limits.max_scan_rows:
            raise VTError("Collection listing exceeds max_scan_rows")
        if any(not isinstance(item, str) or not item for item in collections):
            raise VTError("Adapter returned invalid collection names")

    def list_connections(self, owner):
        self.authorize(owner)
        items = self.store.list(owner)
        return {"connections": items, "count": len(items)}

    def list_collections(self, owner, connection_id):
        self.authorize(owner)
        spec = self.store.get(owner, connection_id)
        with open_adapter(spec) as adapter:
            collections = adapter.list_collections()
            self.validate_collections(collections)
        return {"connection_id": connection_id, "collections": sorted(collections), "count": len(collections)}

    def disconnect(self, owner, connection_id):
        self.authorize(owner)
        return self.store.delete(owner, connection_id)

    def retrieve(self, owner, connection_id, collection, query=None, query_vector=None, search_limit=1000):
        self.authorize(owner)
        if not isinstance(collection, str) or not collection.strip() or len(collection) > 256:
            raise VTError("Collection name must contain 1..256 characters")
        if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 8192):
            raise VTError("Query must contain 1..8192 characters")
        if query is not None and query_vector is not None:
            raise VTError("Supply query or query_vector, not both")
        semantic = query is not None or query_vector is not None
        if type(search_limit) is not int or search_limit < 1 or (semantic and search_limit > self.service.config.limits.max_rows):
            raise VTError("search_limit must be positive and, for semantic search, no greater than max_rows")
        spec = self.store.get(owner, connection_id)
        limits = self.service.config.limits
        deadline = time.monotonic() + limits.max_seconds
        records, seen, byte_count, dimension = [], set(), 0, None
        model, revision, field = None, None, None
        with open_adapter(spec) as adapter:
            info = adapter.describe(collection)
            field = info.get("vector_field")
            opts = adapter.collection_options(collection)
            model, revision = opts.embedding_model, opts.embedding_revision
            if semantic and search_limit > adapter.max_search_limit:
                raise VTError("search_limit exceeds this database adapter's search limit")
            if query is not None:
                sample = adapter.sample(collection)
                if sample is None:
                    raise VTError("Collection is empty; no query embedding can be resolved")
                source_model = sample.metadata.get("embedding_model")
                source_revision = sample.metadata.get("embedding_revision")
                if model and source_model and model != source_model:
                    raise VTError("Configured embedding model conflicts with stored collection metadata")
                if revision and source_revision and revision != source_revision:
                    raise VTError("Configured embedding revision conflicts with stored collection metadata")
                model = model or source_model
                revision = revision or source_revision
                if not isinstance(model, str) or not model:
                    raise VTError("Embedding model is unknown; configure embedding_model on the connection or supply query_vector")
                # Query text can use a SentenceTransformer model. Other embedding
                # providers work through externally generated query_vector.
                try:
                    query_vector = embed_query(query, model, revision, opts.normalize_embeddings)
                except Exception:
                    raise VTError("Query embedding failed; configure a usable SentenceTransformer model and model access, or supply query_vector from the original embedding provider") from None
            if semantic:
                try:
                    query_vector = Record(id="query", vector=query_vector).vector
                    if query_vector is None or all(value == 0 for value in query_vector):
                        raise ValueError()
                except Exception:
                    raise VTError("Query vector must be a finite, nonzero dense vector") from None
                expected = info.get("dimension")
                if expected is not None and len(query_vector) != expected:
                    raise VTError("Query vector dimension does not match the selected vector field")
                stream = iter(adapter.search(collection, query_vector, search_limit))
            else:
                stream = iter(adapter.scan(collection, limits.batch_size))
            try:
                for item in stream:
                    if time.monotonic() > deadline:
                        raise VTError("Retrieval exceeded max_seconds; no partial dataset registered")
                    record = item if isinstance(item, Record) else Record.model_validate(item)
                    if record.id in seen:
                        raise VTError("Database returned duplicate IDs across pages; no partial dataset registered")
                    seen.add(record.id)
                    if len(seen) > limits.max_scan_rows or len(records) >= limits.max_rows:
                        raise VTError("Retrieval exceeds row limits; no partial dataset registered")
                    if semantic and len(records) >= search_limit:
                        raise VTError("Adapter returned more than the requested search_limit")
                    if not record.vector or any(not math.isfinite(value) for value in record.vector):
                        raise VTError("Database did not return a finite stored dense vector")
                    dimension = dimension or len(record.vector)
                    if len(record.vector) != dimension or (semantic and len(record.vector) != len(query_vector)):
                        raise VTError("Database records have incompatible vector dimensions")
                    if info.get("dimension") is not None and len(record.vector) != info["dimension"]:
                        raise VTError("Stored vector dimension conflicts with collection schema")
                    stored_model = record.metadata.get("embedding_model")
                    if stored_model:
                        if model and stored_model != model:
                            raise VTError("Selection contains conflicting embedding models")
                        model = model or stored_model
                    stored_revision = record.metadata.get("embedding_revision")
                    if stored_revision:
                        if revision and stored_revision != revision:
                            raise VTError("Selection contains conflicting embedding revisions")
                        revision = revision or stored_revision
                    encoded = record.model_dump(mode="json")
                    byte_count += len(canonical(encoded).encode()) + 1
                    if byte_count > limits.max_bytes:
                        raise VTError("Retrieval exceeds max_bytes; no partial dataset registered")
                    records.append(encoded)
            finally:
                if hasattr(stream, "close"):
                    stream.close()
        if time.monotonic() > deadline:
            raise VTError("Retrieval exceeded max_seconds; no partial dataset registered")
        source = {"kind": "vector_database", "db_type": spec["db_type"], "connection_id": connection_id,
                  "collection": collection, "mode": "top_k" if semantic else "all", "vector_field": field,
                  "embedding_model": model, "embedding_revision": revision,
                  "embedding_provenance": "stored database vectors"}
        if semantic:
            source["search_limit"] = search_limit
            source["query_fingerprint"] = hashlib.sha256(canonical(query_vector).encode()).hexdigest()
        registered = self.service._register_snapshot(owner, records,
            "retrieval_results" if semantic else "complete_selection", source=source)
        return {**registered, "connection_id": connection_id, "collection": collection,
                "selection": source, "bytes": byte_count}
