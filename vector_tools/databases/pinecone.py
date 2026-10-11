import httpx
from pinecone.db_data import Index
from .base import Adapter, record, value
from ..schemas import VTError


class PineconeAdapter(Adapter):
    def __init__(self, spec):
        super().__init__(spec)
        if not spec.get("api_key"):
            raise VTError("Pinecone requires api_key")
        # SDK control-plane helpers do not expose request timeouts. Use the
        # documented HTTP API there and the public SDK for vector operations.
        self.client = httpx.Client(base_url="https://api.pinecone.io", timeout=self.options.timeout,
            headers={"Api-Key": spec["api_key"], "X-Pinecone-API-Version": "2025-04"})
        self.indexes = []

    def list_collections(self):
        response = self.client.get("/indexes")
        response.raise_for_status()
        items = response.json()["indexes"]
        if self.spec.get("url"):
            host = self.spec["url"].removeprefix("https://").rstrip("/")
            items = [item for item in items if value(item, "host") == host]
        return sorted(value(item, "name") for item in items)

    def describe(self, collection):
        from urllib.parse import quote
        response = self.client.get("/indexes/" + quote(collection, safe=""))
        response.raise_for_status()
        info = response.json()
        host = value(info, "host")
        if self.spec.get("url") and self.spec["url"].removeprefix("https://").rstrip("/") != host:
            raise VTError("Collection does not belong to the configured Pinecone index host")
        return {"dimension": value(info, "dimension"), "vector_field": None, "host": host}

    def index(self, collection):
        index = Index(host=self.describe(collection)["host"], api_key=self.spec["api_key"], pool_threads=1)
        self.indexes.append(index)
        return index

    def scan(self, collection, batch_size):
        index, opts = self.index(collection), self.collection_options(collection)
        token, seen = None, set()
        while True:
            kwargs = {"namespace": self.options.namespace, "limit": min(batch_size, 100)}
            if token is not None:
                kwargs["pagination_token"] = token
            response = index.list_paginated(**kwargs, _request_timeout=self.options.timeout)
            ids = [value(item, "id") for item in value(response, "vectors", [])]
            if ids:
                fetched = value(index.fetch(ids=ids, namespace=self.options.namespace, _request_timeout=self.options.timeout), "vectors", {})
                for identifier in ids:
                    if identifier not in fetched:
                        raise VTError("A listed vector disappeared during retrieval; retry on a stable collection")
                    point = fetched[identifier]
                    yield record(identifier, value(point, "values"), value(point, "metadata", {}), opts.text_field)
            token = value(value(response, "pagination", {}), "next")
            if not token:
                return
            if token in seen:
                raise VTError("Database pagination repeated a cursor; no partial dataset registered")
            seen.add(token)

    def search(self, collection, vector, limit):
        response = self.index(collection).query(vector=vector, top_k=limit, namespace=self.options.namespace,
            include_values=True, include_metadata=True, _request_timeout=self.options.timeout)
        opts = self.collection_options(collection)
        for point in value(response, "matches", []):
            yield record(value(point, "id"), value(point, "values"), value(point, "metadata", {}), opts.text_field)

    def close(self):
        for index in self.indexes:
            index.close()
        self.client.close()
