from qdrant_client import QdrantClient
from .base import Adapter, choose_vector, record
from ..schemas import VTError


class QdrantAdapter(Adapter):
    def __init__(self, spec, *, client=None):
        super().__init__(spec)
        self.owns_client = client is None
        if client is not None:
            self.client = client
        elif not spec.get("url"):
            raise VTError("Qdrant requires url")
        else:
            self.client = QdrantClient(url=spec["url"], api_key=spec.get("api_key"), timeout=self.options.timeout)

    def list_collections(self):
        return sorted(item.name for item in self.client.get_collections().collections)

    def describe(self, collection):
        vectors = self.client.get_collection(collection).config.params.vectors
        fields = vectors if isinstance(vectors, dict) else {"": vectors}
        field = choose_vector(fields, self.collection_options(collection).vector_field)
        if fields[field] is None:
            raise VTError("Collection has no dense vectors")
        return {"vector_field": field or None, "dimension": fields[field].size}

    def scan(self, collection, batch_size):
        info = self.describe(collection)
        opts = self.collection_options(collection)
        offset, seen = None, set()
        while True:
            points, next_offset = self.client.scroll(collection_name=collection, offset=offset,
                limit=batch_size, with_vectors=[info["vector_field"]] if info["vector_field"] else True,
                with_payload=True)
            for point in points:
                vector = point.vector.get(info["vector_field"]) if isinstance(point.vector, dict) else point.vector
                yield record(point.id, vector, point.payload, opts.text_field)
            if next_offset is None:
                return
            marker = str(next_offset)
            if marker in seen:
                raise VTError("Database pagination repeated a cursor; no partial dataset registered")
            seen.add(marker)
            offset = next_offset

    def search(self, collection, vector, limit):
        info = self.describe(collection)
        opts = self.collection_options(collection)
        response = self.client.query_points(collection_name=collection, query=vector,
            using=info["vector_field"], limit=limit, with_vectors=True, with_payload=True)
        for point in response.points:
            stored = point.vector.get(info["vector_field"]) if isinstance(point.vector, dict) else point.vector
            yield record(point.id, stored, point.payload, opts.text_field)

    def close(self):
        if self.owns_client:
            self.client.close()
