from urllib.parse import urlsplit
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from .base import Adapter, choose_vector, record
from ..schemas import VTError


class WeaviateAdapter(Adapter):
    def __init__(self, spec, *, client=None):
        super().__init__(spec)
        self.owns_client = client is None
        if client is not None:
            self.client = client
            return
        parsed = urlsplit(spec.get("url") or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path.strip("/"):
            raise VTError("Weaviate requires an HTTP(S) origin URL")
        secure = parsed.scheme == "https"
        timeout = self.options.timeout
        self.client = weaviate.connect_to_custom(http_host=parsed.hostname, http_port=parsed.port or (443 if secure else 8080),
            http_secure=secure, grpc_host=self.options.grpc_host or parsed.hostname,
            grpc_port=self.options.grpc_port or (443 if secure else 50051),
            grpc_secure=secure if self.options.grpc_secure is None else self.options.grpc_secure,
            auth_credentials=Auth.api_key(spec["api_key"]) if spec.get("api_key") else None,
            additional_config=AdditionalConfig(timeout=Timeout(init=timeout, query=timeout, insert=timeout)))

    def list_collections(self):
        return sorted(self.client.collections.list_all(simple=True))

    def describe(self, collection):
        config = self.client.collections.use(collection).config.get()
        fields = list(config.vector_config) if config.vector_config else ["default"]
        selected = choose_vector(fields, self.collection_options(collection).vector_field)
        return {"vector_field": selected, "dimension": None, "named_vectors": bool(config.vector_config)}

    def mapped(self, point, collection, field):
        return record(point.uuid, point.vector.get(field), point.properties,
                      self.collection_options(collection).text_field)

    def scan(self, collection, batch_size):
        field = self.describe(collection)["vector_field"]
        stream = iter(self.client.collections.use(collection).iterator(include_vector=True, cache_size=batch_size))
        try:
            for point in stream:
                yield self.mapped(point, collection, field)
        finally:
            if hasattr(stream, "close"):
                stream.close()

    def search(self, collection, vector, limit):
        info = self.describe(collection)
        field = info["vector_field"]
        response = self.client.collections.use(collection).query.near_vector(near_vector=vector, limit=limit,
            target_vector=field if info["named_vectors"] else None, include_vector=True)
        for point in response.objects:
            yield self.mapped(point, collection, field)

    def close(self):
        if self.owns_client:
            self.client.close()
