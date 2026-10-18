from pymilvus import MilvusClient
from .base import Adapter, choose_vector, record
from ..schemas import VTError


class MilvusAdapter(Adapter):
    max_search_limit = 16384

    def __init__(self, spec):
        super().__init__(spec)
        if not spec.get("url"):
            raise VTError("Milvus requires url")
        self.client = MilvusClient(uri=spec["url"], token=spec.get("api_key") or "",
                                   db_name=self.options.db_name, timeout=self.options.timeout)

    def list_collections(self):
        return sorted(self.client.list_collections())

    def describe(self, collection):
        info = self.client.describe_collection(collection_name=collection)
        fields = info["fields"]
        # Only FLOAT_VECTOR (101); sparse and multi-vector fields need other pipelines.
        vectors = {field["name"]: field for field in fields if int(field["type"]) == 101}
        selected = choose_vector(vectors, self.collection_options(collection).vector_field)
        primary = next((field["name"] for field in fields if field.get("is_primary")), None)
        if primary is None:
            raise VTError("Milvus collection has no primary key")
        return {"vector_field": selected, "dimension": int(vectors[selected]["params"]["dim"]),
                "id_field": primary, "fields": [field["name"] for field in fields]}

    def mapped(self, collection, row, info):
        payload = {key: val for key, val in row.items() if key not in {info["vector_field"], info["id_field"]}}
        return record(row[info["id_field"]], row.get(info["vector_field"]), payload,
                      self.collection_options(collection).text_field)

    def scan(self, collection, batch_size):
        info = self.describe(collection)
        iterator = self.client.query_iterator(collection_name=collection, filter="", batch_size=batch_size,
                                              output_fields=info["fields"], timeout=self.options.timeout)
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    return
                for row in rows:
                    yield self.mapped(collection, row, info)
        finally:
            iterator.close()

    def search(self, collection, vector, limit):
        info = self.describe(collection)
        rows = self.client.search(collection_name=collection, data=[vector], anns_field=info["vector_field"],
            limit=limit, output_fields=info["fields"], timeout=self.options.timeout)[0]
        for hit in rows:
            row = dict(hit["entity"])
            row.setdefault(info["id_field"], hit["id"])
            yield self.mapped(collection, row, info)

    def close(self):
        self.client.close()
