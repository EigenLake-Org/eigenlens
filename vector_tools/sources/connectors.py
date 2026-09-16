"""Optional clients are imported only when their connector is used."""
from __future__ import annotations
import json
from .base import Adapter
from ..schemas import Record, VTError

def chunks(values, n):
    for i in range(0, len(values), n):
        yield values[i:i+n]

def dense(value, field):
    if isinstance(value, dict):
        if field not in value:
            raise VTError(f"Vector field {field!r} unavailable; available: {list(value)}")
        value = value[field]
    if value is None:
        raise VTError("Stored dense vector missing. Enable vector retrieval and select a dense field.")
    if isinstance(value, str):
        value = json.loads(value)
    if type(value).__name__ in {"SparseVector", "HalfVector"}:
        raise VTError("pgvector MVP supports vector dense float32 columns, not sparsevec or halfvec")
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    try:
        vector = list(value)
        if any(isinstance(x, (list, dict)) for x in vector):
            raise TypeError()
        return vector
    except TypeError as exc:
        raise VTError("Only single dense vectors are supported; sparse/multivectors are unavailable") from exc

def record(id, vector, payload, text_field, parent_field, fields):
    payload = dict(payload or {})
    text = payload.pop(text_field, None) if text_field else None
    parent = payload.get(parent_field) if parent_field else None
    if text is not None and not isinstance(text, str):
        raise VTError(f"Record {id}: configured text field is not a string")
    return Record(id=str(id), vector=vector if "vector" in fields else None,
                  text=text, metadata=payload, parent_id=str(parent) if parent is not None else None)

class Qdrant(Adapter):
    def __init__(self, client, collection, vector_field=None, text_field="text", parent_field=None, source_name=None):
        self.client, self.collection = client, collection
        self.vector_field, self.text_field, self.parent_field = vector_field, text_field, parent_field
        self.identity = {"kind": "qdrant", "source_name": source_name, "endpoint": self.endpoint(client), "collection": collection,
                         "vector_field": vector_field, "text_field": text_field, "parent_field": parent_field}
    @staticmethod
    def endpoint(client):
        from urllib.parse import urlsplit
        options = getattr(client, "init_options", {})
        url = options.get("url")
        if url:
            parsed = urlsplit(url)
            return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 6333}{parsed.path}"
        return options.get("host") or options.get("location") or options.get("path") or "caller-supplied-client"
    def inspect(self):
        cfg = self.client.get_collection(self.collection).config.params.vectors
        cfg = cfg.get(self.vector_field) if isinstance(cfg, dict) else cfg
        if cfg is None:
            raise VTError("Configured Qdrant dense vector field is unavailable")
        if getattr(cfg, "multivector_config", None):
            raise VTError("Qdrant multivectors are unsupported")
        self.identity.update(dimension=cfg.size, distance=str(cfg.distance))
    def batches(self, selection, batch_size, fields):
        self.validate(selection)
        self.inspect()
        vectors = ([self.vector_field] if self.vector_field else True) if "vector" in fields else False
        def convert(rows):
            return [record(r.id, dense(r.vector, self.vector_field) if "vector" in fields else None,
                           r.payload, self.text_field, self.parent_field, fields) for r in rows]
        if selection.ids is not None:
            for ids in chunks(selection.ids, batch_size):
                ids = [int(i) if i.isdecimal() else i for i in ids]
                yield convert(self.client.retrieve(self.collection, ids=ids, with_vectors=vectors, with_payload=True))
        else:
            offset = None
            while True:
                rows, offset = self.client.scroll(self.collection, limit=batch_size, offset=offset,
                                                  with_vectors=vectors, with_payload=True)
                yield convert(rows)
                if offset is None:
                    break

class Weaviate(Adapter):
    def __init__(self, client, collection, tenant=None, vector_field="default", text_field="text", parent_field=None, source_name=None):
        self.client, self.name, self.tenant = client, collection, tenant
        self.vector_field, self.text_field, self.parent_field = vector_field, text_field, parent_field
        from urllib.parse import urlsplit
        url = urlsplit(str(getattr(getattr(client, "_connection", None), "url", "")))
        endpoint = f"{url.scheme}://{url.hostname}:{url.port or 8080}" if url.hostname else "caller-supplied-client"
        self.identity = {"kind": "weaviate", "source_name": source_name, "endpoint": endpoint, "collection": collection,
                         "tenant": tenant, "vector_field": vector_field, "text_field": text_field, "parent_field": parent_field}
    def collection(self):
        c = self.client.collections.get(self.name)
        config = c.config.get()
        if getattr(getattr(config, "multi_tenancy_config", None), "enabled", False) and not self.tenant:
            raise VTError("Explicit tenant required for this multi-tenant Weaviate collection")
        vector_cfg = getattr(config, "vector_config", None)
        if vector_cfg and self.vector_field not in vector_cfg:
            raise VTError("Configured Weaviate named vector is unavailable")
        index_config = getattr(vector_cfg[self.vector_field], "vector_index_config", None) if vector_cfg else getattr(config, "vector_index_config", None)
        self.identity["distance_configuration"] = str(index_config)[:1000]
        return c.with_tenant(self.tenant) if self.tenant else c
    def batches(self, selection, batch_size, fields):
        self.validate(selection)
        c = self.collection()
        include = "vector" in fields
        def convert(r):
            return record(r.uuid, dense(r.vector, self.vector_field) if include else None,
                          r.properties, self.text_field, self.parent_field, fields)
        if selection.ids is not None:
            for ids in chunks(selection.ids, batch_size):
                rows = [c.query.fetch_object_by_id(i, include_vector=include) for i in ids]
                yield [convert(r) for r in rows if r is not None]
        else:
            # iterator uses an ordered UUID cursor, retrieving stored vectors explicitly.
            batch = []
            for r in c.iterator(include_vector=include, cache_size=batch_size):
                batch.append(convert(r))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

class Pinecone(Adapter):
    def __init__(self, index, namespace, text_field="text", parent_field=None, serverless=None, index_description=None, source_name=None):
        if namespace is None:
            raise VTError("An explicit Pinecone namespace is required (empty string explicitly selects the default)")
        self.index, self.namespace = index, namespace
        self.text_field, self.parent_field = text_field, parent_field
        desc = index_description.to_dict() if hasattr(index_description, "to_dict") else index_description or {}
        self.serverless = bool(desc.get("spec", {}).get("serverless"))
        if desc.get("vector_type", "dense") != "dense":
            raise VTError("Pinecone sparse indexes are unsupported")
        # A bare boolean cannot verify an index deployment; description comes from control-plane describe_index.
        self.identity = {"kind": "pinecone", "source_name": source_name,
                         "host": getattr(getattr(index, "config", None), "host", None),
                         "namespace": namespace, "text_field": text_field, "parent_field": parent_field,
                         "serverless_verified": self.serverless, "distance": desc.get("metric"),
                         "dimension": desc.get("dimension")}
    def capabilities(self):
        return {**super().capabilities(), "iteration": "native" if self.serverless else "unavailable",
                "seeded_sampling": "adapter" if self.serverless else "unavailable",
                "projection": "unavailable", "constraints": "ID listing only on control-plane-verified serverless dense indexes; fetch batches <=1000"}
    def batches(self, selection, batch_size, fields):
        self.validate(selection)
        pages = chunks(selection.ids, min(batch_size, 1000)) if selection.ids is not None else self.index.list(namespace=self.namespace, limit=min(batch_size, 1000))
        for ids in pages:
            response = self.index.fetch(ids=list(ids), namespace=self.namespace)
            vectors = response.vectors
            yield [record(id, dense(v.values, None) if "vector" in fields else None, v.metadata,
                          self.text_field, self.parent_field, fields) for id, v in vectors.items()]

class Pgvector(Adapter):
    filter_pushdown = True
    def __init__(self, connection, schema, table, id_column="id", vector_column="embedding", text_column="text", metadata_columns=None, parent_column=None, metric="cosine", source_name=None):
        self.connection = connection
        self.schema, self.table = schema, table
        self.id_column, self.vector_column, self.text_column = id_column, vector_column, text_column
        self.metadata_columns, self.parent_column = metadata_columns or [], parent_column
        if metric not in {"cosine", "l2", "inner_product"}:
            raise VTError("metric must be cosine, l2, or inner_product")
        self.identity = {"kind": "pgvector", "source_name": source_name, "database": getattr(getattr(connection, "info", None), "dbname", None), "host": getattr(getattr(connection, "info", None), "host", None), "port": getattr(getattr(connection, "info", None), "port", None), "schema": schema, "table": table,
                         "id_column": id_column, "vector_column": vector_column, "text_column": text_column,
                         "metadata_columns": self.metadata_columns, "parent_column": parent_column,
                         "query_metric": metric, "operator": {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}[metric],
                         "index_usage": "Ordered row retrieval; no vector index used"}
    def capabilities(self):
        return {**super().capabilities(), "filtered_reads": "native", "consistency": "Read-only REPEATABLE READ transaction for each complete iteration", "filter_cost": "SQL pushdown with parameterized values"}
    def query(self, selection, fields):
        from psycopg import sql
        self.validate(selection)
        cols = [self.id_column]
        if "vector" in fields:
            cols.append(self.vector_column)
        if self.text_column:
            cols.append(self.text_column)
        cols += self.metadata_columns + ([self.parent_column] if self.parent_column else [])
        cols = list(dict.fromkeys(cols))
        mappings = {"id": self.id_column, "text": self.text_column, "parent_id": self.parent_column,
                    **{c: c for c in self.metadata_columns}}
        conditions, args = [], []
        if selection.ids is not None:
            conditions.append(sql.SQL("{}::text = ANY(%s)").format(sql.Identifier(self.id_column)))
            args.append(selection.ids)
        operators = {"eq": "=", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
        for p in selection.filters:
            col = mappings.get(p.field)
            if not col:
                raise VTError(f"Filter field {p.field!r} is not configured for PostgreSQL")
            ident = sql.Identifier(col)
            if p.field == "id":
                ident = sql.SQL("{}::text").format(ident)
            if p.op == "exists":
                conditions.append(sql.SQL("{} IS {}NULL").format(ident, sql.SQL("NOT " if p.value else "")))
            elif p.op == "in":
                conditions.append(sql.SQL("{} = ANY(%s)").format(ident)); args.append(p.value)
            else:
                conditions.append(sql.SQL("{} {} %s").format(ident, sql.SQL(operators[p.op]))); args.append(p.value)
        query = sql.SQL("SELECT {} FROM {}.{}{}").format(
            sql.SQL(", ").join(map(sql.Identifier, cols)), sql.Identifier(self.schema), sql.Identifier(self.table),
            sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL(""))
        query += sql.SQL(" ORDER BY {} ASC").format(sql.Identifier(self.id_column))
        return query, args, cols
    def batches(self, selection, batch_size, fields):
        import uuid
        from psycopg.pq import TransactionStatus
        if self.connection.info.transaction_status != TransactionStatus.IDLE:
            raise VTError("pgvector requires an idle connection to establish its read-only repeatable-read transaction")
        query, args, cols = self.query(selection, fields)
        with self.connection.transaction():
            self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            with self.connection.cursor(name="vt_" + uuid.uuid4().hex) as cur:
                cur.execute(query, args)
                while rows := cur.fetchmany(batch_size):
                    output = []
                    for row in rows:
                        r = dict(zip(cols, row))
                        value = r.pop(self.vector_column, None)
                        id = r.pop(self.id_column)
                        output.append(record(id, dense(value, None) if "vector" in fields else None, r,
                                             self.text_column, self.parent_column, fields))
                    yield output
