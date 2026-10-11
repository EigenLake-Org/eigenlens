"""Read-only PostgreSQL retrieval with quoted identifiers and bound values."""
import json
import uuid
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from .base import Adapter, choose_vector, record
from ..schemas import VTError


class PgvectorAdapter(Adapter):
    max_search_limit = 100000

    def __init__(self, spec):
        super().__init__(spec)
        if not spec.get("url"):
            raise VTError("pgvector requires a PostgreSQL connection URL or DSN")
        kwargs = {"connect_timeout": self.options.timeout, "row_factory": dict_row,
                  "options": f"-c statement_timeout={self.options.timeout * 1000} -c default_transaction_read_only=on"}
        if spec.get("api_key"):
            kwargs["password"] = spec["api_key"]
        self.client = psycopg.connect(spec["url"], **kwargs)
        self.client.read_only = True

    def columns(self):
        return self.client.execute("""SELECT c.relname AS name,a.attname AS field,a.atttypmod AS dimension
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            JOIN pg_attribute a ON a.attrelid=c.oid JOIN pg_type t ON t.oid=a.atttypid
            WHERE n.nspname=%s AND c.relkind IN ('r','p','v','m') AND t.typname='vector'
            AND a.attnum>0 AND NOT a.attisdropped ORDER BY c.relname,a.attnum""",
            (self.options.schema_name,)).fetchall()

    def list_collections(self):
        return sorted({row["name"] for row in self.columns()})

    def describe(self, collection):
        columns = {row["field"]: row for row in self.columns() if row["name"] == collection}
        field = choose_vector(columns, self.collection_options(collection).vector_field)
        dim = columns[field]["dimension"]
        return {"vector_field": field, "dimension": dim if dim > 0 else None}

    def statement(self, collection, search=False):
        info, opts = self.describe(collection), self.collection_options(collection)
        statement = sql.SQL("SELECT {id}::text AS _id, {vector}::text AS _vector, "
                            "to_jsonb(t) - {field} AS _payload FROM {table} AS t").format(
            id=sql.Identifier("t", opts.id_field), vector=sql.Identifier("t", info["vector_field"]),
            field=sql.Literal(info["vector_field"]), table=sql.Identifier(self.options.schema_name, collection))
        if search:
            op = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}[self.options.distance]
            statement += sql.SQL(" ORDER BY {vector} {operator} %s::vector LIMIT %s").format(
                vector=sql.Identifier("t", info["vector_field"]), operator=sql.SQL(op))
        else:
            statement += sql.SQL(" ORDER BY {id}").format(id=sql.Identifier("t", opts.id_field))
        return statement

    def mapped(self, collection, row):
        return record(row["_id"], json.loads(row["_vector"]) if row["_vector"] is not None else None,
                      row["_payload"], self.collection_options(collection).text_field)

    def scan(self, collection, batch_size):
        statement = self.statement(collection)
        with self.client.cursor(name="vt_" + uuid.uuid4().hex) as cursor:
            cursor.execute(statement)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    return
                for row in rows:
                    yield self.mapped(collection, row)

    def search(self, collection, vector, limit):
        with self.client.cursor() as cursor:
            cursor.execute(self.statement(collection, search=True), (json.dumps(vector, allow_nan=False), limit))
            for row in cursor:
                yield self.mapped(collection, row)

    def close(self):
        self.client.close()
