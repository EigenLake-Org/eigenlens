"""Seed explicitly named demo collections; reruns upsert these fixture IDs only.
Run: PGVECTOR_DSN=... python -m examples.seed_sources qdrant weaviate pgvector
Pinecone requires an existing 4-dimensional dense index named in config/local.json.
"""
import argparse
import os
import uuid
from vector_tools.demo import tickets
from vector_tools.config import Config, close_source

def fixture():
    return [r.model_copy(update={"id":str(uuid.uuid5(uuid.NAMESPACE_URL, "vector-ml-demo/" + r.id))}) for r in tickets()]

def payload(r):
    return {**r.metadata, "text":r.text, **({"parent_id":r.parent_id} if r.parent_id else {})}

def seed(kind, config):
    rows = fixture()
    if kind == "qdrant":
        from qdrant_client import QdrantClient, models
        cfg = config.raw["sources"][kind]
        client = QdrantClient(url=cfg["url"])
        collection = cfg["mapping"]["collection"]
        if not client.collection_exists(collection):
            client.create_collection(collection, vectors_config={"body":models.VectorParams(size=4, distance=models.Distance.COSINE)})
        client.upsert(collection, points=[models.PointStruct(id=r.id, vector={"body":r.vector}, payload=payload(r)) for r in rows], wait=True)
        client.close()
    elif kind == "weaviate":
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        from weaviate.classes.tenants import Tenant
        from weaviate.classes.data import DataObject
        cfg = config.raw["sources"][kind]
        client = weaviate.connect_to_local(host=cfg["host"])
        name, tenant = cfg["mapping"]["collection"], cfg["mapping"]["tenant"]
        try:
            if not client.collections.exists(name):
                client.collections.create(name, vector_config=Configure.Vectors.self_provided(name="body"),
                    multi_tenancy_config=Configure.multi_tenancy(enabled=True), properties=[
                        Property(name=k, data_type=DataType.TEXT) for k in ["text", "theme", "status", "parent_id"]] + [Property(name="synthetic", data_type=DataType.BOOL)])
            c = client.collections.get(name)
            if tenant not in c.tenants.get():
                c.tenants.create([Tenant(name=tenant)])
            c = c.with_tenant(tenant)
            for r in rows:
                if c.data.exists(r.id):
                    c.data.replace(uuid=r.id, properties=payload(r), vector={"body":r.vector})
                else:
                    c.data.insert(uuid=r.id, properties=payload(r), vector={"body":r.vector})
        finally:
            client.close()
    elif kind == "pgvector":
        import psycopg
        from psycopg import sql
        from pgvector.psycopg import register_vector
        import numpy as np
        with psycopg.connect(os.environ["PGVECTOR_DSN"], autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            mapping = config.raw["sources"][kind]["mapping"]
            table = sql.SQL("{}.{}").format(sql.Identifier(mapping["schema"]), sql.Identifier(mapping["table"]))
            conn.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} (id text PRIMARY KEY, embedding vector(4), text text, theme text, status text, synthetic boolean, parent_id text)").format(table))
            with conn.cursor() as cur:
                cur.executemany(sql.SQL("INSERT INTO {} VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE SET embedding=EXCLUDED.embedding,text=EXCLUDED.text,theme=EXCLUDED.theme,status=EXCLUDED.status,synthetic=EXCLUDED.synthetic,parent_id=EXCLUDED.parent_id").format(table),
                    [(r.id,np.array(r.vector),r.text,r.metadata["theme"],r.metadata.get("status"),True,r.parent_id) for r in rows])
    elif kind == "pinecone":
        source = config.source(kind)
        try:
            if source.identity.get("dimension") != 4:
                raise ValueError("Demo requires a configured 4-dimensional dense Pinecone test index")
            source.index.upsert(vectors=[{"id":r.id,"values":r.vector,"metadata":payload(r)} for r in rows], namespace=source.namespace)
        finally:
            close_source(source)
    else:
        raise ValueError(kind)
    print(f"Seeded {len(rows)} synthetic records into configured {kind} demo scope")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", choices=["qdrant","weaviate","pgvector","pinecone"])
    parser.add_argument("--config", default="config/local.json")
    args = parser.parse_args()
    cfg = Config(args.config)
    for source in args.sources:
        seed(source, cfg)
