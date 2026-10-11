"""Service-owned configuration; tools never accept credentials or arbitrary endpoints."""
import json
import os
import re
from pathlib import Path
from .schemas import Limits, VTError
from .sources import InMemory, Qdrant, Weaviate, Pgvector, Pinecone
from .demo import tickets

class Config:
    def __init__(self, path=None):
        self.path = path or os.getenv("VECTOR_ML_CONFIG")
        self.raw = json.loads(Path(self.path).read_text()) if self.path else {"sources": {"demo": {"kind": "demo"}}, "callers": {"local": {"sources": ["demo"]}}}
        limits=dict(self.raw.get("limits", {}))
        workers=os.getenv("VECTOR_ML_WORKERS")
        if workers is not None:
            if not re.fullmatch(r"([1-9]|1[0-6])",workers):
                raise VTError("VECTOR_ML_WORKERS must be an integer from 1 to 16")
            limits["max_jobs"]=int(workers)
        total=os.getenv("VECTOR_ML_TOTAL_WORKING_BYTES")
        if total is not None:
            if not total.isascii() or not total.isdigit():
                raise VTError("VECTOR_ML_TOTAL_WORKING_BYTES must be a positive byte count")
            count=limits.get("max_jobs",Limits().max_jobs)
            limits["max_working_bytes"]=int(total)//count
        self.limits = Limits.model_validate(limits)
        self.root = Path(os.getenv("VECTOR_ML_DATA", self.raw.get("data_dir", ".vector-ml"))).resolve()
        self.callers = self.raw.get("callers", {"local": {"sources": list(self.raw.get("sources", {}))}})
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", owner) for owner in self.callers):
            raise VTError("Caller IDs must use letters, digits, underscore or hyphen")
    def authorized(self, owner, source_id):
        if source_id not in self.callers.get(owner, {}).get("sources", []):
            raise VTError("Source unavailable or unauthorized")
    def source(self, source_id, owner="local"):
        self.authorized(owner, source_id)
        cfg = dict(self.raw["sources"][source_id]); kind = cfg.pop("kind")
        kwargs = cfg.pop("mapping", {})
        if kind == "demo":
            return InMemory(tickets(), name=source_id)
        if kind == "qdrant":
            from qdrant_client import QdrantClient
            client = QdrantClient(url=cfg["url"], api_key=os.getenv(cfg.get("api_key_env", "QDRANT_API_KEY")), timeout=30)
            return Qdrant(client, source_name=source_id, **kwargs)
        if kind == "weaviate":
            import weaviate
            from weaviate.classes.init import Auth
            token = os.getenv(cfg.get("api_key_env", "WEAVIATE_API_KEY"))
            client = weaviate.connect_to_custom(http_host=cfg["host"], http_port=cfg.get("port", 8080), http_secure=cfg.get("secure", False),
                grpc_host=cfg.get("grpc_host", cfg["host"]), grpc_port=cfg.get("grpc_port", 50051), grpc_secure=cfg.get("secure", False),
                auth_credentials=Auth.api_key(token) if token else None)
            return Weaviate(client, source_name=source_id, **kwargs)
        if kind == "pgvector":
            import psycopg
            from pgvector.psycopg import register_vector
            dsn = os.getenv(cfg.get("dsn_env", "PGVECTOR_DSN"))
            if not dsn:
                raise VTError("Configured PostgreSQL DSN environment variable is absent")
            conn = psycopg.connect(dsn, autocommit=True)
            register_vector(conn)
            return Pgvector(conn, source_name=source_id, **kwargs)
        if kind == "pinecone":
            from pinecone import Pinecone as Client
            key = os.getenv(cfg.get("api_key_env", "PINECONE_API_KEY"))
            if not key:
                raise VTError("Configured Pinecone API key environment variable is absent")
            client = Client(api_key=key)
            description = client.describe_index(cfg["index"])
            return Pinecone(client.Index(host=description.host), index_description=description, source_name=source_id, **kwargs)
        raise VTError(f"Unknown source kind: {kind}")

def close_source(source):
    client = getattr(source, "client", None) or getattr(source, "connection", None) or getattr(source, "index", None)
    if client is not None and callable(getattr(client, "close", None)):
        client.close()
