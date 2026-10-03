"""Shared application API, independent of MCP transport."""
import base64
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from .artifacts import Store, now
from .config import Config, close_source
from .datasets import canonical, Dataset
from .schemas import Selection, Execution, Result, Record, VTError
from .registry import plan, REGISTRY
from .backends import LocalBackend, SPARK_CAPABILITIES
from .operators import VaderModel
from .jobs import Supervisor

class Service:
    def __init__(self, config=None):
        self.config = config or Config()
        self.store = Store(self.config.root)
        self.supervisor = Supervisor(self.config)
    def close(self):
        self.supervisor.close()
    def get_capabilities(self, owner):
        try:
            model = VaderModel()
            sentiment = {"available": True, "model": "vader", "version": model.version}
        except VTError as exc:
            sentiment = {"available": False, "error": str(exc)}
        from .sources.base import Adapter
        source_capabilities = {}
        for source_id in self.config.callers.get(owner, {}).get("sources", []):
            kind = self.config.raw["sources"][source_id]["kind"]
            caps = Adapter().capabilities()
            if kind == "pgvector":
                caps.update(filtered_reads="native", consistency="read-only repeatable read")
            if kind == "demo":
                caps.update(exact_count="native", iteration="adapter", id_fetch="adapter", projection="adapter")
            if kind == "pinecone":
                caps.update(iteration="unavailable", seeded_sampling="unavailable", projection="unavailable",
                            constraints="describe_source verifies serverless deployment before enabling listing and sampling")
            source_capabilities[source_id] = caps
        return {"schema_version": "1.0", "source_capabilities": source_capabilities, "sources": self.config.callers.get(owner, {}).get("sources", []),
                "operations": {k: {"algorithm": v["algorithm"], "parameters": sorted(v["params"]), "required_fields": sorted(v["fields"]), "algorithms": {a: sorted(p) for a,p in v.get("algorithms", {v["algorithm"]: v["params"]}).items()}} for k,v in REGISTRY.items()},
                "sentiment": sentiment, "backends": [LocalBackend().capabilities(), SPARK_CAPABILITIES], "limits": self.config.limits.model_dump()}
    def describe_source(self, owner, source_id):
        source = self.config.source(source_id, owner)
        try:
            if hasattr(source, "inspect"):
                source.inspect()
            if hasattr(source, "collection") and callable(source.collection):
                source.collection()
            return {"source_id": source_id, "identity": source.identity, "capabilities": source.capabilities(),
                    "available_fields": ["id", "vector", "text", "metadata", "parent_id"]}
        finally:
            close_source(source)
    def create_dataset(self, owner, source_id, selection):
        selection = Selection.model_validate(selection)
        source = self.config.source(source_id, owner)
        try:
            source.validate(selection)
            if hasattr(source, "collection") and callable(source.collection):
                source.collection()
            if hasattr(source, "query"):
                source.query(selection, {"vector", "text", "metadata"})
            if selection.sample_n and selection.sample_n > self.config.limits.max_rows:
                raise VTError("Sample exceeds max_rows")
            if selection.ids is not None and len(selection.ids) > self.config.limits.max_scan_rows:
                raise VTError("ID selection exceeds max_scan_rows")
            dataset = Dataset(source, selection, self.config.limits)
            spec = {"source_id": source_id, "source_config": self.config.raw["sources"][source_id], "selection": selection.model_dump()}
            id = self.store.put(owner, "ds", spec)
            return {"dataset_id": id, "scope": dataset.scope, "selection": selection.model_dump(),
                    "count": source.count(selection) if hasattr(source, "count") else None, "count_kind": "exact" if hasattr(source, "count") else "unknown", "requested_id_count": len(selection.ids) if selection.ids is not None else None,
                    "count_unit": "vector_records", "source": source.identity, "available_fields": ["id", "vector", "text", "metadata", "parent_id"],
                    "read_plan": {"filter": "source" if getattr(source, "filter_pushdown", False) else "client", "sampling": "client full bounded scan" if selection.sample_n else None}}
        finally:
            close_source(source)
    def register_records(self, owner, records=None, artifact_name=None, selection_scope="retrieval_results"):
        if owner not in self.config.callers:
            raise VTError("Caller unauthorized")
        if (records is None) == (artifact_name is None):
            raise VTError("Supply exactly one of records or artifact_name")
        if artifact_name is not None:
            # Application stages JSONL into its principal-specific ingress folder. No URLs or arbitrary paths.
            if Path(artifact_name).name != artifact_name or not artifact_name.endswith(".jsonl"):
                raise VTError("Artifact must be a simple .jsonl filename in configured ingress/<caller>/")
            folder = (self.config.root / "ingress" / owner).resolve()
            path = (folder / artifact_name).resolve()
            if path.parent != folder or not path.is_file():
                raise VTError("Registered artifact is unavailable")
            if path.stat().st_size > self.config.limits.max_bytes:
                raise VTError("Artifact exceeds max_bytes")
            records = []
            with path.open() as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
                    if len(records) > self.config.limits.max_rows:
                        raise VTError("Artifact exceeds max_rows")
        if len(records) > self.config.limits.max_rows:
            raise VTError("Registration exceeds max_rows")
        records = [Record.model_validate(r).model_dump(mode="json") for r in records]
        if len(canonical(records).encode()) > self.config.limits.max_bytes:
            raise VTError("Registration exceeds max_bytes")
        if len({r["id"] for r in records}) != len(records):
            raise VTError("Duplicate registration IDs")
        if selection_scope not in {"retrieval_results", "complete_selection"}:
            raise VTError("Registration scope must be retrieval_results or complete_selection")
        spec = {"records": records, "selection": Selection(ids=[r["id"] for r in records] if selection_scope == "retrieval_results" else None).model_dump(), "name": "registered"}
        id = self.store.put(owner, "ds", spec)
        return {"dataset_id": id, "count": len(records), "count_kind": "exact", "count_unit": "vector_records", "scope": selection_scope}
    def run_analysis(self, owner, dataset_id, operations, execution=None, idempotency_key=None):
        self.store.get(owner, dataset_id, "ds")
        execution = Execution.model_validate(execution or {})
        if execution.mode != "async":
            raise VTError("Service jobs require async mode. Use the Python SDK for synchronous execution.")
        ordered = plan(operations, execution.backend, self.config.limits)
        if any(o.type == "sentiment" for o in ordered):
            VaderModel()  # Availability check before launch.
        payload = {"dataset_id": dataset_id, "operations": [o.model_dump() for o in ordered], "execution": execution.model_dump()}
        digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 128:
            raise VTError("idempotency_key must contain 1..128 characters")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT id, request_hash FROM jobs WHERE owner=? AND idempotency_key=?", (owner, idempotency_key)).fetchone() if idempotency_key else None
            if old:
                if old["request_hash"] != digest:
                    raise VTError("Idempotency key already used for a different request")
                return self.get_job(owner, old["id"])
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= self.config.limits.max_pending_jobs:
                raise VTError("Job queue is full; retry later with the same idempotency key")
            id = "job_" + uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,owner,status,payload,created_at,updated_at,idempotency_key,request_hash) VALUES(?,?,?,?,?,?,?,?)", (id, owner, "queued", canonical(payload), now(), now(), idempotency_key, digest))
        return self.get_job(owner, id)
    def get_job(self, owner, job_id):
        job = self.store.job(owner, job_id)
        return {"job_id": job_id, **{k:job[k] for k in ["status", "results", "error", "created_at", "updated_at"]},
                "backend": "local", "progress": {"phase": job["status"], "fraction": 1 if job["status"] == "succeeded" else None}}
    def cancel_job(self, owner, job_id):
        self.supervisor.cancel(owner, job_id)
        return self.get_job(owner, job_id)
    def result(self, owner, result_id):
        return Result.model_validate(self.store.get(owner, result_id, "res"))
    def get_summary(self, owner, result_id):
        result = self.result(owner, result_id)
        summary = result.summary()
        # Centroids can be large: full values remain in the durable result artifact.
        if "centroids" in summary["summary_data"]:
            centers = summary["summary_data"].pop("centroids")
            summary["summary_data"]["centroid_shape"] = [len(centers), len(centers[0]) if centers else 0]
        selection = summary["summary_data"].pop("selection", {})
        summary["summary_data"]["selection_context"] = {
            "sample_n": selection.get("sample_n"), "seed": selection.get("seed"),
            "sample_population": "retrieved IDs" if selection.get("ids") is not None else "complete bounded selection",
            "filter_count": len(selection.get("filters", [])),
        }
        if "topics" in summary["summary_data"]:
            rows = summary["summary_data"]["topics"]
            summary["summary_data"]["topics"] = [{**r, "terms": r["terms"][:10]} for r in rows[:20]]
            summary["summary_data"]["topic_groups_total"] = len(rows)
            summary["summary_data"]["topic_groups_shown"] = min(20, len(rows))
        return summary
    def get_examples(self, owner, result_id, cluster_id=None, limit=5):
        return {"result_id": result_id, "examples": self.result(owner, result_id).examples(cluster_id, limit)}
    def get_result_page(self, owner, result_id, cursor=None, limit=100):
        if not 1 <= limit <= 1000:
            raise VTError("limit must be 1..1000")
        result = self.result(owner, result_id)
        offset = 0
        if cursor:
            try:
                token = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if token["result_id"] != result_id or token["fingerprint"] != result.dataset_fingerprint:
                    raise ValueError()
                offset = token["offset"]
                if type(offset) is not int or not 0 <= offset <= len(result.rows):
                    raise ValueError()
            except Exception as exc:
                raise VTError("Invalid cursor or cursor belongs to another result") from exc
        end = min(offset + limit, len(result.rows))
        next_cursor = base64.urlsafe_b64encode(canonical({"result_id": result_id, "fingerprint": result.dataset_fingerprint, "offset": end}).encode()).decode() if end < len(result.rows) else None
        return {"result_id": result_id, "rows": result.rows[offset:end], "next_cursor": next_cursor, "total": len(result.rows)}
