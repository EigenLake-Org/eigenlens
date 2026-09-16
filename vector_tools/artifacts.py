"""Atomic JSON artifacts and transactional SQLite metadata scoped to a principal."""
from pathlib import Path
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from .datasets import canonical
from .schemas import VTError

def now():
    return datetime.now(timezone.utc).isoformat()

class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        (self.root / "ingress").mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS objects(id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, results TEXT NOT NULL DEFAULT '{}', error TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, idempotency_key TEXT, request_hash TEXT,
                    UNIQUE(owner,idempotency_key));
            ''')
    def connect(self):
        db = sqlite3.connect(self.root / "metadata.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        return db
    def put(self, owner, kind, value, id=None):
        id = id or kind + "_" + uuid.uuid4().hex
        path = self.root / "artifacts" / (id + ".json")
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        with temporary.open("w") as f:
            f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        with self.connect() as db:
            db.execute("INSERT INTO objects VALUES(?,?,?,?)", (id, owner, kind, path.name))
        return id
    def get(self, owner, id, kind=None):
        with self.connect() as db:
            row = db.execute("SELECT * FROM objects WHERE id=? AND owner=?", (id, owner)).fetchone()
        if row is None or (kind and row["kind"] != kind):
            raise VTError("Handle unavailable, expired, or unauthorized")
        path = self.root / "artifacts" / row["path"]
        if not path.is_file():
            raise VTError("Handle artifact is unavailable")
        return json.loads(path.read_text())
    def job(self, owner, id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (id, owner)).fetchone()
        if row is None:
            raise VTError("Job unavailable or unauthorized")
        row = dict(row)
        row["payload"] = json.loads(row["payload"])
        row["results"] = json.loads(row["results"])
        return row
    def update_job(self, id, status, results=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,results=?,error=?,updated_at=? WHERE id=?", (status, canonical(results or {}), error, now(), id))
