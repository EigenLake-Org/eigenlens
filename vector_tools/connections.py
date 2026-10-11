"""Caller-owned, encrypted database connection definitions (no live clients)."""
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .artifacts import now
from .schemas import VTError


class ConnectionStore:
    def __init__(self, root):
        self.path = Path(root) / "connections.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS connections(
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, db_type TEXT NOT NULL,
                    name TEXT NOT NULL, encrypted BLOB NOT NULL, created_at TEXT NOT NULL
                );
            """)
        self.path.chmod(0o600)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def cipher():
        key = os.getenv("VECTOR_ML_CONNECTION_KEY")
        if not key:
            raise VTError("Set VECTOR_ML_CONNECTION_KEY to a Fernet key before using database connections")
        try:
            return Fernet(key.encode())
        except (ValueError, TypeError):
            raise VTError("VECTOR_ML_CONNECTION_KEY must be a valid Fernet key") from None

    def put(self, owner, spec, name=None, max_connections=128):
        if name is not None and (not name.strip() or len(name) > 128):
            raise VTError("Connection name must contain 1..128 characters")
        encrypted = self.cipher().encrypt(json.dumps({"owner": owner, "spec": spec}, allow_nan=False).encode())
        connection_id = "conn_" + uuid.uuid4().hex
        created_at = now()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM connections WHERE owner=?", (owner,)).fetchone()[0] >= max_connections:
                raise VTError("Connection limit reached; disconnect unused connections")
            db.execute("INSERT INTO connections VALUES(?,?,?,?,?,?)",
                       (connection_id, owner, spec["db_type"], name or spec["db_type"], encrypted, created_at))
        return {"connection_id": connection_id, "db_type": spec["db_type"],
                "name": name or spec["db_type"], "created_at": created_at}

    def get(self, owner, connection_id):
        with self.db() as db:
            row = db.execute("SELECT encrypted FROM connections WHERE id=? AND owner=?",
                             (connection_id, owner)).fetchone()
        if row is None:
            raise VTError("Connection unavailable or unauthorized")
        try:
            value = json.loads(self.cipher().decrypt(row["encrypted"]))
            if value["owner"] != owner:
                raise VTError("Connection unavailable or unauthorized")
            return value["spec"]
        except InvalidToken:
            raise VTError("Connection cannot be decrypted; restore its original VECTOR_ML_CONNECTION_KEY") from None

    def list(self, owner):
        with self.db() as db:
            rows = db.execute("SELECT id,db_type,name,created_at FROM connections WHERE owner=? ORDER BY created_at,id",
                              (owner,)).fetchall()
        return [{"connection_id": row["id"], **{key: row[key] for key in ("db_type", "name", "created_at")}}
                for row in rows]

    def delete(self, owner, connection_id):
        with self.db() as db:
            changed = db.execute("DELETE FROM connections WHERE id=? AND owner=?", (connection_id, owner)).rowcount
        if not changed:
            raise VTError("Connection unavailable or unauthorized")
        return {"connection_id": connection_id, "status": "disconnected"}
