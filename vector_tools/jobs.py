"""Single supervisor, spawn-isolated workers, durable jobs and truthful cancellation."""
import fcntl
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
from .artifacts import Store
from .config import Config, close_source
from .datasets import Dataset
from .schemas import Selection, VTError
from .sources import InMemory
from .backends import LocalBackend
from .registry import REGISTRY, plan

TERMINAL = {"succeeded", "failed", "cancelled"}

def worker(root, config_path, job_id, owner):
    # Worker belongs to supervisor lifetime; if parent dies, do not continue orphan computation.
    parent = os.getppid()
    def watch_parent():
        while True:
            time.sleep(.25)
            if os.getppid() != parent:
                os._exit(70)
    threading.Thread(target=watch_parent, daemon=True).start()
    store = Store(root)
    source = None
    try:
        config = Config(config_path)
        job = store.job(owner, job_id)
        payload = job["payload"]
        spec = store.get(owner, payload["dataset_id"], "ds")
        ordered = plan(payload["operations"], payload["execution"]["backend"], config.limits)
        store.update_job(job_id, "running")
        if "records" in spec:
            source = InMemory(spec["records"], name=spec.get("name", "registered"))
        else:
            if spec.get("source_config") != config.raw.get("sources", {}).get(spec["source_id"]):
                raise VTError("Source configuration changed since dataset creation; create a new dataset handle")
            source = config.source(spec["source_id"], owner)
        data = Dataset(source, Selection.model_validate(spec["selection"]), config.limits).materialize(
            fields={"metadata", "text"} | set().union(*(REGISTRY[o.type]["fields"] for o in ordered)))
        # All operations share this one execution-time materialization.
        stage_id = store.put(owner, "data", data.serializable())
        results = LocalBackend().execute(data, ordered, run_id=job_id)
        refs = {}
        for op_id, result in results.items():
            result.summary_data["materialized_artifact_id"] = stage_id
            store.put(owner, "res", result.model_dump(), result.result_id)
            refs[op_id] = result.result_id
        store.update_job(job_id, "succeeded", refs)
    except BaseException as exc:
        # Source/driver exceptions can contain credentials; only controlled errors are exposed.
        message = str(exc) if isinstance(exc, VTError) else f"{type(exc).__name__}: execution failed; inspect service diagnostics"
        store.update_job(job_id, "failed", error=message)
        import sys
        print(f"Job {job_id} failed ({type(exc).__name__})", file=sys.stderr)
    finally:
        if source is not None:
            close_source(source)

class Supervisor:
    def __init__(self, config):
        self.config = config
        self.store = Store(config.root)
        self.lockfile = (config.root / "supervisor.lock").open("a+")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lockfile.close()
            raise VTError("A supervisor already owns this data directory; use one service replica") from exc
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET status='failed', error='Service interrupted before completion' WHERE status IN ('queued','running')")
        self.running = {}
        self.mutex = threading.RLock()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
    def loop(self):
        while not self.stopped.wait(.1):
            with self.mutex:
                for job_id, (process, started, owner) in list(self.running.items()):
                    if process.is_alive() and time.monotonic() - started > self.config.limits.max_seconds:
                        process.terminate(); process.join(2)
                        if process.is_alive():
                            process.kill(); process.join()
                        self.store.update_job(job_id, "failed", error="Execution exceeded max_seconds; worker terminated")
                    if not process.is_alive():
                        process.join()
                        if self.store.job(owner, job_id)["status"] not in TERMINAL:
                            self.store.update_job(job_id, "failed", error=f"Worker exited before completion (exit {process.exitcode})")
                        process.close()
                        del self.running[job_id]
                with self.store.connect() as db:
                    queued = db.execute("SELECT id,owner FROM jobs WHERE status='queued' ORDER BY created_at LIMIT ?", (self.config.limits.max_jobs,)).fetchall()
                for row in queued:
                    if len(self.running) >= self.config.limits.max_jobs:
                        break
                    if row["id"] in self.running:
                        continue
                    p = mp.get_context("spawn").Process(target=worker, args=(str(self.config.root), self.config.path, row["id"], row["owner"]), daemon=True)
                    p.start()
                    self.running[row["id"]] = (p, time.monotonic(), row["owner"])
    def cancel(self, owner, job_id):
        with self.mutex:
            job = self.store.job(owner, job_id)
            if job["status"] in TERMINAL:
                return job
            entry = self.running.get(job_id)
            if entry:
                p = entry[0]
                p.terminate(); p.join(2)
                if p.is_alive():
                    p.kill(); p.join()
                p.close(); del self.running[job_id]
            # If completion won the race, retain success and its result handles.
            job = self.store.job(owner, job_id)
            if job["status"] not in TERMINAL:
                self.store.update_job(job_id, "cancelled", error="Worker stopped; unfinished artifacts may remain for operator cleanup")
            return self.store.job(owner, job_id)
    def close(self):
        self.stopped.set(); self.thread.join()
        with self.mutex:
            for job_id, (_, _, owner) in list(self.running.items()):
                self.cancel(owner, job_id)
            with self.store.connect() as db:
                db.execute("UPDATE jobs SET status='failed',error='Service shutdown before execution' WHERE status='queued'")
        fcntl.flock(self.lockfile, fcntl.LOCK_UN); self.lockfile.close()
