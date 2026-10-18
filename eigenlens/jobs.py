"""Single supervisor, spawn-isolated workers, durable jobs and truthful cancellation."""
import fcntl
import multiprocessing as mp
import os
import socket
import threading
import time
from .artifacts import Store, now
from .config import Config
from .datasets import Dataset
from .schemas import Selection, VTError
from .records import InMemory
from .backends import LocalBackend
from .registry import REGISTRY, plan
from .execution_context import numeric_budget, progress_reporter, report_stage

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
    try:
        config = Config(config_path)
        job = store.job(owner, job_id)
        payload = job["payload"]
        store.update_job(job_id, "running")
        store.update_stage(job_id, {"stage": "loading_records", "message": "Loading registered records and validating the operation plan"})
        spec = store.get(owner, payload["dataset_id"], "ds")
        ordered = plan(payload["operations"], payload["execution"]["backend"], config.limits)
        if "records" not in spec:
            raise VTError("Only registered record datasets are supported; use register_records")
        source = InMemory(spec["records"], name=spec.get("name", "registered"))
        source.identity.update(spec.get("source", {}))
        store.update_stage(job_id, {"stage": "materializing", "message": "Validating and materializing the selected records",
                                   "input_records": len(spec['records'])})
        data = Dataset(source, Selection.model_validate(spec["selection"]), config.limits).materialize(
            fields={"metadata", "text"} | set().union(*(REGISTRY[o.type]["fields"] for o in ordered)))
        # All operations share this one execution-time materialization.
        store.update_stage(job_id, {"stage": "saving_input", "message": "Saving the shared materialized input",
                                   "selected_records": len(data.records)})
        stage_id = store.put(owner, "data", data.serializable())
        with numeric_budget(config.numeric_threads), progress_reporter(lambda info: store.update_stage(job_id, info)):
            results = LocalBackend().execute(data, ordered, run_id=job_id)
            report_stage('saving_results', 'Saving analysis results', completed=0, total=len(results), unit='artifacts')
        refs = {}
        for op_id, result in results.items():
            result.summary_data["materialized_artifact_id"] = stage_id
            result.summary_data["execution_host"] = socket.gethostname()
            result.summary_data["worker_pid"] = os.getpid()
            store.put(owner, "res", result.model_dump(), result.result_id)
            refs[op_id] = result.result_id
            store.update_stage(job_id, {"stage": "saving_results", "message": "Saving analysis results",
                                       "completed": len(refs), "total": len(results), "unit": "artifacts"})
        store.update_job(job_id, "succeeded", refs)
    except BaseException as exc:
        # Internal exceptions may contain private records; only controlled errors are exposed.
        message = str(exc) if isinstance(exc, VTError) else f"{type(exc).__name__}: execution failed; inspect service diagnostics"
        store.update_job(job_id, "failed", error=message)
        import sys
        print(f"Job {job_id} failed ({type(exc).__name__})", file=sys.stderr)

class Supervisor:
    def __init__(self, config, worker_target=worker):
        self.config = config
        self.worker_target = worker_target
        self.store = Store(config.root)
        self.lockfile = (config.root / "supervisor.lock").open("a+")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lockfile.close()
            raise VTError("A supervisor already owns this data directory; use one service replica") from exc
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET status='failed', error='Service interrupted before completion',updated_at=? WHERE status IN ('queued','running')", (now(),))
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
                    p = mp.get_context("spawn").Process(target=self.worker_target, args=(str(self.config.root), self.config.path, row["id"], row["owner"]), daemon=True)
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
                db.execute("UPDATE jobs SET status='failed',error='Service shutdown before execution',updated_at=? WHERE status='queued'", (now(),))
        fcntl.flock(self.lockfile, fcntl.LOCK_UN); self.lockfile.close()
