"""Pause before the real clustering worker to test queue and cancellation deterministically."""
from pathlib import Path
import time
from vector_tools.artifacts import Store
from vector_tools.jobs import worker

def paused_cluster_worker(root, config_path, job_id, owner):
    root = Path(root)
    Store(root).update_job(job_id, "running")
    (root / f"started-{job_id}").touch()
    deadline = time.monotonic() + 45
    while not (root / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Test did not release worker")
        time.sleep(.01)
    worker(str(root), config_path, job_id, owner)

def wait_started(root, job_id):
    deadline = time.monotonic() + 30
    while not (Path(root) / f"started-{job_id}").exists():
        assert time.monotonic() < deadline, "Worker did not start"
        time.sleep(.02)
