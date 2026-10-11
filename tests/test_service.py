import time
import pytest
from vector_tools.service import Service
from vector_tools.schemas import VTError
from vector_tools.demo import tickets

def wait(service, id):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        j = service.get_job("local", id)
        if j["status"] in {"succeeded", "failed", "cancelled"}:
            return j
        time.sleep(.05)
    raise AssertionError("job timeout")

@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_ML_DATA", str(tmp_path))
    service = Service()
    yield service
    service.close()

def test_durable_jobs_idempotency_access_pages(service):
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()[:30]])
    ops = [{"id":"c", "type":"cluster", "parameters":{"n_clusters":3}}]
    job = service.run_analysis("local", ds["dataset_id"], ops, idempotency_key="test")
    duplicate = service.run_analysis("local", ds["dataset_id"], ops, idempotency_key="test")
    assert duplicate["job_id"] == job["job_id"]
    with pytest.raises(VTError, match="different"):
        service.run_analysis("local", ds["dataset_id"], [{"id":"c","type":"cluster", "parameters":{"n_clusters":2}}], idempotency_key="test")
    j = wait(service, job["job_id"])
    assert j["status"] == "succeeded", j
    ref = j["results"]["c"]
    assert service.get_summary("local", ref)["processed"] == 30
    page = service.get_result_page("local", ref, limit=7)
    ids = [r["id"] for r in page["rows"]]
    while page["next_cursor"]:
        page = service.get_result_page("local", ref, page["next_cursor"], limit=7)
        ids += [r["id"] for r in page["rows"]]
    assert len(set(ids)) == 30
    with pytest.raises(VTError, match="unauthorized"):
        service.get_summary("other", ref)
    with pytest.raises(VTError, match="unauthorized"):
        service.get_job("other", j["job_id"])
    from vector_tools.artifacts import Store
    assert Store(service.config.root).get("local", ref, "res")["processed"] == 30

def test_failed_cancelled_and_artifact(service):
    ds = service.register_records("local", records=[])
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"c","type":"cluster"}])
    assert wait(service, job["job_id"])["status"] == "failed"
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()])
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"c","type":"cluster"}])
    result = service.cancel_job("local", job["job_id"])
    assert result["status"] in {"cancelled", "succeeded"}
    assert job["job_id"] not in service.supervisor.running
    folder = service.config.root / "ingress" / "local"
    folder.mkdir()
    (folder / "records.jsonl").write_text('\n'.join(r.model_dump_json() for r in tickets()))
    assert service.register_records("local", artifact_name="records.jsonl")["count"] == 62
    with pytest.raises(VTError):
        service.register_records("local", artifact_name="../records.jsonl")

def test_restart_marks_interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_ML_DATA", str(tmp_path))
    from vector_tools.artifacts import Store, now
    store = Store(tmp_path)
    with store.connect() as db:
        db.execute("INSERT INTO jobs(id,owner,status,payload,created_at,updated_at) VALUES(?,?,?,?,?,?)", ("job_interrupted", "local", "running", "{}", now(), now()))
    service = Service()
    try:
        assert service.get_job("local", "job_interrupted")["status"] == "failed"
        with pytest.raises(VTError, match="supervisor"):
            Service()
    finally:
        service.close()

def test_registration_scope_and_persistent_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_ML_DATA", str(tmp_path))
    service = Service()
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()])
    assert ds["scope"] == "retrieval_results"
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"s", "type":"cluster", "parameters":{"n_clusters":3}}])
    done = wait(service, job["job_id"])
    assert done["status"] == "succeeded"
    ref = done["results"]["s"]
    before = service.get_summary("local", ref)
    service.close()
    restarted = Service()
    try:
        assert restarted.get_summary("local", ref) == before
        assert restarted.get_job("local", job["job_id"])["results"]["s"] == ref
        assert before["scope"] == "retrieval_results"
    finally:
        restarted.close()

def test_responsive_and_cancel_active_worker(service):
    from worker_helpers import paused_cluster_worker, wait_started
    service.supervisor.worker_target = paused_cluster_worker
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()])
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"groups", "type":"cluster"}])
    wait_started(service.config.root, job["job_id"])
    started = time.monotonic()
    assert set(service.get_capabilities("local")["operations"]) == {"cluster", "topics"}
    assert time.monotonic() - started < 1
    assert service.cancel_job("local", job["job_id"])["status"] == "cancelled"
    assert job["job_id"] not in service.supervisor.running

def test_old_source_dataset_rejected_before_enqueue(service):
    ds = service.store.put("local", "ds", {"source_id":"removed", "selection":{}})
    with pytest.raises(VTError, match="register_records"):
        service.run_analysis("local", ds, [{"id":"groups", "type":"cluster"}])
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
