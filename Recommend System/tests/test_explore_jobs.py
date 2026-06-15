import json
from pathlib import Path

from absa_recommender.explore_jobs import (
    ExploreJobRequest,
    coerce_progress_percent,
    read_job_status,
    read_latest_job_status,
    read_running_job_status,
    start_explore_job,
)


class FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid


def test_start_explore_job_writes_request_status_and_log_path(tmp_path: Path) -> None:
    launched = {}

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    request = ExploreJobRequest(
        input_path="data/raw.jsonl",
        restaurant_id="res_demo",
        review_month="2026-06",
        top_n=5,
        db_path="data/app.duckdb",
        force=False,
        area_id="area_demo",
        absa_adapter="placeholder",
        source_adapter="google-maps",
        gmaps_live=True,
        gmaps_discover_from_area=True,
        gmaps_area_name="PhÆ°á»ng Test, HÃ  Ná»™i, Viá»‡t Nam",
        gmaps_bbox=None,
        gmaps_target_url="https://www.google.com/maps/place/demo",
    )

    status = start_explore_job(
        request,
        job_dir=tmp_path,
        python_executable="python-test",
        popen_factory=fake_popen,
    )

    assert status["status"] == "running"
    assert status["pid"] == 4321
    assert status["restaurant_id"] == "res_demo"
    assert Path(status["log_path"]).exists()
    assert read_job_status(status["status_path"])["job_id"] == status["job_id"]

    request_path = Path(launched["command"][launched["command"].index("--request") + 1])
    assert json.loads(request_path.read_text(encoding="utf-8"))["review_month"] == "2026-06"
    assert launched["command"][:3] == ["python-test", "-m", "absa_recommender.explore_jobs"]
    assert launched["kwargs"]["stdout"] is None
    assert launched["kwargs"]["stderr"] is None
    assert launched["kwargs"].get("creationflags", 0) or launched["kwargs"].get("start_new_session")


def test_read_job_status_marks_missing_file_as_idle(tmp_path: Path) -> None:
    status = read_job_status(tmp_path / "missing.json")

    assert status["status"] == "idle"
    assert status["message"] == "No Explore job has been started."


def test_read_latest_job_status_returns_newest_status_file(tmp_path: Path) -> None:
    older = tmp_path / "20260606010101-aaaaaaaa.status.json"
    newer = tmp_path / "20260606020202-bbbbbbbb.status.json"
    older.write_text(json.dumps({"job_id": "older", "status": "success"}), encoding="utf-8")
    newer.write_text(json.dumps({"job_id": "newer", "status": "running"}), encoding="utf-8")

    status = read_latest_job_status(tmp_path)

    assert status["job_id"] == "newer"
    assert status["status"] == "running"


def test_read_running_job_status_prefers_running_over_newer_finished(tmp_path: Path) -> None:
    running = tmp_path / "20260606010101-aaaaaaaa.status.json"
    newer_finished = tmp_path / "20260606020202-bbbbbbbb.status.json"
    running.write_text(json.dumps({"job_id": "running", "status": "running"}), encoding="utf-8")
    newer_finished.write_text(json.dumps({"job_id": "finished", "status": "success"}), encoding="utf-8")

    status = read_running_job_status(tmp_path)

    assert status["job_id"] == "running"
    assert status["status"] == "running"



def test_read_running_job_status_marks_dead_pid_as_stale(tmp_path: Path) -> None:
    running = tmp_path / "20260606010101-aaaaaaaa.status.json"
    running.write_text(
        json.dumps({"job_id": "stale", "status": "running", "pid": 999999, "status_path": str(running)}),
        encoding="utf-8",
    )

    status = read_running_job_status(tmp_path, is_pid_running=lambda pid: False)

    assert status["status"] == "idle"
    persisted = json.loads(running.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert "no longer running" in persisted["message"]


def test_start_explore_job_returns_existing_running_job_without_launching(tmp_path: Path) -> None:
    existing = tmp_path / "20260606010101-aaaaaaaa.status.json"
    existing.write_text(
        json.dumps({"job_id": "existing", "status": "running", "status_path": str(existing)}),
        encoding="utf-8",
    )

    def fail_popen(command, **kwargs):
        raise AssertionError("should not launch a second Explore job")

    request = ExploreJobRequest(
        input_path="data/raw.jsonl",
        restaurant_id="res_demo",
        review_month="2026-06",
        top_n=5,
        db_path="data/app.duckdb",
        force=False,
        area_id="area_demo",
        absa_adapter="placeholder",
        source_adapter="google-maps",
        gmaps_live=True,
        gmaps_discover_from_area=True,
        gmaps_area_name="PhÆ°á»ng Test, HÃ  Ná»™i, Viá»‡t Nam",
        gmaps_bbox=None,
        gmaps_target_url="https://www.google.com/maps/place/demo",
    )

    status = start_explore_job(request, job_dir=tmp_path, popen_factory=fail_popen)

    assert status["job_id"] == "existing"
    assert status["status"] == "running"


def test_coerce_progress_percent_handles_missing_and_bounds() -> None:
    assert coerce_progress_percent(None) == 0
    assert coerce_progress_percent("bad") == 0
    assert coerce_progress_percent(-5) == 0
    assert coerce_progress_percent(42.7) == 42
    assert coerce_progress_percent(120) == 100

