from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from absa_recommender.monthly_pipeline import run_monthly_from_source


@dataclass(frozen=True)
class ExploreJobRequest:
    input_path: str
    restaurant_id: str
    review_month: str
    top_n: int
    db_path: str
    force: bool
    area_id: str
    absa_adapter: str
    source_adapter: str
    gmaps_live: bool
    gmaps_discover_from_area: bool
    gmaps_area_name: str | None
    gmaps_bbox: str | None
    gmaps_target_url: str | None


def start_explore_job(
    request: ExploreJobRequest,
    *,
    job_dir: str | Path = "data/explore_jobs",
    python_executable: str | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    job_root = Path(job_dir)
    job_root.mkdir(parents=True, exist_ok=True)
    running = read_running_job_status(job_root)
    if running.get("status") == "running":
        running = dict(running)
        running["reused_existing"] = True
        return running

    job_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    request_path = job_root / f"{job_id}.request.json"
    status_path = job_root / f"{job_id}.status.json"
    log_path = job_root / f"{job_id}.log"

    request_path.write_text(json.dumps(asdict(request), ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    status = _base_status(job_id, request, status_path, log_path)
    _write_status(status_path, status)

    command = [
        python_executable or sys.executable,
        "-m",
        "absa_recommender.explore_jobs",
        "run",
        "--request",
        str(request_path),
        "--status",
        str(status_path),
        "--log",
        str(log_path),
    ]
    process = popen_factory(
        command,
        stdout=None,
        stderr=None,
        text=True,
        **_process_isolation_kwargs(),
    )
    status["pid"] = getattr(process, "pid", None)
    _write_status(status_path, status)
    return status



def _process_isolation_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}

def read_job_status(status_path: str | Path | None) -> dict[str, Any]:
    if not status_path:
        return _idle_status()
    path = Path(status_path)
    if not path.exists():
        return _idle_status()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "error", "message": f"Invalid job status file: {path}", "status_path": str(path)}


def read_latest_job_status(job_dir: str | Path = "data/explore_jobs") -> dict[str, Any]:
    root = Path(job_dir)
    if not root.exists():
        return _idle_status()
    status_files = sorted(root.glob("*.status.json"))
    if not status_files:
        return _idle_status()
    return read_job_status(status_files[-1])


def read_running_job_status(
    job_dir: str | Path = "data/explore_jobs",
    *,
    is_pid_running: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    root = Path(job_dir)
    if not root.exists():
        return _idle_status()
    pid_checker = is_pid_running or is_process_running
    status_files = sorted(root.glob("*.status.json"), reverse=True)
    for status_file in status_files:
        status = read_job_status(status_file)
        if status.get("status") != "running":
            continue
        pid = status.get("pid")
        if pid is None or pid_checker(int(pid)):
            return status
        stale = dict(status)
        stale.update(
            {
                "status": "failed",
                "message": f"Explore job process {pid} is no longer running.",
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }
        )
        _write_status(status_file, stale)
    return _idle_status()


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except ValueError:
        return False
    return True


def read_job_log(log_path: str | Path | None, *, max_lines: int = 50) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def coerce_progress_percent(value: Any) -> int:
    try:
        percent = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, percent))


def run_job(request_path: str | Path, status_path: str | Path, log_path: str | Path) -> int:
    request_data = json.loads(Path(request_path).read_text(encoding="utf-8"))
    request = ExploreJobRequest(**request_data)
    status_file = Path(status_path)
    log_file = Path(log_path)

    def log(message: str, percent: int | None = None) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        suffix = f" ({percent}%)" if percent is not None else ""
        line = f"[{timestamp}] {message}{suffix}"
        print(f"[streamlit-explore] {line}", flush=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        current = read_job_status(status_file)
        current.update(
            {
                "status": "running",
                "message": message,
                "percent": percent,
                "updated_at": _utc_now(),
            }
        )
        _write_status(status_file, current)

    try:
        log("Explore pipeline started", 1)
        result = run_monthly_from_source(
            request.input_path,
            restaurant_id=request.restaurant_id,
            review_month=request.review_month,
            top_n=request.top_n,
            db_path=request.db_path,
            force=request.force,
            area_id=request.area_id,
            absa_adapter=request.absa_adapter,
            source_adapter=request.source_adapter,
            gmaps_live=request.gmaps_live,
            gmaps_discover_from_area=request.gmaps_discover_from_area,
            gmaps_area_name=request.gmaps_area_name,
            gmaps_bbox=request.gmaps_bbox,
            gmaps_target_url=request.gmaps_target_url,
            progress_callback=log,
        )
    except Exception as exc:
        print(f"[streamlit-explore] Explore pipeline failed: {exc}", flush=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write(f"Explore pipeline failed: {exc}\n")
        failed = read_job_status(status_file)
        failed.update({"status": "failed", "message": str(exc), "updated_at": _utc_now(), "finished_at": _utc_now()})
        _write_status(status_file, failed)
        return 1

    completed = read_job_status(status_file)
    completed.update(
        {
            "status": "success",
            "message": "Explore pipeline completed",
            "percent": 100,
            "priority_run_id": result.get("priority_run_id"),
            "result": _compact_result(result),
            "updated_at": _utc_now(),
            "finished_at": _utc_now(),
        }
    )
    _write_status(status_file, completed)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--request", required=True)
    run_parser.add_argument("--status", required=True)
    run_parser.add_argument("--log", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_job(args.request, args.status, args.log)
    return 2


def _base_status(job_id: str, request: ExploreJobRequest, status_path: Path, log_path: Path) -> dict[str, Any]:
    now = _utc_now()
    return {
        "job_id": job_id,
        "status": "running",
        "message": "Explore job started",
        "percent": 0,
        "pid": None,
        "restaurant_id": request.restaurant_id,
        "review_month": request.review_month,
        "status_path": str(status_path),
        "log_path": str(log_path),
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "priority_run_id": result.get("priority_run_id"),
        "source_reviews_fetched": result.get("source_reviews_fetched"),
        "source_duplicates": result.get("source_duplicates"),
        "source_restaurants": result.get("source_restaurants"),
        "absa_adapter": result.get("absa_adapter"),
        "source_adapter": result.get("source_adapter"),
        "db_path": result.get("db_path"),
    }


def _idle_status() -> dict[str, Any]:
    return {"status": "idle", "message": "No Explore job has been started."}


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

