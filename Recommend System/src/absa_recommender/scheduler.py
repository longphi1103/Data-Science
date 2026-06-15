import calendar
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from absa_recommender.config import load_yaml


def previous_month(today: datetime | None = None) -> str:
    current = today or datetime.now()
    first_day = current.replace(day=1)
    previous = first_day - timedelta(days=1)
    return previous.strftime("%Y-%m")


def previous_month_for_run(today: datetime | None = None) -> str:
    return previous_month(today)


def priority_idempotency_key(
    restaurant_id: str,
    review_month: str,
    scoring_config_hash: str,
    absa_model_version: str,
) -> tuple[str, str, str, str]:
    return (restaurant_id, review_month, scoring_config_hash, absa_model_version)


def current_or_previous_month(process_previous_month: bool = True) -> str:
    if process_previous_month:
        return previous_month()
    return datetime.now().strftime("%Y-%m")


def monthly_interval_seconds(config_path: str | Path = "configs/scheduler.yaml") -> int:
    config = load_yaml(config_path)
    monthly = config.get("monthly", {})
    day = int(monthly.get("day_of_month", 3))
    time_local = str(monthly.get("time_local", "03:00"))
    hour, minute = [int(part) for part in time_local.split(":", maxsplit=1)]
    now = datetime.now()
    target_day = min(day, calendar.monthrange(now.year, now.month)[1])
    next_run = now.replace(day=target_day, hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        year = now.year + (1 if now.month == 12 else 0)
        month = 1 if now.month == 12 else now.month + 1
        target_day = min(day, calendar.monthrange(year, month)[1])
        next_run = next_run.replace(year=year, month=month, day=target_day)
    return max(1, int((next_run - now).total_seconds()))


def run_monthly_command_once(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def scheduler_loop(
    command: list[str],
    config_path: str | Path = "configs/scheduler.yaml",
    run_immediately: bool = False,
) -> None:
    config = load_yaml(config_path)
    enabled = bool(config.get("monthly", {}).get("enabled", False))
    if not enabled:
        print(json.dumps({"status": "disabled", "config_path": str(config_path)}), flush=True)
        return

    if run_immediately:
        code = run_monthly_command_once(command)
        print(json.dumps({"status": "ran", "returncode": code}), flush=True)

    while True:
        sleep_seconds = monthly_interval_seconds(config_path)
        print(json.dumps({"status": "sleeping", "seconds": sleep_seconds}), flush=True)
        time.sleep(sleep_seconds)
        code = run_monthly_command_once(command)
        print(json.dumps({"status": "ran", "returncode": code}), flush=True)


def build_docker_monthly_command(
    input_path: str = "/app/data/gmaps_monthly_raw.jsonl",
    restaurant_id: str = "res_demo",
    top_n: int = 10,
    absa_adapter: str = "trained",
    db_path: str = "/app/data/local.duckdb",
    output: str = "/app/out/priority.json",
    force: bool = False,
    month: str | None = None,
    process_previous_month: bool = True,
) -> list[str]:
    review_month = month or current_or_previous_month(process_previous_month)
    command = [
        sys.executable,
        "-m",
        "absa_recommender.cli",
        "run-full",
        "--input",
        input_path,
        "--restaurant-id",
        restaurant_id,
        "--month",
        review_month,
        "--top-n",
        str(top_n),
        "--absa-adapter",
        absa_adapter,
        "--source-adapter",
        "google-maps",
        "--db-path",
        db_path,
        "--output",
        output,
    ]
    if force:
        command.append("--force")
    return command


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Monthly ABSA crawler/pipeline scheduler")
    parser.add_argument("--config", default="configs/scheduler.yaml")
    parser.add_argument("--input", default="/app/data/gmaps_monthly_raw.jsonl")
    parser.add_argument("--restaurant-id", default="res_demo")
    parser.add_argument("--month")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--absa-adapter", default="trained")
    parser.add_argument("--db-path", default="/app/data/local.duckdb")
    parser.add_argument("--output", default="/app/out/priority.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-immediately", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    command = build_docker_monthly_command(
        input_path=args.input,
        restaurant_id=args.restaurant_id,
        month=args.month,
        top_n=args.top_n,
        absa_adapter=args.absa_adapter,
        db_path=args.db_path,
        output=args.output,
        force=args.force,
        process_previous_month=bool(config.get("monthly", {}).get("process_previous_month", True)),
    )
    scheduler_loop(command, config_path=args.config, run_immediately=args.run_immediately)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())