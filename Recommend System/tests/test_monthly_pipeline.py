from pathlib import Path

from absa_recommender.monthly_pipeline import run_monthly_from_absa_jsonl
from absa_recommender.storage import (
    dashboard_payload,
    get_aspect_monthly_stats,
    get_latest_priority_response,
    list_peer_benchmark,
    list_priority_runs,
    list_restaurants,
    list_review_months,
)


SAMPLE_PATH = Path("data/samples/streamlit_priority_200.jsonl")


def test_run_monthly_from_absa_jsonl_persists_full_pipeline(tmp_path: Path) -> None:
    db_path = tmp_path / "pipeline.duckdb"

    result = run_monthly_from_absa_jsonl(
        SAMPLE_PATH,
        restaurant_id="res_demo",
        review_month="2026-06",
        top_n=10,
        db_path=db_path,
        force=True,
    )

    assert result["status"] == "completed"
    assert result["reviews_inserted"] == 200
    assert result["annotations"] == 671
    assert result["aspect_stats"] > 0
    assert result["peer_benchmarks"] > 0
    assert result["output"]["items"]
    assert list_restaurants(db_path)
    assert list_review_months(db_path, "res_demo") == ["2026-06"]
    assert list_priority_runs(db_path, "res_demo")
    assert get_latest_priority_response(db_path, "res_demo", "2026-06") is not None
    assert get_aspect_monthly_stats(db_path, "res_demo", "2026-06")
    assert list_peer_benchmark(db_path, "res_demo", "2026-06")
    assert dashboard_payload(db_path, "res_demo", "2026-06")["priority"]


def test_run_monthly_is_idempotent_without_force(tmp_path: Path) -> None:
    db_path = tmp_path / "pipeline.duckdb"

    first = run_monthly_from_absa_jsonl(
        SAMPLE_PATH,
        restaurant_id="res_demo",
        review_month="2026-06",
        db_path=db_path,
        force=True,
    )
    second = run_monthly_from_absa_jsonl(
        SAMPLE_PATH,
        restaurant_id="res_demo",
        review_month="2026-06",
        db_path=db_path,
        force=False,
    )

    assert first["priority_run_id"] == second["priority_run_id"]
    assert second["status"] == "existing"
