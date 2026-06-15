from pathlib import Path

from absa_recommender.api import (
    aspect_history,
    health,
    infer_absa,
    labels,
    monthly_run,
    monthly_run_raw,
    peer_benchmark,
    priority_from_absa,
    restaurant_dashboard,
    restaurant_history,
    restaurant_priority,
)
from absa_recommender.normalize_absa import load_absa_jsonl


SAMPLE_PATH = Path("data/samples/absa_outputs.jsonl")


def test_health() -> None:
    assert health() == {"status": "ok"}


def test_labels_include_official_aspects() -> None:
    payload = labels()

    assert "Food Quality" in payload["aspects"]
    assert "Menu" in payload["aspects"]


def test_priority_from_absa_returns_items() -> None:
    response = priority_from_absa(load_absa_jsonl(SAMPLE_PATH), top_n=5, restaurant_id="res_demo")

    assert response.items
    assert response.items[0].rank == 1


def test_infer_absa_uses_preannotated_adapter() -> None:
    payload = infer_absa([review.model_dump(mode="json") for review in load_absa_jsonl(SAMPLE_PATH)])

    assert payload["status"] == "completed"
    assert payload["adapter"] == "preannotated-jsonl"
    assert payload["review_count"] == 3
    assert payload["annotation_count"] == 7


def test_infer_absa_can_use_placeholder_adapter() -> None:
    payload = infer_absa(
        [
            {
                "review_id": "raw_1",
                "review_text": "Nhân viên phục vụ chậm.",
                "restaurant_id": "res_demo",
                "rating": 2,
                "review_month": "2026-06",
            }
        ],
        adapter_name="placeholder",
    )

    assert payload["status"] == "completed"
    assert payload["adapter"] == "placeholder-rule-absa-v0"
    assert payload["annotation_count"] >= 1


def test_monthly_run_raw_uses_placeholder_adapter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ABSA_DB_PATH", str(tmp_path / "api_raw.duckdb"))

    result = monthly_run_raw(
        [
            {
                "review_id": "raw_1",
                "review_text": "Nhân viên phục vụ chậm và bàn bẩn.",
                "restaurant_id": "res_demo",
                "rating": 2,
                "review_month": "2026-06",
            },
            {
                "review_id": "raw_2",
                "review_text": "Món ăn ngon.",
                "restaurant_id": "res_peer",
                "rating": 5,
                "review_month": "2026-06",
            },
        ],
        restaurant_id="res_demo",
        month="2026-06",
        force=True,
        absa_adapter="placeholder",
    )

    assert result["status"] == "completed"
    assert result["absa_adapter"] == "placeholder-rule-absa-v0"
    assert result["output"]["items"]


def test_monthly_and_dashboard_routes_have_priority_shape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ABSA_DB_PATH", str(tmp_path / "api.duckdb"))

    monthly = monthly_run(
        load_absa_jsonl(SAMPLE_PATH),
        top_n=3,
        restaurant_id="res_demo",
        month="2026-06",
        force=True,
    )

    assert monthly["status"] == "completed"
    assert monthly["output"]["items"]
    assert restaurant_priority("res_demo", "2026-06")["items"]
    assert restaurant_dashboard("res_demo", "2026-06")["overview"]
    assert restaurant_history("res_demo")["runs"]
    assert aspect_history("res_demo", "Cleanliness")["history"]
    assert "items" in peer_benchmark("res_demo", "2026-06")
