import json
from pathlib import Path

from absa_recommender.absa_inference import PlaceholderABSAAdapter
from absa_recommender.crawler.monthly import (
    CrawlStrategy,
    crawl_reviews_for_month,
    persist_crawl_result,
)
from absa_recommender.monthly_pipeline import run_monthly_from_source
from absa_recommender.sources.local_jsonl_adapter import LocalJsonlAdapter
from absa_recommender.storage import dashboard_payload, list_restaurants, list_review_months


def test_placeholder_absa_adapter_generates_pluggable_annotations() -> None:
    adapter = PlaceholderABSAAdapter()

    reviews = adapter.infer(
        [
            {
                "review_id": "raw_1",
                "review_text": "Nhân viên phục vụ chậm, bàn hơi bẩn nhưng món ăn ổn.",
                "restaurant_id": "res_demo",
                "rating": 2,
                "review_month": "2026-06",
            }
        ]
    )

    assert adapter.model_version == "placeholder-rule-absa-v0"
    assert reviews[0].annotations
    assert {item.aspect_category for item in reviews[0].annotations} >= {
        "Service",
        "Cleanliness",
        "Food Quality",
    }
    assert all(item.model_confidence is not None for item in reviews[0].annotations)


def test_local_jsonl_crawler_normalizes_dedups_and_persists(tmp_path: Path) -> None:
    source_path = tmp_path / "raw.jsonl"
    rows = [
        {
            "review_id": "r1",
            "review_text": "Nhân viên phục vụ chậm.",
            "restaurant_id": "res_demo",
            "rating": 2,
            "review_month": "2026-06",
        },
        {
            "review_id": "r1_dup",
            "source_review_id": "r1",
            "review_text": "Nhân viên phục vụ chậm.",
            "restaurant_id": "res_demo",
            "rating": 2,
            "review_month": "2026-06",
        },
        {
            "review_id": "p1",
            "review_text": "Món ăn ngon.",
            "restaurant_id": "res_peer",
            "rating": 5,
            "review_month": "2026-06",
        },
    ]
    source_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    result = crawl_reviews_for_month(
        LocalJsonlAdapter(source_path),
        restaurant_id="res_demo",
        month="2026-06",
        strategy=CrawlStrategy(max_reviews_per_restaurant=10),
    )
    db_path = tmp_path / "crawl.duckdb"
    crawl_run_id = persist_crawl_result(db_path, result)

    assert crawl_run_id.startswith("crawl_")
    assert len(result.reviews) == 2
    assert len(result.duplicate_reviews) == 1
    assert {row["restaurant_id"] for row in list_restaurants(db_path)} == {"res_demo", "res_peer"}


def test_run_monthly_from_source_connects_crawler_absa_and_priority(tmp_path: Path) -> None:
    source_path = tmp_path / "raw.jsonl"
    rows = [
        {
            "review_id": "target_1",
            "review_text": "Nhân viên phục vụ chậm và bàn bẩn.",
            "restaurant_id": "res_demo",
            "restaurant_name": "Demo",
            "rating": 2,
            "review_month": "2026-06",
        },
        {
            "review_id": "target_2",
            "review_text": "Món ăn không ngon, giá đắt.",
            "restaurant_id": "res_demo",
            "restaurant_name": "Demo",
            "rating": 2,
            "review_month": "2026-06",
        },
        {
            "review_id": "peer_1",
            "review_text": "Món ăn ngon và phục vụ nhanh.",
            "restaurant_id": "res_peer",
            "restaurant_name": "Peer",
            "rating": 5,
            "review_month": "2026-06",
        },
    ]
    source_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    db_path = tmp_path / "full.duckdb"

    result = run_monthly_from_source(
        source_path,
        restaurant_id="res_demo",
        review_month="2026-06",
        db_path=db_path,
        force=True,
        top_n=5,
    )

    assert result["status"] == "completed"
    assert result["absa_adapter"] == "placeholder-rule-absa-v0"
    assert result["output"]["items"]
    assert list_review_months(db_path, "res_demo") == ["2026-06"]
    assert dashboard_payload(db_path, "res_demo", "2026-06")["priority"]
