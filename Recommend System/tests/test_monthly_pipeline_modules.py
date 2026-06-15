from datetime import date

import pytest

from absa_recommender.absa_inference import infer_absa_with_adapter
from absa_recommender.benchmark import compute_peer_aspect_stats
from absa_recommender.crawler import build_crawl_run
from absa_recommender.dedup import deduplicate_reviews
from absa_recommender.peer_discovery import discover_peer_restaurants
from absa_recommender.priority import generate_priority_ranking
from absa_recommender.ranking import rank_priority_items
from absa_recommender.review_normalizer import normalize_review
from absa_recommender.schemas import PeerSummary, PriorityItem, TrendSummary
from absa_recommender.scheduler import previous_month_for_run, priority_idempotency_key
from absa_recommender.sources.google_maps_adapter import GoogleMapsAdapter
from absa_recommender.sources.local_jsonl_adapter import LocalJsonlAdapter
from absa_recommender.trend import compute_trend_score


def test_normalize_review_and_dedup() -> None:
    review = normalize_review(
        {
            "review_id": "rv_1",
            "restaurant_id": "res_1",
            "text": " Ban hoi ban ",
            "rating": 2,
            "review_time": "2026-06-10T00:00:00",
        }
    )
    unique, duplicates = deduplicate_reviews([review, review])

    assert review["review_month"] == "2026-06"
    assert review["review_text_hash"]
    assert len(unique) == 1
    assert len(duplicates) == 1


def test_discover_peer_restaurants_filters_target_and_radius() -> None:
    target = {"restaurant_id": "target", "lat": 10.0, "lng": 106.0}
    peers = discover_peer_restaurants(
        target,
        [
            {"restaurant_id": "target", "lat": 10.0, "lng": 106.0, "types": ["restaurant"]},
            {"restaurant_id": "peer", "lat": 10.001, "lng": 106.001, "types": ["restaurant"]},
            {"restaurant_id": "far", "lat": 11.0, "lng": 107.0, "types": ["restaurant"]},
        ],
        radius_meters=500,
        max_peers=10,
    )

    assert [peer["restaurant_id"] for peer in peers] == ["peer"]


def test_benchmark_and_trend_helpers() -> None:
    peer = compute_peer_aspect_stats(
        "target",
        "area_1",
        "2026-06",
        "Service",
        [
            {
                "restaurant_id": "peer_1",
                "review_month": "2026-06",
                "aspect": "Service",
                "mention_count": 10,
                "negative_rate_smoothed": 0.2,
                "avg_severity": 0.4,
                "avg_rating": 4.0,
            }
        ],
    )
    trend = compute_trend_score(
        {"mention_count": 10, "negative_rate_smoothed": 0.5, "avg_severity": 0.8},
        {"mention_count": 10, "negative_rate_smoothed": 0.25, "avg_severity": 0.5},
    )

    assert peer["peer_restaurant_count"] == 1
    assert peer["peer_negative_rate"] == 0.2
    assert trend["trend_score"] > 0


def test_scheduler_and_crawler_helpers() -> None:
    crawl_run = build_crawl_run("local", "2026-06", "area_1", [{"restaurant_id": "res_1"}])

    assert previous_month_for_run(date(2026, 7, 3)) == "2026-06"
    assert priority_idempotency_key("res_1", "2026-06", "hash", "v1") == (
        "res_1",
        "2026-06",
        "hash",
        "v1",
    )
    assert crawl_run["num_restaurants"] == 1
    assert crawl_run["status"] == "created"


def test_source_and_inference_adapters_are_explicit() -> None:
    with pytest.raises(RuntimeError):
        GoogleMapsAdapter().fetch_reviews()
    reviews = infer_absa_with_adapter(
        [
            {
                "review_id": "rv_1",
                "review_text": "bad service",
                "annotations": [
                    {
                        "aspect_expression": "service",
                        "aspect_category": "Service",
                        "opinion_expression": "bad",
                        "sentiment": "negative",
                    }
                ],
            }
        ]
    )
    assert reviews[0].review_id == "rv_1"
    local_rows = LocalJsonlAdapter("data/samples/streamlit_priority_200.jsonl").fetch_reviews(
        restaurant_id="res_demo",
        month="2026-06",
    )
    assert local_rows


def test_ranking_helper_and_priority_alias() -> None:
    items = [
        _priority_item("Service", 90, 0.5),
        _priority_item("Food Safety", 10, 0.2),
        _priority_item("Menu", 80, 0.5),
        _priority_item("Cleanliness", 70, 0.5),
    ]

    ranked = rank_priority_items(items, top_n=3)

    assert generate_priority_ranking
    assert [item.aspect for item in ranked] == ["Service", "Menu", "Food Safety"]
    assert [item.rank for item in ranked] == [1, 2, 3]


def _priority_item(aspect: str, score: float, negative_rate: float) -> PriorityItem:
    return PriorityItem(
        rank=0,
        aspect=aspect,
        priority_score=score,
        priority_confidence=0.8,
        severity=0.7,
        mention_count=10,
        negative_count=5,
        negative_rate_smoothed=negative_rate,
        mention_share=0.5,
        rating_gap=0.5,
        trend_score=0.0,
        benchmark_gap=0.0,
        risk_multiplier=1.0,
        opinion_examples=[],
        component_scores={},
        peer_summary=PeerSummary(),
        trend_summary=TrendSummary(),
        data_quality_flags=[],
    )
