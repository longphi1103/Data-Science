from pathlib import Path

from absa_recommender.aggregation import aggregate_aspect_stats
from absa_recommender.config import load_label_schema, load_yaml
from absa_recommender.normalize_absa import flatten_reviews, load_absa_jsonl
from absa_recommender.schemas import AspectExtraction


LABEL_SCHEMA = load_label_schema(Path("configs/label_schema.yaml"))
SCORING_CONFIG = load_yaml(Path("configs/scoring.yaml"))


def test_grouping_works_for_food_quality_cleanliness_location_menu() -> None:
    reviews = load_absa_jsonl(Path("data/samples/absa_outputs.jsonl"))
    extractions = flatten_reviews(reviews, LABEL_SCHEMA)

    stats = aggregate_aspect_stats(extractions, SCORING_CONFIG)
    stats_by_key = {(item.restaurant_id, item.review_month, item.aspect): item for item in stats}

    assert ("rest_001", "2026-05", "Food Quality") in stats_by_key
    assert ("rest_002", "2026-05", "Location") in stats_by_key
    assert ("rest_003", "2026-06", "Cleanliness") in stats_by_key
    assert ("rest_003", "2026-06", "Menu") in stats_by_key
    assert stats_by_key[("rest_002", "2026-05", "Location")].mention_count == 2
    assert stats_by_key[("rest_002", "2026-05", "Location")].negative_count == 2
    assert stats_by_key[("rest_002", "2026-05", "Location")].total_mentions_for_restaurant == 3


def test_missing_rating_uses_default() -> None:
    extraction = _extraction(rating=None, model_confidence=0.5)

    stats = aggregate_aspect_stats([extraction], SCORING_CONFIG)

    assert stats[0].avg_rating == 3.5


def test_missing_confidence_uses_default() -> None:
    extraction = _extraction(rating=4, model_confidence=None)

    stats = aggregate_aspect_stats([extraction], SCORING_CONFIG)

    assert stats[0].avg_confidence == 0.75


def _extraction(
    rating: int | None,
    model_confidence: float | None,
) -> AspectExtraction:
    return AspectExtraction(
        extraction_id="rv_missing_0",
        review_id="rv_missing",
        restaurant_id="rest_missing",
        restaurant_name=None,
        aspect="Menu",
        aspect_term="menu",
        opinion_text="menu khó hiểu",
        sentiment="negative",
        severity=0.75,
        model_confidence=model_confidence,
        review_text="Menu khó hiểu.",
        rating=rating,
        review_time=None,
        review_month="2026-06",
    )
