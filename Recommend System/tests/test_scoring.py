from datetime import datetime
from pathlib import Path

from absa_recommender.config import load_label_schema
from absa_recommender.schemas import AspectExtraction, AspectStats
from absa_recommender.scoring import (
    benchmark_gap,
    combined_confidence,
    compute_global_negative_rate_by_aspect,
    compute_priority_score,
    load_scoring_config,
    log_mention_share,
    model_confidence,
    normalized_rating_gap,
    priority_confidence,
    smoothed_negative_rate,
    scaled_benchmark_gap,
    support_confidence,
)


SCORING_CONFIG = load_scoring_config(Path("configs/scoring.yaml"))
LABEL_SCHEMA = load_label_schema(Path("configs/label_schema.yaml"))
OFFICIAL_NON_UNKNOWN_ASPECTS = [
    "Food Quality",
    "Food Safety",
    "Service",
    "Price",
    "Cleanliness",
    "Ambience",
    "Location",
    "Menu",
]


def test_all_8_aspects_can_be_scored() -> None:
    component_scores = _component_scores()

    scores = [
        compute_priority_score(_stats(aspect=aspect), component_scores, SCORING_CONFIG)
        for aspect in OFFICIAL_NON_UNKNOWN_ASPECTS
    ]

    assert len(scores) == 8
    assert all(0.0 <= score <= 100.0 for score in scores)


def test_final_scores_are_within_0_100() -> None:
    stats = _stats(aspect="Food Safety")
    component_scores = {
        "negative_rate": 1.5,
        "sentiment_severity": 1.5,
        "mention_share": 1.5,
        "rating_gap": 1.5,
        "trend_score": 1.5,
        "benchmark_gap": 1.5,
    }

    score = compute_priority_score(stats, component_scores, SCORING_CONFIG)

    assert 0.0 <= score <= 100.0


def test_low_mention_count_lowers_confidence() -> None:
    low = support_confidence(mention_count=1, tau=30)
    high = support_confidence(mention_count=100, tau=30)

    assert low < high
    assert combined_confidence(low, model_confidence(0.8), 0.7) < combined_confidence(
        high,
        model_confidence(0.8),
        0.7,
    )


def test_food_safety_risk_multiplier_increases_score() -> None:
    component_scores = _component_scores()

    food_quality_score = compute_priority_score(
        _stats(aspect="Food Quality"),
        component_scores,
        SCORING_CONFIG,
    )
    food_safety_score = compute_priority_score(
        _stats(aspect="Food Safety"),
        component_scores,
        SCORING_CONFIG,
    )

    assert food_safety_score > food_quality_score


def test_location_uses_0_95_multiplier_from_config() -> None:
    component_scores = _component_scores()

    menu_score = compute_priority_score(_stats(aspect="Menu"), component_scores, SCORING_CONFIG)
    location_score = compute_priority_score(
        _stats(aspect="Location"),
        component_scores,
        SCORING_CONFIG,
    )

    assert location_score == round(menu_score * 0.95, 4)


def test_menu_uses_1_00_multiplier_from_config() -> None:
    component_scores = _component_scores()

    food_quality_score = compute_priority_score(
        _stats(aspect="Food Quality"),
        component_scores,
        SCORING_CONFIG,
    )
    menu_score = compute_priority_score(_stats(aspect="Menu"), component_scores, SCORING_CONFIG)

    assert menu_score == food_quality_score


def test_unknown_uses_0_70_multiplier() -> None:
    component_scores = _component_scores()

    menu_score = compute_priority_score(_stats(aspect="Menu"), component_scores, SCORING_CONFIG)
    unknown_score = compute_priority_score(
        _stats(aspect="Unknown"),
        component_scores,
        SCORING_CONFIG,
    )

    assert unknown_score == round(menu_score * 0.70, 4)


def test_component_scores_are_clamped_to_0_1() -> None:
    assert smoothed_negative_rate(negative_count=10, mention_count=1, global_mu=2.0, alpha=10) == 1.0
    assert log_mention_share(mention_count=5, total_mentions=3) == 1.0
    assert normalized_rating_gap(avg_rating=0.0) == 1.0
    assert model_confidence(avg_confidence=2.0) == 1.0
    assert benchmark_gap(neg_rate=0.8, peer_avg_neg_rate=None) == 0.0
    assert scaled_benchmark_gap(0.6, 0.3, 0.3) == 1.0


def test_priority_confidence_uses_support_model_peer_and_history() -> None:
    confidence = priority_confidence(1.0, 0.8, 0.5, 1.0, SCORING_CONFIG)

    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.7


def test_compute_global_negative_rate_by_aspect() -> None:
    extractions = [
        _extraction("Food Quality", "negative"),
        _extraction("Food Quality", "positive"),
        _extraction("Menu", "negative"),
    ]

    rates = compute_global_negative_rate_by_aspect(extractions, LABEL_SCHEMA)

    assert rates["Food Quality"] == 0.5
    assert rates["Menu"] == 1.0
    assert rates["Location"] == 0.0


def _component_scores() -> dict[str, float]:
    return {
        "negative_rate": 0.5,
        "sentiment_severity": 0.75,
        "mention_share": 0.4,
        "rating_gap": 0.5,
        "trend_score": 0.0,
        "benchmark_gap": 0.0,
    }


def _stats(aspect: str) -> AspectStats:
    return AspectStats(
        restaurant_id="rest_001",
        review_month="2026-06",
        aspect=aspect,
        mention_count=10,
        negative_count=5,
        positive_count=3,
        neutral_count=2,
        negative_rate_raw=0.5,
        negative_rate_smoothed=0.5,
        avg_severity=0.75,
        avg_rating=3.0,
        avg_confidence=0.8,
        mention_share=0.4,
        rating_gap=0.5,
        total_mentions_for_restaurant=20,
        window_start=datetime(2026, 5, 1),
        window_end=datetime(2026, 6, 1),
    )


def _extraction(aspect: str, sentiment: str) -> AspectExtraction:
    return AspectExtraction(
        extraction_id=f"{aspect}_{sentiment}",
        review_id="rv_001",
        restaurant_id="rest_001",
        restaurant_name=None,
        aspect=aspect,
        aspect_term="term",
        opinion_text="text",
        sentiment=sentiment,
        severity=0.75,
        model_confidence=0.8,
        review_text="review",
        rating=3,
        review_time=None,
        review_month="2026-06",
    )
