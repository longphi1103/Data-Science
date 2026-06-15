from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from absa_recommender.aggregation import aggregate_aspect_stats
from absa_recommender.config import load_label_schema, load_yaml
from absa_recommender.normalize_absa import flatten_reviews
from absa_recommender.ranking import rank_priority_items
from absa_recommender.schemas import (
    ABSAReview,
    AspectExtraction,
    AspectStats,
    PeerSummary,
    PriorityItem,
    PriorityResponse,
    TrendSummary,
)
from absa_recommender.scoring import (
    compute_global_negative_rate_by_aspect,
    compute_priority_score,
    model_confidence,
    priority_confidence,
    risk_multiplier,
    scaled_benchmark_gap,
    smoothed_negative_rate,
    support_confidence,
)
from absa_recommender.severity import load_severity_config


DEFAULT_CONFIG_PATHS = {
    "label_schema": Path("configs/label_schema.yaml"),
    "scoring": Path("configs/scoring.yaml"),
    "severity": Path("configs/severity_lexicon.yaml"),
}


def generate_priority_ranking(
    reviews_or_extractions: list[ABSAReview] | list[AspectExtraction],
    top_n: int = 5,
    config_paths: dict[str, str | Path] | None = None,
    default_restaurant_id: str = "unknown",
    review_month: str | None = None,
    peer_benchmarks: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    previous_priority: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> PriorityResponse:
    configs = _load_configs(config_paths)
    extractions = _ensure_extractions(
        reviews_or_extractions,
        configs["label_schema"],
        configs["severity"],
        default_restaurant_id,
    )
    target_month = review_month or _response_review_month(extractions)
    if target_month != "multiple":
        extractions = [item for item in extractions if item.review_month == target_month]

    restaurant_id = _response_restaurant_id(extractions, default_restaurant_id)
    restaurant_name = _response_restaurant_name(extractions)
    if not extractions:
        return PriorityResponse(
            restaurant_id=restaurant_id,
            restaurant_name=restaurant_name,
            review_month=target_month,
            generated_at=_now(),
            top_n=top_n,
            items=[],
        )

    stats = aggregate_aspect_stats(extractions, configs["scoring"])
    global_negative_rates = compute_global_negative_rate_by_aspect(
        extractions,
        configs["label_schema"],
    )
    items = [
        _build_priority_item(
            stat,
            extractions,
            global_negative_rates,
            configs["scoring"],
            peer_benchmarks or {},
            previous_priority or {},
        )
        for stat in stats
        if stat.negative_count > 0
    ]
    scoring = configs["scoring"].get("scoring", configs["scoring"])
    ranking = scoring.get("ranking", {})
    ranked = rank_priority_items(
        items,
        top_n,
        force_food_safety_top3=bool(ranking.get("force_food_safety_top3", True)),
        food_safety_negative_threshold=float(
            ranking.get("food_safety_negative_threshold", 0.10)
        ),
    )
    return PriorityResponse(
        restaurant_id=restaurant_id,
        restaurant_name=restaurant_name,
        review_month=target_month,
        generated_at=_now(),
        top_n=top_n,
        items=ranked,
    )


def _load_configs(config_paths: dict[str, str | Path] | None) -> dict[str, Any]:
    paths = {**DEFAULT_CONFIG_PATHS, **(config_paths or {})}
    return {
        "label_schema": load_label_schema(paths["label_schema"]),
        "scoring": load_yaml(paths["scoring"]),
        "severity": load_severity_config(paths["severity"]),
    }


def _ensure_extractions(
    reviews_or_extractions: list[ABSAReview] | list[AspectExtraction],
    label_schema: dict[str, Any],
    severity_config: dict[str, Any],
    default_restaurant_id: str,
) -> list[AspectExtraction]:
    if not reviews_or_extractions:
        return []
    first = reviews_or_extractions[0]
    if isinstance(first, AspectExtraction):
        return list(reviews_or_extractions)
    return flatten_reviews(
        list(reviews_or_extractions),
        label_schema,
        default_restaurant_id=default_restaurant_id,
        severity_config=severity_config,
    )


def _build_priority_item(
    stats: AspectStats,
    extractions: list[AspectExtraction],
    global_negative_rates: dict[str, float],
    scoring_config: dict[str, Any],
    peer_benchmarks: dict[tuple[str, str, str], dict[str, Any]],
    previous_priority: dict[tuple[str, str], dict[str, Any]],
) -> PriorityItem:
    scoring = scoring_config.get("scoring", scoring_config)
    alpha = float(scoring.get("smoothing", {}).get("alpha", 10))
    neg_rate = smoothed_negative_rate(
        stats.negative_count,
        stats.mention_count,
        global_negative_rates.get(stats.aspect, 0.0),
        alpha,
    )
    stat = stats.model_copy(update={"negative_rate_smoothed": neg_rate})
    peer_summary, peer_confidence, benchmark = _peer_summary(stat, scoring, peer_benchmarks)
    trend_summary, history_confidence, trend_score = _trend_summary(stat, scoring, previous_priority)
    component_scores = {
        "negative_rate": neg_rate,
        "sentiment_severity": stat.avg_severity,
        "mention_share": stat.mention_share,
        "rating_gap": stat.rating_gap,
        "trend_score": trend_score,
        "benchmark_gap": benchmark,
    }
    priority_score = compute_priority_score(stat, component_scores, scoring_config)
    if trend_summary.previous_month_priority_score is not None:
        trend_summary = trend_summary.model_copy(
            update={
                "priority_delta": priority_score - trend_summary.previous_month_priority_score,
            }
        )
    confidence = priority_confidence(
        support_confidence(stat.mention_count, float(scoring.get("confidence", {}).get("support_threshold_tau", 30))),
        model_confidence(stat.avg_confidence),
        peer_confidence,
        history_confidence,
        scoring_config,
    )
    flags = _data_quality_flags(stat, scoring, peer_summary, trend_summary)
    return PriorityItem(
        rank=0,
        aspect=stat.aspect,
        priority_score=priority_score,
        priority_confidence=confidence,
        severity=stat.avg_severity,
        mention_count=stat.mention_count,
        negative_count=stat.negative_count,
        negative_rate_smoothed=neg_rate,
        mention_share=stat.mention_share,
        rating_gap=stat.rating_gap,
        trend_score=trend_score,
        benchmark_gap=benchmark,
        risk_multiplier=risk_multiplier(stat.aspect, scoring),
        opinion_examples=_opinion_examples(
            [
                item
                for item in extractions
                if item.restaurant_id == stat.restaurant_id
                and item.review_month == stat.review_month
                and item.aspect == stat.aspect
                and item.sentiment == "negative"
            ]
        ),
        component_scores=component_scores,
        peer_summary=peer_summary,
        trend_summary=trend_summary,
        data_quality_flags=flags,
    )


def _peer_summary(
    stats: AspectStats,
    scoring: dict[str, Any],
    peer_benchmarks: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[PeerSummary, float, float]:
    benchmark_config = scoring.get("benchmark", {})
    minimum_peers = int(benchmark_config.get("min_peer_restaurants", 5))
    minimum_mentions = int(benchmark_config.get("min_peer_mentions_per_aspect", 20))
    scale = float(benchmark_config.get("benchmark_scale", 0.30))
    peer = peer_benchmarks.get((stats.restaurant_id, stats.review_month, stats.aspect), {})
    peer_count = int(peer.get("peer_restaurant_count", 0))
    peer_mentions = int(peer.get("peer_total_mentions", 0))
    peer_rate = peer.get("peer_negative_rate")
    has_support = peer_count >= minimum_peers and peer_mentions >= minimum_mentions
    benchmark = scaled_benchmark_gap(
        stats.negative_rate_smoothed,
        float(peer_rate) if peer_rate is not None and has_support else None,
        scale,
    )
    peer_conf = min(1.0, peer_mentions / float(scoring.get("confidence", {}).get("peer_support_tau", 100)))
    return (
        PeerSummary(
            peer_restaurant_count=peer_count,
            peer_negative_rate=float(peer_rate or 0.0),
            target_vs_peer_gap=benchmark,
            peer_support_flag=None if has_support else "low_peer_support",
        ),
        peer_conf if has_support else 0.0,
        benchmark,
    )


def _trend_summary(
    stats: AspectStats,
    scoring: dict[str, Any],
    previous_priority: dict[tuple[str, str], dict[str, Any]],
) -> tuple[TrendSummary, float, float]:
    trend_config = scoring.get("trend", {})
    previous = previous_priority.get((stats.restaurant_id, stats.aspect), {})
    min_current = int(trend_config.get("min_mentions_current", 5))
    min_previous = int(trend_config.get("min_mentions_previous", 5))
    previous_mentions = int(previous.get("mention_count", 0))
    has_history = stats.mention_count >= min_current and previous_mentions >= min_previous
    if not has_history:
        return TrendSummary(trend_flag="insufficient_history"), 0.0, 0.0

    previous_negative_rate = float(previous.get("negative_rate_smoothed", 0.0))
    previous_severity = float(previous.get("severity", previous.get("avg_severity", 0.0)))
    negative_delta = stats.negative_rate_smoothed - previous_negative_rate
    severity_delta = stats.avg_severity - previous_severity
    negative_trend = max(
        0.0,
        min(1.0, negative_delta / float(trend_config.get("negative_scale", 0.25))),
    )
    severity_trend = max(
        0.0,
        min(1.0, severity_delta / float(trend_config.get("severity_scale", 0.30))),
    )
    trend_score = 0.7 * negative_trend + 0.3 * severity_trend
    previous_score = previous.get("priority_score")
    return (
        TrendSummary(
            previous_month_priority_score=float(previous_score) if previous_score is not None else None,
            priority_delta=None,
            negative_rate_delta=negative_delta,
        ),
        1.0,
        trend_score,
    )


def _data_quality_flags(
    stats: AspectStats,
    scoring: dict[str, Any],
    peer_summary: PeerSummary,
    trend_summary: TrendSummary,
) -> list[str]:
    flags: list[str] = []
    if stats.mention_count < int(scoring.get("trend", {}).get("min_mentions_current", 5)):
        flags.append("low_mentions")
    if peer_summary.peer_support_flag:
        flags.append(peer_summary.peer_support_flag)
    if trend_summary.trend_flag:
        flags.append(trend_summary.trend_flag)
    if stats.avg_confidence < 0.5:
        flags.append("low_model_confidence")
    if stats.window_start is None or stats.window_end is None:
        flags.append("missing_review_time")
    return flags


def _opinion_examples(extractions: list[AspectExtraction], limit: int = 3) -> list[str]:
    examples: list[str] = []
    for extraction in extractions:
        if extraction.opinion_text not in examples:
            examples.append(extraction.opinion_text)
        if len(examples) >= limit:
            break
    return examples


def _response_restaurant_id(
    extractions: list[AspectExtraction],
    default_restaurant_id: str,
) -> str:
    restaurant_ids = sorted({item.restaurant_id for item in extractions})
    if not restaurant_ids:
        return default_restaurant_id
    if len(restaurant_ids) == 1:
        return restaurant_ids[0]
    return "multiple"


def _response_restaurant_name(extractions: list[AspectExtraction]) -> str | None:
    names = sorted({item.restaurant_name for item in extractions if item.restaurant_name})
    if len(names) == 1:
        return names[0]
    return None


def _response_review_month(extractions: list[AspectExtraction]) -> str:
    months = sorted({item.review_month for item in extractions})
    if not months:
        return "unknown"
    if len(months) == 1:
        return months[0]
    return "multiple"


def _now() -> datetime:
    return datetime.now(timezone.utc)
