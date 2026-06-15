import math
from pathlib import Path
from typing import Any

from absa_recommender.config import load_yaml
from absa_recommender.schemas import AspectExtraction, AspectStats


def load_scoring_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def compute_global_negative_rate_by_aspect(
    extractions: list[AspectExtraction],
    label_schema: dict[str, Any],
) -> dict[str, float]:
    aspects = _schema_aspects(label_schema)
    counts = {aspect: {"mentions": 0, "negative": 0} for aspect in aspects}
    for extraction in extractions:
        counts.setdefault(extraction.aspect, {"mentions": 0, "negative": 0})
        counts[extraction.aspect]["mentions"] += 1
        if extraction.sentiment == "negative":
            counts[extraction.aspect]["negative"] += 1

    return {
        aspect: _safe_divide(values["negative"], values["mentions"])
        for aspect, values in counts.items()
    }


def smoothed_negative_rate(
    negative_count: int,
    mention_count: int,
    global_mu: float,
    alpha: float,
) -> float:
    if mention_count + alpha <= 0:
        return _clamp(global_mu)
    return _clamp((negative_count + alpha * global_mu) / (mention_count + alpha))


def log_mention_share(mention_count: int, total_mentions: int) -> float:
    if mention_count <= 0 or total_mentions <= 0:
        return 0.0
    return _clamp(math.log1p(mention_count) / math.log1p(total_mentions))


def normalized_rating_gap(avg_rating: float) -> float:
    return _clamp((5.0 - avg_rating) / 4.0)


def support_confidence(mention_count: int, tau: float) -> float:
    if mention_count <= 0 or tau <= 0:
        return 0.0
    return _clamp(1.0 - math.exp(-mention_count / tau))


def model_confidence(avg_confidence: float) -> float:
    return _clamp(avg_confidence)


def combined_confidence(
    support_conf: float,
    model_conf: float,
    lambda_support: float,
) -> float:
    support_weight = _clamp(lambda_support)
    return _clamp(support_weight * support_conf + (1.0 - support_weight) * model_conf)


def priority_confidence(
    support_conf: float,
    model_conf: float,
    peer_conf: float,
    history_conf: float,
    config: dict[str, Any],
) -> float:
    scoring = _scoring_section(config)
    weights = scoring.get("confidence", {}).get(
        "weights",
        {"support": 0.45, "model": 0.30, "peer": 0.15, "history": 0.10},
    )
    return _clamp(
        float(weights.get("support", 0.45)) * _clamp(support_conf)
        + float(weights.get("model", 0.30)) * _clamp(model_conf)
        + float(weights.get("peer", 0.15)) * _clamp(peer_conf)
        + float(weights.get("history", 0.10)) * _clamp(history_conf)
    )


def benchmark_gap(neg_rate: float, peer_avg_neg_rate: float | None) -> float:
    if peer_avg_neg_rate is None:
        return 0.0
    return _clamp(neg_rate - peer_avg_neg_rate)


def scaled_benchmark_gap(
    target_negative_rate: float,
    peer_negative_rate: float | None,
    benchmark_scale: float,
) -> float:
    if peer_negative_rate is None or benchmark_scale <= 0:
        return 0.0
    return _clamp((target_negative_rate - peer_negative_rate) / benchmark_scale)


def compute_priority_score(
    stats: AspectStats,
    component_scores: dict[str, float],
    config: dict[str, Any],
) -> float:
    scoring = _scoring_section(config)
    weights = scoring.get("weights", {})
    weighted_score = sum(
        float(weight) * _clamp(component_scores.get(component_name, 0.0))
        for component_name, weight in weights.items()
    )
    multiplier = risk_multiplier(stats.aspect, scoring)
    return round(_clamp(weighted_score * multiplier) * 100.0, 4)


def _schema_aspects(label_schema: dict[str, Any]) -> list[str]:
    aspects = label_schema.get("aspects", [])
    if isinstance(aspects, dict):
        return list(aspects.get("labels", []))
    return list(aspects)


def risk_multiplier(aspect: str, scoring: dict[str, Any]) -> float:
    default = float(scoring.get("defaults", {}).get("risk_multiplier_if_missing", 1.0))
    return float(scoring.get("risk_multiplier", {}).get(aspect, default))


def _scoring_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("scoring", config)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _clamp(numerator / denominator)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
