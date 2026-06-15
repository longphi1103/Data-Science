from statistics import median
from typing import Any


def compute_peer_aspect_stats(
    target_restaurant_id: str,
    area_id: str,
    review_month: str,
    aspect: str,
    peer_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    peers = [
        row
        for row in peer_stats
        if row.get("restaurant_id") != target_restaurant_id
        and row.get("review_month") == review_month
        and row.get("aspect") == aspect
    ]
    rates = [float(row.get("negative_rate_smoothed", 0.0)) for row in peers]
    return {
        "area_id": area_id,
        "target_restaurant_id": target_restaurant_id,
        "review_month": review_month,
        "aspect": aspect,
        "peer_restaurant_count": len(peers),
        "peer_total_mentions": sum(int(row.get("mention_count", 0)) for row in peers),
        "peer_negative_rate": _mean(rates),
        "peer_avg_severity": _mean([float(row.get("avg_severity", 0.0)) for row in peers]),
        "peer_avg_rating": _mean([float(row.get("avg_rating", 0.0)) for row in peers]),
        "peer_p50_negative_rate": median(rates) if rates else 0.0,
        "peer_p75_negative_rate": _percentile(rates, 0.75),
        "peer_p90_negative_rate": _percentile(rates, 0.90),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
    return ordered[index]
