from typing import Any


def compute_trend_score(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    negative_scale: float = 0.25,
    severity_scale: float = 0.30,
    min_mentions_current: int = 5,
    min_mentions_previous: int = 5,
) -> dict[str, Any]:
    if (
        previous is None
        or int(current.get("mention_count", 0)) < min_mentions_current
        or int(previous.get("mention_count", 0)) < min_mentions_previous
    ):
        return {"trend_score": 0.0, "trend_flag": "insufficient_history"}
    negative_delta = float(current.get("negative_rate_smoothed", 0.0)) - float(
        previous.get("negative_rate_smoothed", 0.0)
    )
    severity_delta = float(current.get("avg_severity", current.get("severity", 0.0))) - float(
        previous.get("avg_severity", previous.get("severity", 0.0))
    )
    score = 0.7 * _scaled_positive_delta(negative_delta, negative_scale) + 0.3 * (
        _scaled_positive_delta(severity_delta, severity_scale)
    )
    return {
        "trend_score": score,
        "trend_flag": None,
        "negative_rate_delta": negative_delta,
        "severity_delta": severity_delta,
    }


def _scaled_positive_delta(delta: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, delta / scale))
