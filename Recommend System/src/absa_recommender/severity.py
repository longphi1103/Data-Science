from pathlib import Path
from typing import Any

from absa_recommender.config import load_yaml


DEFAULT_SEVERITY_CONFIG_PATH = Path("configs/severity_lexicon.yaml")


def load_severity_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def compute_severity(
    sentiment: str,
    opinion_text: str,
    aspect: str | None = None,
    config: dict[str, Any] | None = None,
) -> float:
    severity_config = config or load_severity_config(DEFAULT_SEVERITY_CONFIG_PATH)
    normalized_sentiment = sentiment.casefold()
    normalized_text = opinion_text.casefold()

    score = float(severity_config.get("base_scores", {}).get(normalized_sentiment, 0.0))
    has_strong_pattern = _contains_any(
        normalized_text,
        severity_config.get("strong_negative_patterns", []),
    )
    has_safety_pattern = _contains_any(
        normalized_text,
        severity_config.get("safety_patterns", []),
    )
    has_mild_pattern = _contains_any(
        normalized_text,
        severity_config.get("mild_negative_patterns", []),
    )

    if normalized_sentiment == "negative" and has_mild_pattern and not has_strong_pattern:
        score = 0.6

    if has_strong_pattern:
        score = max(score, 0.9)

    if has_safety_pattern or aspect == "Food Safety":
        score = max(score, 0.95)

    return _clamp(score)


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern.casefold() in text for pattern in patterns)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
