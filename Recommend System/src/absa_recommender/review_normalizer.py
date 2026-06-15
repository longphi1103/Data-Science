import hashlib
from datetime import datetime
from typing import Any

from absa_recommender.text_normalizer import normalize_review_text


def normalize_review(raw: dict[str, Any], default_source: str = "local") -> dict[str, Any]:
    review_time = raw.get("review_time")
    review_month = raw.get("review_month") or _review_month(review_time)
    raw_text = str(raw.get("review_text") or raw.get("text") or "")
    text = normalize_review_text(raw_text)
    return {
        "source": raw.get("source", default_source),
        "source_place_id": raw.get("source_place_id"),
        "source_review_id": raw.get("source_review_id") or raw.get("review_id"),
        "restaurant_id": raw.get("restaurant_id", "unknown"),
        "review_text": text,
        "review_text_hash": review_text_hash(text),
        "rating": raw.get("rating"),
        "review_time": review_time,
        "review_month": review_month,
        "language": raw.get("language", "vi"),
        "fetched_at": raw.get("fetched_at"),
    }


def review_text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def _review_month(review_time: Any) -> str:
    if review_time is None:
        return "unknown"
    if isinstance(review_time, datetime):
        return review_time.strftime("%Y-%m")
    return str(review_time)[:7]
