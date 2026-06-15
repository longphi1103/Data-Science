from typing import Any


def deduplicate_reviews(reviews: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for review in reviews:
        key = dedup_key(review)
        if key in seen:
            duplicates.append({**review, "is_duplicate": True})
            continue
        seen.add(key)
        unique.append({**review, "is_duplicate": False})
    return unique, duplicates


def dedup_key(review: dict[str, Any]) -> tuple[Any, ...]:
    if review.get("source_review_id"):
        return ("source_review_id", review.get("source"), review.get("source_review_id"))
    return (
        "content",
        review.get("source_place_id"),
        review.get("review_text_hash"),
        review.get("rating"),
        review.get("review_time"),
    )
