from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from absa_recommender.config import load_yaml
from absa_recommender.dedup import deduplicate_reviews
from absa_recommender.review_normalizer import normalize_review
from absa_recommender.storage import save_crawl_run, save_restaurants, save_reviews


@dataclass(frozen=True)
class CrawlStrategy:
    max_reviews_per_restaurant: int = 200
    max_restaurants_per_run: int = 50
    min_delay_seconds: float = 1.0
    max_delay_seconds: float = 3.0
    max_attempts: int = 3
    backoff_seconds: float = 5.0
    respect_source_policy: bool = True


@dataclass(frozen=True)
class CrawlResult:
    source: str
    target_restaurant_id: str
    target_month: str
    area_id: str
    restaurants: list[dict[str, Any]]
    reviews: list[dict[str, Any]]
    duplicate_reviews: list[dict[str, Any]]
    strategy: CrawlStrategy


def build_crawl_run(
    source: str,
    target_month: str,
    area_id: str,
    restaurants: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": source,
        "target_month": target_month,
        "area_id": area_id,
        "started_at": datetime.now(timezone.utc),
        "finished_at": None,
        "status": "created",
        "num_restaurants": len(restaurants),
        "num_reviews_fetched": 0,
        "num_reviews_inserted": 0,
        "num_duplicates": 0,
        "error_message": None,
    }


def load_crawl_strategy(path: str = "configs/crawler.yaml") -> CrawlStrategy:
    config = load_yaml(path)
    retry = config.get("retry", {})
    politeness = config.get("politeness", {})
    return CrawlStrategy(
        max_reviews_per_restaurant=int(config.get("max_reviews_per_restaurant", 200)),
        max_restaurants_per_run=int(config.get("max_restaurants_per_run", 50)),
        min_delay_seconds=float(politeness.get("min_delay_seconds", 1.0)),
        max_delay_seconds=float(politeness.get("max_delay_seconds", 3.0)),
        max_attempts=int(retry.get("max_attempts", 3)),
        backoff_seconds=float(retry.get("backoff_seconds", 5.0)),
        respect_source_policy=bool(config.get("respect_source_policy", True)),
    )


def crawl_reviews_for_month(
    adapter: Any,
    restaurant_id: str,
    month: str,
    peer_restaurant_ids: list[str] | None = None,
    area_id: str = "local",
    strategy: CrawlStrategy | None = None,
) -> CrawlResult:
    strategy = strategy or load_crawl_strategy()
    raw_rows = _fetch_source_rows(adapter, restaurant_id, month, peer_restaurant_ids)
    selected = _cap_reviews_per_restaurant(raw_rows, strategy.max_reviews_per_restaurant)
    normalized = [
        _normalized_review(row, default_source=getattr(adapter, "source", "local_jsonl"))
        for row in selected
    ]
    unique_reviews, duplicate_reviews = deduplicate_reviews(normalized)
    restaurant_rows = _restaurant_rows(
        unique_reviews,
        target_restaurant_id=restaurant_id,
        area_id=area_id,
        source=getattr(adapter, "source", "local_jsonl"),
    )
    return CrawlResult(
        source=getattr(adapter, "source", "local_jsonl"),
        target_restaurant_id=restaurant_id,
        target_month=month,
        area_id=area_id,
        restaurants=restaurant_rows[: strategy.max_restaurants_per_run],
        reviews=unique_reviews,
        duplicate_reviews=duplicate_reviews,
        strategy=strategy,
    )


def persist_crawl_result(db_path: str | Path, result: CrawlResult, status: str = "success") -> str:
    crawl_run_id = save_crawl_run(
        db_path,
        source=result.source,
        target_month=result.target_month,
        area_id=result.area_id,
        status=status,
        num_restaurants=len(result.restaurants),
        num_reviews_fetched=len(result.reviews) + len(result.duplicate_reviews),
        num_reviews_inserted=len(result.reviews),
        num_duplicates=len(result.duplicate_reviews),
    )
    reviews = [dict(review, crawl_run_id=crawl_run_id) for review in result.reviews]
    save_restaurants(db_path, result.restaurants)
    save_reviews(db_path, reviews)
    return crawl_run_id


def _fetch_source_rows(
    adapter: Any,
    restaurant_id: str,
    month: str,
    peer_restaurant_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if peer_restaurant_ids:
        allowed = {restaurant_id, *peer_restaurant_ids}
        rows = []
        for current_id in sorted(allowed):
            rows.extend(adapter.fetch_reviews(restaurant_id=current_id, month=month))
        return rows
    return adapter.fetch_reviews(month=month)


def _cap_reviews_per_restaurant(
    rows: list[dict[str, Any]],
    max_reviews_per_restaurant: int,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    selected = []
    for row in rows:
        restaurant_id = str(row.get("restaurant_id") or "unknown")
        count = counts.get(restaurant_id, 0)
        if count >= max_reviews_per_restaurant:
            continue
        counts[restaurant_id] = count + 1
        selected.append(row)
    return selected


def _normalized_review(row: dict[str, Any], default_source: str) -> dict[str, Any]:
    normalized = normalize_review(row, default_source=default_source)
    normalized["review_id"] = str(
        row.get("review_id") or row.get("source_review_id") or normalized["review_text_hash"][:16]
    )
    return normalized


def _restaurant_rows(
    reviews: list[dict[str, Any]],
    target_restaurant_id: str,
    area_id: str,
    source: str,
) -> list[dict[str, Any]]:
    rows = {}
    for review in reviews:
        restaurant_id = review.get("restaurant_id") or target_restaurant_id
        rows[restaurant_id] = {
            "restaurant_id": restaurant_id,
            "source": source,
            "source_place_id": review.get("source_place_id") or restaurant_id,
            "name": review.get("restaurant_name") or restaurant_id,
            "area_id": area_id,
            "is_target": restaurant_id == target_restaurant_id,
            "is_peer": restaurant_id != target_restaurant_id,
            "status": "active",
        }
    return list(rows.values())
