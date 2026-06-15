import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from absa_recommender.aggregation import aggregate_aspect_stats
from absa_recommender.absa_inference import ABSAInferenceAdapter, build_absa_adapter
from absa_recommender.benchmark import compute_peer_aspect_stats
from absa_recommender.config import load_label_schema, load_yaml
from absa_recommender.crawler.monthly import crawl_reviews_for_month
from absa_recommender.dedup import deduplicate_reviews
from absa_recommender.normalize_absa import flatten_reviews, load_absa_jsonl
from absa_recommender.recommender import generate_priority_ranking
from absa_recommender.review_normalizer import normalize_review
from absa_recommender.schemas import ABSAReview, AspectExtraction, AspectStats
from absa_recommender.scoring import (
    compute_global_negative_rate_by_aspect,
    smoothed_negative_rate,
)
from absa_recommender.severity import load_severity_config
from absa_recommender.storage import (
    default_db_path,
    find_priority_run,
    previous_priority_by_aspect,
    save_absa_annotations,
    save_aspect_monthly_stats,
    save_crawl_run,
    save_peer_aspect_monthly_stats,
    save_priority_run,
    save_restaurants,
    save_reviews,
)
from absa_recommender.sources.google_maps_crawler_adapter import GoogleMapsCrawlerAdapter
from absa_recommender.sources.local_jsonl_adapter import LocalJsonlAdapter


def run_monthly_from_absa_jsonl(
    input_path: str | Path,
    restaurant_id: str,
    review_month: str,
    top_n: int = 5,
    db_path: str | Path | None = None,
    force: bool = False,
    area_id: str = "local",
    source: str = "local_jsonl",
    absa_model_version: str | None = None,
) -> dict[str, Any]:
    reviews = load_absa_jsonl(input_path)
    return run_monthly_from_reviews(
        reviews,
        restaurant_id=restaurant_id,
        review_month=review_month,
        top_n=top_n,
        db_path=db_path,
        force=force,
        area_id=area_id,
        source=source,
        absa_model_version=absa_model_version,
    )


def run_monthly_from_source(
    input_path: str | Path,
    restaurant_id: str,
    review_month: str,
    top_n: int = 5,
    db_path: str | Path | None = None,
    force: bool = False,
    area_id: str = "local",
    absa_adapter: str | ABSAInferenceAdapter = "placeholder",
    source_adapter: str = "local-jsonl",
    gmaps_live: bool = False,
    gmaps_discover_from_area: bool = False,
    gmaps_area_name: str | None = None,
    gmaps_bbox: str | None = None,
    gmaps_target_url: str | None = None,
    progress_callback: Callable[[str, int | None], None] | None = None,
) -> dict[str, Any]:
    _report_progress(progress_callback, "Preparing source adapter", 5)
    source_adapter_obj = _build_source_adapter(
        source_adapter,
        input_path=input_path,
        restaurant_id=restaurant_id,
        gmaps_live=gmaps_live,
        gmaps_discover_from_area=gmaps_discover_from_area,
        gmaps_area_name=gmaps_area_name,
        gmaps_bbox=gmaps_bbox,
        gmaps_target_url=gmaps_target_url,
        progress_callback=progress_callback,
    )
    _report_progress(progress_callback, "Crawling reviews from source", 15)
    crawl_result = crawl_reviews_for_month(
        source_adapter_obj,
        restaurant_id=restaurant_id,
        month=review_month,
        area_id=area_id,
    )
    _report_progress(
        progress_callback,
        (
            "Crawl completed: "
            f"{len(crawl_result.reviews)} unique reviews, "
            f"{len(crawl_result.duplicate_reviews)} duplicates, "
            f"{len(crawl_result.restaurants)} restaurants"
        ),
        45,
    )
    _report_progress(progress_callback, f"Building ABSA adapter: {absa_adapter}", 50)
    adapter = build_absa_adapter(absa_adapter) if isinstance(absa_adapter, str) else absa_adapter
    _report_progress(progress_callback, "Running ABSA inference", 60)
    reviews = adapter.infer(crawl_result.reviews)
    _report_progress(progress_callback, f"ABSA inference completed: {len(reviews)} reviews", 72)
    _report_progress(progress_callback, "Aggregating, scoring, and saving monthly results", 82)
    result = run_monthly_from_reviews(
        reviews,
        restaurant_id=restaurant_id,
        review_month=review_month,
        top_n=top_n,
        db_path=db_path,
        force=force,
        area_id=area_id,
        source=crawl_result.source,
        absa_model_version=adapter.model_version,
    )
    result["source_reviews_fetched"] = len(crawl_result.reviews) + len(crawl_result.duplicate_reviews)
    result["source_duplicates"] = len(crawl_result.duplicate_reviews)
    result["source_restaurants"] = len(crawl_result.restaurants)
    result["absa_adapter"] = adapter.model_version
    result["source_adapter"] = crawl_result.source
    _report_progress(progress_callback, "Pipeline completed", 100)
    return result


def _build_source_adapter(
    source_adapter: str,
    input_path: str | Path,
    restaurant_id: str,
    gmaps_live: bool = False,
    gmaps_discover_from_area: bool = False,
    gmaps_area_name: str | None = None,
    gmaps_bbox: str | None = None,
    gmaps_target_url: str | None = None,
    progress_callback: Callable[[str, int | None], None] | None = None,
) -> Any:
    normalized = source_adapter.strip().lower().replace("_", "-")
    if normalized in {"local", "local-jsonl", "jsonl"}:
        return LocalJsonlAdapter(input_path)
    if normalized in {"google-maps", "gmaps", "google-maps-url-crawler"}:
        crawler_config = load_yaml("configs/crawler.yaml")
        gmaps_config = crawler_config.get("google_maps", {})
        adapter_config = gmaps_config.get("adapter", {})
        peer_config = gmaps_config.get("peer_discovery", {})
        return GoogleMapsCrawlerAdapter(
            crawler_script=adapter_config.get("crawler_script"),
            output_path=input_path,
            input_urls_path=adapter_config.get("input_urls_path"),
            mode=str(adapter_config.get("mode", "benchmark")),
            target_restaurant_id=restaurant_id,
            target_url=gmaps_target_url,
            target_restaurant_name=adapter_config.get("target_restaurant_name"),
            area_name=gmaps_area_name,
            bbox=gmaps_bbox,
            discover_from_area=gmaps_discover_from_area,
            live=gmaps_live,
            headful=bool(_config_value(adapter_config, "headful", True)),
            max_discovered_places=int(_config_value(adapter_config, "max_discovered_places", 3)),
            max_reviews_per_restaurant=int(
                _config_value(
                    adapter_config,
                    "max_reviews_per_restaurant",
                    _config_value(crawler_config, "max_reviews_per_restaurant", 10),
                )
            ),
            stop_after_old_reviews=int(
                _config_value(
                    adapter_config,
                    "stop_after_old_reviews",
                    _config_value(crawler_config, "stop_after_old_reviews", 20),
                )
            ),
            min_peers=int(
                _config_value(
                    adapter_config,
                    "min_peers",
                    _config_value(peer_config, "min_peers", 1 if gmaps_discover_from_area else 0),
                )
            ),
            min_restaurants=int(
                _config_value(
                    adapter_config,
                    "min_restaurants",
                    _config_value(peer_config, "min_restaurants", 2 if gmaps_discover_from_area else 1),
                )
            ),
            no_area_network=bool(_config_value(adapter_config, "no_area_network", False)),
            search_queries=adapter_config.get("search_queries"),
            log_callback=lambda message: _report_progress(
                progress_callback,
                f"crawler: {message}",
                None,
            ),
        )
    raise ValueError(f"Unknown source adapter: {source_adapter}")


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _report_progress(
    progress_callback: Callable[[str, int | None], None] | None,
    message: str,
    percent: int | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(message, percent)
    print(f"[monthly-pipeline] {message}", flush=True)


def _legacy_run_monthly_from_source(
    input_path: str | Path,
    restaurant_id: str,
    review_month: str,
    top_n: int = 5,
    db_path: str | Path | None = None,
    force: bool = False,
    area_id: str = "local",
    absa_adapter: str | ABSAInferenceAdapter = "placeholder",
) -> dict[str, Any]:
    source_adapter = LocalJsonlAdapter(input_path)
    crawl_result = crawl_reviews_for_month(
        source_adapter,
        restaurant_id=restaurant_id,
        month=review_month,
        area_id=area_id,
    )
    adapter = build_absa_adapter(absa_adapter) if isinstance(absa_adapter, str) else absa_adapter
    reviews = adapter.infer(crawl_result.reviews)
    result = run_monthly_from_reviews(
        reviews,
        restaurant_id=restaurant_id,
        review_month=review_month,
        top_n=top_n,
        db_path=db_path,
        force=force,
        area_id=area_id,
        source=crawl_result.source,
        absa_model_version=adapter.model_version,
    )
    result["source_reviews_fetched"] = len(crawl_result.reviews) + len(crawl_result.duplicate_reviews)
    result["source_duplicates"] = len(crawl_result.duplicate_reviews)
    result["source_restaurants"] = len(crawl_result.restaurants)
    result["absa_adapter"] = adapter.model_version
    return result


def run_monthly_from_reviews(
    reviews: list[ABSAReview],
    restaurant_id: str,
    review_month: str,
    top_n: int = 5,
    db_path: str | Path | None = None,
    force: bool = False,
    area_id: str = "local",
    source: str = "local_jsonl",
    absa_model_version: str | None = None,
) -> dict[str, Any]:
    database = Path(db_path) if db_path is not None else default_db_path()
    scoring_config_hash = config_hash("configs/scoring.yaml")
    model_version = absa_model_version or _absa_model_version()
    existing = find_priority_run(
        database,
        restaurant_id,
        review_month,
        scoring_config_hash=scoring_config_hash,
        absa_model_version=model_version,
    )
    if existing is not None and not force:
        return {
            "status": "existing",
            "db_path": str(database),
            "priority_run_id": existing["priority_run_id"],
            "output": existing["output"],
        }

    month_reviews = _month_reviews(reviews, review_month)
    if not month_reviews:
        raise ValueError(f"No reviews found for review_month={review_month}")

    label_schema = load_label_schema("configs/label_schema.yaml")
    severity_config = load_severity_config("configs/severity_lexicon.yaml")
    scoring_config = load_yaml("configs/scoring.yaml")
    extractions = flatten_reviews(
        month_reviews,
        label_schema,
        default_restaurant_id=restaurant_id,
        strict=True,
        severity_config=severity_config,
    )
    stats = _stats_with_smoothed_rates(extractions, label_schema, scoring_config)
    peer_rows = _peer_rows_for_target(stats, restaurant_id, review_month, area_id)
    peer_benchmarks = {
        (restaurant_id, review_month, row["aspect"]): {
            "peer_restaurant_count": row["peer_restaurant_count"],
            "peer_total_mentions": row["peer_total_mentions"],
            "peer_negative_rate": row["peer_negative_rate"],
        }
        for row in peer_rows
    }
    previous_priority = previous_priority_by_aspect(database, restaurant_id, review_month)
    target_reviews = [review for review in month_reviews if review.restaurant_id == restaurant_id]
    if not target_reviews:
        target_reviews = [
            review.model_copy(update={"restaurant_id": restaurant_id})
            for review in month_reviews
        ]
    response = generate_priority_ranking(
        target_reviews,
        top_n=top_n,
        default_restaurant_id=restaurant_id,
        review_month=review_month,
        peer_benchmarks=peer_benchmarks,
        previous_priority=previous_priority,
    )

    normalized_reviews = [_review_row(review, source) for review in month_reviews]
    unique_reviews, duplicate_reviews = deduplicate_reviews(normalized_reviews)
    crawl_run_id = save_crawl_run(
        database,
        source=source,
        target_month=review_month,
        area_id=area_id,
        status="success",
        num_restaurants=len({review.get("restaurant_id") for review in unique_reviews}),
        num_reviews_fetched=len(normalized_reviews),
        num_reviews_inserted=len(unique_reviews),
        num_duplicates=len(duplicate_reviews),
    )
    for review in unique_reviews:
        review["crawl_run_id"] = crawl_run_id
    save_restaurants(database, _restaurant_rows(month_reviews, restaurant_id, area_id, source))
    save_reviews(database, unique_reviews)
    save_absa_annotations(
        database,
        _annotation_rows(extractions, model_version),
    )
    save_aspect_monthly_stats(database, stats)
    save_peer_aspect_monthly_stats(database, peer_rows)
    priority_run_id = save_priority_run(
        database,
        response,
        scoring_config_hash=scoring_config_hash,
        crawl_run_id=crawl_run_id,
        absa_model_version=model_version,
    )
    return {
        "status": "completed",
        "db_path": str(database),
        "crawl_run_id": crawl_run_id,
        "priority_run_id": priority_run_id,
        "reviews_inserted": len(unique_reviews),
        "duplicates": len(duplicate_reviews),
        "annotations": len(extractions),
        "aspect_stats": len(stats),
        "peer_benchmarks": len(peer_rows),
        "output": response.model_dump(mode="json"),
    }


def config_hash(path: str | Path) -> str:
    payload = load_yaml(path)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _month_reviews(reviews: list[ABSAReview], review_month: str) -> list[ABSAReview]:
    result = []
    for review in reviews:
        current_month = review.review_month
        if current_month is None and review.review_time is not None:
            current_month = review.review_time.strftime("%Y-%m")
        if current_month == review_month:
            result.append(review.model_copy(update={"review_month": review_month}))
    return result


def _stats_with_smoothed_rates(
    extractions: list[AspectExtraction],
    label_schema: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[AspectStats]:
    stats = aggregate_aspect_stats(extractions, scoring_config)
    scoring = scoring_config.get("scoring", scoring_config)
    alpha = float(scoring.get("smoothing", {}).get("alpha", 10))
    global_rates = compute_global_negative_rate_by_aspect(extractions, label_schema)
    return [
        item.model_copy(
            update={
                "negative_rate_smoothed": smoothed_negative_rate(
                    item.negative_count,
                    item.mention_count,
                    global_rates.get(item.aspect, 0.0),
                    alpha,
                )
            }
        )
        for item in stats
    ]


def _peer_rows_for_target(
    stats: list[AspectStats],
    restaurant_id: str,
    review_month: str,
    area_id: str,
) -> list[dict[str, Any]]:
    aspects = sorted(
        {
            item.aspect
            for item in stats
            if item.restaurant_id == restaurant_id and item.review_month == review_month
        }
    )
    stat_rows = [item.model_dump(mode="json") for item in stats]
    rows = [
        compute_peer_aspect_stats(
            restaurant_id,
            area_id,
            review_month,
            aspect,
            stat_rows,
        )
        for aspect in aspects
    ]
    for row in rows:
        total = float(row.get("peer_total_mentions") or 0)
        row["peer_support_confidence"] = min(1.0, total / 100.0)
    return rows


def _review_row(review: ABSAReview, source: str) -> dict[str, Any]:
    row = normalize_review(
        {
            "review_id": review.review_id,
            "review_text": review.review_text,
            "restaurant_id": review.restaurant_id,
            "rating": review.rating,
            "review_time": review.review_time,
            "review_month": review.review_month,
            "source": source,
            "source_review_id": review.review_id,
            "language": "vi",
        },
        default_source=source,
    )
    row["review_id"] = review.review_id
    return row


def _restaurant_rows(
    reviews: list[ABSAReview],
    target_restaurant_id: str,
    area_id: str,
    source: str,
) -> list[dict[str, Any]]:
    rows = {}
    for review in reviews:
        restaurant_id = review.restaurant_id or target_restaurant_id
        rows[restaurant_id] = {
            "restaurant_id": restaurant_id,
            "source": source,
            "source_place_id": restaurant_id,
            "name": review.restaurant_name or restaurant_id,
            "area_id": area_id,
            "is_target": restaurant_id == target_restaurant_id,
            "is_peer": restaurant_id != target_restaurant_id,
            "status": "active",
        }
    return list(rows.values())


def _annotation_rows(
    extractions: list[AspectExtraction],
    absa_model_version: str,
) -> list[dict[str, Any]]:
    return [
        {
            "annotation_id": extraction.extraction_id,
            "review_id": extraction.review_id,
            "restaurant_id": extraction.restaurant_id,
            "review_month": extraction.review_month,
            "aspect": extraction.aspect,
            "aspect_term": extraction.aspect_term,
            "opinion_text": extraction.opinion_text,
            "sentiment": extraction.sentiment,
            "model_confidence": extraction.model_confidence,
            "severity": extraction.severity,
            "absa_model_version": absa_model_version,
        }
        for extraction in extractions
    ]


def _absa_model_version() -> str:
    config = load_yaml("configs/absa_model.yaml")
    return str(config.get("model", {}).get("version", "unknown"))
