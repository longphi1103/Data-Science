from typing import Any

from absa_recommender.evaluation import aspect_coverage, peer_support_rate


def build_monitoring_snapshot(
    crawl_runs: list[dict[str, Any]] | None = None,
    reviews: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    priority_items: list[dict[str, Any]] | None = None,
    dashboard_generated_at: str | None = None,
    latest_crawl_finished_at: str | None = None,
    low_confidence_threshold: float = 0.50,
) -> dict[str, Any]:
    crawl_rows = crawl_runs or []
    review_rows = reviews or []
    annotation_rows = annotations or []
    item_rows = priority_items or []
    snapshot = {
        "crawl_success_rate": crawl_success_rate(crawl_rows),
        "reviews_fetched_count": reviews_fetched_count(crawl_rows, review_rows),
        "new_review_count": new_review_count(crawl_rows, review_rows),
        "duplicate_rate": duplicate_rate(crawl_rows),
        "missing_review_time_rate": missing_review_time_rate(review_rows),
        "absa_inference_failure_rate": absa_inference_failure_rate(annotation_rows),
        "low_confidence_annotation_rate": low_confidence_annotation_rate(
            annotation_rows,
            threshold=low_confidence_threshold,
        ),
        "aspect_coverage": aspect_coverage(item_rows),
        "peer_support_rate": peer_support_rate(item_rows),
        "dashboard_data_freshness": dashboard_data_freshness(
            dashboard_generated_at,
            latest_crawl_finished_at,
        ),
    }
    snapshot["alerts"] = suggest_monitoring_alerts(snapshot)
    return snapshot


def crawl_success_rate(crawl_runs: list[dict[str, Any]]) -> float:
    if not crawl_runs:
        return 0.0
    successes = sum(str(run.get("status", "")).lower() == "success" for run in crawl_runs)
    return successes / len(crawl_runs)


def reviews_fetched_count(crawl_runs: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> int:
    if crawl_runs:
        return sum(int(run.get("num_reviews_fetched", 0)) for run in crawl_runs)
    return len(reviews)


def new_review_count(crawl_runs: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> int:
    if crawl_runs:
        return sum(int(run.get("num_reviews_inserted", 0)) for run in crawl_runs)
    return sum(not bool(review.get("is_duplicate", False)) for review in reviews)


def duplicate_rate(crawl_runs: list[dict[str, Any]]) -> float:
    inserted = sum(int(run.get("num_reviews_inserted", 0)) for run in crawl_runs)
    duplicates = sum(int(run.get("num_duplicates", 0)) for run in crawl_runs)
    total = inserted + duplicates
    if total <= 0:
        return 0.0
    return duplicates / total


def missing_review_time_rate(reviews: list[dict[str, Any]]) -> float:
    if not reviews:
        return 0.0
    missing = sum(not review.get("review_time") for review in reviews)
    return missing / len(reviews)


def absa_inference_failure_rate(annotations: list[dict[str, Any]]) -> float:
    if not annotations:
        return 0.0
    failed = sum(str(row.get("status", "success")).lower() == "failed" for row in annotations)
    return failed / len(annotations)


def low_confidence_annotation_rate(
    annotations: list[dict[str, Any]],
    threshold: float = 0.50,
) -> float:
    if not annotations:
        return 0.0
    low = sum(float(row.get("model_confidence") or 0.0) < threshold for row in annotations)
    return low / len(annotations)


def dashboard_data_freshness(
    dashboard_generated_at: str | None,
    latest_crawl_finished_at: str | None,
) -> dict[str, Any]:
    return {
        "dashboard_generated_at": dashboard_generated_at,
        "latest_crawl_finished_at": latest_crawl_finished_at,
        "is_stale": bool(
            dashboard_generated_at
            and latest_crawl_finished_at
            and dashboard_generated_at < latest_crawl_finished_at
        ),
    }


def suggest_monitoring_alerts(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if snapshot.get("crawl_success_rate", 1.0) < 0.90:
        alerts.append(
            {
                "metric": "crawl_success_rate",
                "value": snapshot.get("crawl_success_rate"),
                "message": "Crawl success rate is below 90%; check source adapter or quota.",
            }
        )
    if snapshot.get("peer_support_rate", 1.0) < 0.70:
        alerts.append(
            {
                "metric": "peer_support_rate",
                "value": snapshot.get("peer_support_rate"),
                "message": "Peer support is below 70%; widen area or review peer filters.",
            }
        )
    if snapshot.get("missing_review_time_rate", 0.0) > 0.30:
        alerts.append(
            {
                "metric": "missing_review_time_rate",
                "value": snapshot.get("missing_review_time_rate"),
                "message": "More than 30% of reviews are missing review_time; trend may be unreliable.",
            }
        )
    if snapshot.get("low_confidence_annotation_rate", 0.0) > 0.25:
        alerts.append(
            {
                "metric": "low_confidence_annotation_rate",
                "value": snapshot.get("low_confidence_annotation_rate"),
                "message": "Low-confidence annotation rate is above 25%; check ABSA model quality.",
            }
        )
    return alerts
