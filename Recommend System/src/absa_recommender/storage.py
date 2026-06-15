import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from absa_recommender.schemas import PriorityResponse


DEFAULT_DB_PATH = Path("data/local.duckdb")


def default_db_path() -> Path:
    return Path(os.environ.get("ABSA_DB_PATH", DEFAULT_DB_PATH))


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS restaurants (
                restaurant_id VARCHAR PRIMARY KEY,
                source VARCHAR,
                source_place_id VARCHAR,
                name VARCHAR,
                lat DOUBLE,
                lng DOUBLE,
                area_id VARCHAR,
                is_target BOOLEAN,
                is_peer BOOLEAN,
                status VARCHAR,
                first_seen_at TIMESTAMP,
                last_seen_at TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_runs (
                crawl_run_id VARCHAR PRIMARY KEY,
                source VARCHAR,
                target_month VARCHAR,
                area_id VARCHAR,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                status VARCHAR,
                num_restaurants INTEGER,
                num_reviews_fetched INTEGER,
                num_reviews_inserted INTEGER,
                num_duplicates INTEGER,
                error_message VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                review_id VARCHAR PRIMARY KEY,
                crawl_run_id VARCHAR,
                restaurant_id VARCHAR,
                source VARCHAR,
                source_review_id VARCHAR,
                review_text VARCHAR,
                review_text_hash VARCHAR,
                rating INTEGER,
                review_time TIMESTAMP,
                review_month VARCHAR,
                language VARCHAR,
                fetched_at TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS absa_annotations (
                annotation_id VARCHAR PRIMARY KEY,
                review_id VARCHAR,
                restaurant_id VARCHAR,
                review_month VARCHAR,
                aspect VARCHAR,
                aspect_term VARCHAR,
                opinion_text VARCHAR,
                sentiment VARCHAR,
                model_confidence DOUBLE,
                severity DOUBLE,
                absa_model_version VARCHAR,
                created_at TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS aspect_monthly_stats (
                restaurant_id VARCHAR,
                review_month VARCHAR,
                aspect VARCHAR,
                mention_count INTEGER,
                negative_count INTEGER,
                positive_count INTEGER,
                neutral_count INTEGER,
                negative_rate_raw DOUBLE,
                negative_rate_smoothed DOUBLE,
                avg_severity DOUBLE,
                avg_rating DOUBLE,
                avg_confidence DOUBLE,
                mention_share DOUBLE,
                rating_gap DOUBLE,
                total_mentions_for_restaurant INTEGER,
                PRIMARY KEY (restaurant_id, review_month, aspect)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_aspect_monthly_stats (
                area_id VARCHAR,
                target_restaurant_id VARCHAR,
                review_month VARCHAR,
                aspect VARCHAR,
                peer_restaurant_count INTEGER,
                peer_total_mentions INTEGER,
                peer_negative_rate DOUBLE,
                peer_avg_severity DOUBLE,
                peer_avg_rating DOUBLE,
                peer_p50_negative_rate DOUBLE,
                peer_p75_negative_rate DOUBLE,
                peer_p90_negative_rate DOUBLE,
                peer_support_confidence DOUBLE,
                PRIMARY KEY (area_id, target_restaurant_id, review_month, aspect)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS priority_runs (
                priority_run_id VARCHAR PRIMARY KEY,
                restaurant_id VARCHAR,
                review_month VARCHAR,
                generated_at TIMESTAMP,
                crawl_run_id VARCHAR,
                absa_model_version VARCHAR,
                scoring_config_hash VARCHAR,
                status VARCHAR,
                output_json VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS priority_items (
                priority_run_id VARCHAR,
                restaurant_id VARCHAR,
                review_month VARCHAR,
                rank INTEGER,
                aspect VARCHAR,
                priority_score DOUBLE,
                priority_confidence DOUBLE,
                severity DOUBLE,
                mention_count INTEGER,
                negative_count INTEGER,
                negative_rate_smoothed DOUBLE,
                mention_share DOUBLE,
                rating_gap DOUBLE,
                trend_score DOUBLE,
                benchmark_gap DOUBLE,
                risk_multiplier DOUBLE,
                component_scores_json VARCHAR,
                peer_summary_json VARCHAR,
                trend_summary_json VARCHAR,
                opinion_examples_json VARCHAR,
                data_quality_flags_json VARCHAR,
                PRIMARY KEY (priority_run_id, rank)
            )
            """
        )


def save_restaurants(db_path: str | Path, restaurants: list[dict[str, Any]]) -> None:
    init_db(db_path)
    with _connect(db_path) as connection:
        for restaurant in restaurants:
            connection.execute(
                """
                INSERT OR REPLACE INTO restaurants
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    restaurant.get("restaurant_id"),
                    restaurant.get("source", "local_jsonl"),
                    restaurant.get("source_place_id"),
                    restaurant.get("name"),
                    restaurant.get("lat"),
                    restaurant.get("lng"),
                    restaurant.get("area_id", "local"),
                    restaurant.get("is_target", False),
                    restaurant.get("is_peer", False),
                    restaurant.get("status", "active"),
                    restaurant.get("first_seen_at") or _now(),
                    restaurant.get("last_seen_at") or _now(),
                ],
            )


def save_crawl_run(
    db_path: str | Path,
    source: str,
    target_month: str,
    area_id: str,
    status: str,
    num_restaurants: int,
    num_reviews_fetched: int,
    num_reviews_inserted: int,
    num_duplicates: int,
    error_message: str | None = None,
    crawl_run_id: str | None = None,
) -> str:
    init_db(db_path)
    run_id = crawl_run_id or _new_id("crawl")
    now = _now()
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO crawl_runs
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                source,
                target_month,
                area_id,
                now,
                now,
                status,
                num_restaurants,
                num_reviews_fetched,
                num_reviews_inserted,
                num_duplicates,
                error_message,
            ],
        )
    return run_id


def save_reviews(db_path: str | Path, reviews: list[dict[str, Any]]) -> None:
    init_db(db_path)
    with _connect(db_path) as connection:
        for review in reviews:
            connection.execute(
                """
                INSERT OR REPLACE INTO reviews
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    review.get("review_id"),
                    review.get("crawl_run_id"),
                    review.get("restaurant_id"),
                    review.get("source", "local_jsonl"),
                    review.get("source_review_id"),
                    review.get("review_text"),
                    review.get("review_text_hash"),
                    review.get("rating"),
                    review.get("review_time"),
                    review.get("review_month"),
                    review.get("language", "vi"),
                    review.get("fetched_at") or _now(),
                ],
            )


def save_absa_annotations(db_path: str | Path, annotations: list[dict[str, Any]]) -> None:
    init_db(db_path)
    with _connect(db_path) as connection:
        for annotation in annotations:
            connection.execute(
                """
                INSERT OR REPLACE INTO absa_annotations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    annotation.get("annotation_id"),
                    annotation.get("review_id"),
                    annotation.get("restaurant_id"),
                    annotation.get("review_month"),
                    annotation.get("aspect"),
                    annotation.get("aspect_term"),
                    annotation.get("opinion_text"),
                    annotation.get("sentiment"),
                    annotation.get("model_confidence"),
                    annotation.get("severity"),
                    annotation.get("absa_model_version", "unknown"),
                    annotation.get("created_at") or _now(),
                ],
            )


def save_aspect_monthly_stats(db_path: str | Path, stats: list[Any]) -> None:
    init_db(db_path)
    with _connect(db_path) as connection:
        for item in stats:
            payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            connection.execute(
                """
                INSERT OR REPLACE INTO aspect_monthly_stats
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    payload.get("restaurant_id"),
                    payload.get("review_month"),
                    payload.get("aspect"),
                    payload.get("mention_count"),
                    payload.get("negative_count"),
                    payload.get("positive_count"),
                    payload.get("neutral_count"),
                    payload.get("negative_rate_raw"),
                    payload.get("negative_rate_smoothed"),
                    payload.get("avg_severity"),
                    payload.get("avg_rating"),
                    payload.get("avg_confidence"),
                    payload.get("mention_share"),
                    payload.get("rating_gap"),
                    payload.get("total_mentions_for_restaurant"),
                ],
            )


def save_peer_aspect_monthly_stats(db_path: str | Path, rows: list[dict[str, Any]]) -> None:
    init_db(db_path)
    with _connect(db_path) as connection:
        for row in rows:
            connection.execute(
                """
                INSERT OR REPLACE INTO peer_aspect_monthly_stats
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row.get("area_id"),
                    row.get("target_restaurant_id"),
                    row.get("review_month"),
                    row.get("aspect"),
                    row.get("peer_restaurant_count"),
                    row.get("peer_total_mentions"),
                    row.get("peer_negative_rate"),
                    row.get("peer_avg_severity"),
                    row.get("peer_avg_rating"),
                    row.get("peer_p50_negative_rate"),
                    row.get("peer_p75_negative_rate"),
                    row.get("peer_p90_negative_rate"),
                    row.get("peer_support_confidence", 0.0),
                ],
            )


def save_priority_run(
    db_path: str | Path,
    response: PriorityResponse,
    scoring_config_hash: str,
    crawl_run_id: str | None = None,
    absa_model_version: str = "unknown",
    status: str = "completed",
    priority_run_id: str | None = None,
) -> str:
    init_db(db_path)
    run_id = priority_run_id or _new_id("priority")
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO priority_runs
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                response.restaurant_id,
                response.review_month,
                response.generated_at,
                crawl_run_id,
                absa_model_version,
                scoring_config_hash,
                status,
                response.model_dump_json(),
            ],
        )
    save_priority_items(db_path, run_id, response)
    return run_id


def save_priority_items(
    db_path: str | Path,
    priority_run_id: str,
    response: PriorityResponse,
) -> None:
    init_db(db_path)
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM priority_items WHERE priority_run_id = ?", [priority_run_id])
        for item in response.items:
            connection.execute(
                """
                INSERT INTO priority_items
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    priority_run_id,
                    response.restaurant_id,
                    response.review_month,
                    item.rank,
                    item.aspect,
                    item.priority_score,
                    item.priority_confidence,
                    item.severity,
                    item.mention_count,
                    item.negative_count,
                    item.negative_rate_smoothed,
                    item.mention_share,
                    item.rating_gap,
                    item.trend_score,
                    item.benchmark_gap,
                    item.risk_multiplier,
                    _json(item.component_scores),
                    item.peer_summary.model_dump_json(),
                    item.trend_summary.model_dump_json(),
                    _json(item.opinion_examples),
                    _json(item.data_quality_flags),
                ],
            )


def list_priority_runs(db_path: str | Path, restaurant_id: str | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as connection:
        if restaurant_id is None:
            rows = connection.execute(
                """
                SELECT priority_run_id, restaurant_id, review_month, generated_at,
                       crawl_run_id, absa_model_version, scoring_config_hash, status
                FROM priority_runs
                ORDER BY generated_at DESC
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT priority_run_id, restaurant_id, review_month, generated_at,
                       crawl_run_id, absa_model_version, scoring_config_hash, status
                FROM priority_runs
                WHERE restaurant_id = ?
                ORDER BY generated_at DESC
                """,
                [restaurant_id],
            ).fetchall()
    columns = [
        "priority_run_id",
        "restaurant_id",
        "review_month",
        "generated_at",
        "crawl_run_id",
        "absa_model_version",
        "scoring_config_hash",
        "status",
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def list_restaurants(db_path: str | Path) -> list[dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT restaurant_id, name, area_id, is_target, is_peer, status
            FROM restaurants
            ORDER BY is_target DESC, restaurant_id
            """
        ).fetchall()
    columns = ["restaurant_id", "name", "area_id", "is_target", "is_peer", "status"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def list_review_months(db_path: str | Path, restaurant_id: str | None = None) -> list[str]:
    init_db(db_path)
    with _connect(db_path) as connection:
        if restaurant_id is None:
            rows = connection.execute(
                "SELECT DISTINCT review_month FROM reviews ORDER BY review_month DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT DISTINCT review_month FROM reviews
                WHERE restaurant_id = ?
                ORDER BY review_month DESC
                """,
                [restaurant_id],
            ).fetchall()
    return [row[0] for row in rows if row[0]]


def find_priority_run(
    db_path: str | Path,
    restaurant_id: str,
    review_month: str,
    scoring_config_hash: str | None = None,
    absa_model_version: str | None = None,
) -> dict[str, Any] | None:
    init_db(db_path)
    filters = ["restaurant_id = ?", "review_month = ?"]
    params: list[Any] = [restaurant_id, review_month]
    if scoring_config_hash is not None:
        filters.append("scoring_config_hash = ?")
        params.append(scoring_config_hash)
    if absa_model_version is not None:
        filters.append("absa_model_version = ?")
        params.append(absa_model_version)
    with _connect(db_path) as connection:
        row = connection.execute(
            f"""
            SELECT priority_run_id
            FROM priority_runs
            WHERE {" AND ".join(filters)}
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    if row is None:
        return None
    return get_priority_run(db_path, row[0])


def get_latest_priority_response(
    db_path: str | Path,
    restaurant_id: str,
    review_month: str,
) -> dict[str, Any] | None:
    return find_priority_run(db_path, restaurant_id, review_month)


def get_aspect_monthly_stats(
    db_path: str | Path,
    restaurant_id: str | None = None,
    review_month: str | None = None,
) -> list[dict[str, Any]]:
    init_db(db_path)
    filters = []
    params: list[Any] = []
    if restaurant_id is not None:
        filters.append("restaurant_id = ?")
        params.append(restaurant_id)
    if review_month is not None:
        filters.append("review_month = ?")
        params.append(review_month)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT restaurant_id, review_month, aspect, mention_count, negative_count,
                   positive_count, neutral_count, negative_rate_raw,
                   negative_rate_smoothed, avg_severity, avg_rating, avg_confidence,
                   mention_share, rating_gap, total_mentions_for_restaurant
            FROM aspect_monthly_stats
            {where}
            ORDER BY restaurant_id, review_month, aspect
            """,
            params,
        ).fetchall()
    columns = [
        "restaurant_id",
        "review_month",
        "aspect",
        "mention_count",
        "negative_count",
        "positive_count",
        "neutral_count",
        "negative_rate_raw",
        "negative_rate_smoothed",
        "avg_severity",
        "avg_rating",
        "avg_confidence",
        "mention_share",
        "rating_gap",
        "total_mentions_for_restaurant",
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def get_peer_benchmarks(
    db_path: str | Path,
    restaurant_id: str,
    review_month: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = list_peer_benchmark(db_path, restaurant_id, review_month)
    return {
        (restaurant_id, review_month, row["aspect"]): {
            "peer_restaurant_count": row["peer_restaurant_count"],
            "peer_total_mentions": row["peer_total_mentions"],
            "peer_negative_rate": row["peer_negative_rate"],
        }
        for row in rows
    }


def list_peer_benchmark(
    db_path: str | Path,
    restaurant_id: str,
    review_month: str,
) -> list[dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT area_id, target_restaurant_id, review_month, aspect,
                   peer_restaurant_count, peer_total_mentions, peer_negative_rate,
                   peer_avg_severity, peer_avg_rating, peer_p50_negative_rate,
                   peer_p75_negative_rate, peer_p90_negative_rate,
                   peer_support_confidence
            FROM peer_aspect_monthly_stats
            WHERE target_restaurant_id = ? AND review_month = ?
            ORDER BY aspect
            """,
            [restaurant_id, review_month],
        ).fetchall()
    columns = [
        "area_id",
        "target_restaurant_id",
        "review_month",
        "aspect",
        "peer_restaurant_count",
        "peer_total_mentions",
        "peer_negative_rate",
        "peer_avg_severity",
        "peer_avg_rating",
        "peer_p50_negative_rate",
        "peer_p75_negative_rate",
        "peer_p90_negative_rate",
        "peer_support_confidence",
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def previous_priority_by_aspect(
    db_path: str | Path,
    restaurant_id: str,
    current_month: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT aspect, priority_score, severity, mention_count, negative_rate_smoothed
            FROM priority_items
            WHERE restaurant_id = ?
              AND review_month < ?
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY aspect ORDER BY review_month DESC, priority_run_id DESC
            ) = 1
            """,
            [restaurant_id, current_month],
        ).fetchall()
    return {
        (restaurant_id, row[0]): {
            "priority_score": row[1],
            "severity": row[2],
            "mention_count": row[3],
            "negative_rate_smoothed": row[4],
        }
        for row in rows
    }


def dashboard_payload(db_path: str | Path, restaurant_id: str, review_month: str) -> dict[str, Any]:
    priority = get_latest_priority_response(db_path, restaurant_id, review_month)
    stats = get_aspect_monthly_stats(db_path, restaurant_id, review_month)
    peer_rows = list_peer_benchmark(db_path, restaurant_id, review_month)
    crawl = latest_crawl_run(db_path, review_month)
    return {
        "restaurant_id": restaurant_id,
        "review_month": review_month,
        "overview": overview_metrics(db_path, restaurant_id, review_month),
        "priority": [] if priority is None else priority["output"].get("items", []),
        "priority_run": priority,
        "aspect_stats": stats,
        "peer_benchmark": peer_rows,
        "data_quality": data_quality_metrics(db_path, restaurant_id, review_month),
        "crawl_run": crawl,
    }


def overview_metrics(db_path: str | Path, restaurant_id: str, review_month: str) -> dict[str, Any]:
    init_db(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*), AVG(rating), SUM(CASE WHEN review_time IS NULL THEN 1 ELSE 0 END)
            FROM reviews
            WHERE restaurant_id = ? AND review_month = ?
            """,
            [restaurant_id, review_month],
        ).fetchone()
        ann = connection.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END)
            FROM absa_annotations
            WHERE restaurant_id = ? AND review_month = ?
            """,
            [restaurant_id, review_month],
        ).fetchone()
    review_count = int(row[0] or 0)
    annotation_count = int(ann[0] or 0)
    negative_count = int(ann[1] or 0)
    return {
        "total_reviews": review_count,
        "total_absa_annotations": annotation_count,
        "average_rating": float(row[1]) if row[1] is not None else None,
        "negative_annotation_rate": (
            negative_count / annotation_count if annotation_count else 0.0
        ),
        "missing_review_time_count": int(row[2] or 0),
    }


def data_quality_metrics(db_path: str | Path, restaurant_id: str, review_month: str) -> dict[str, Any]:
    init_db(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN model_confidence IS NOT NULL
                            AND model_confidence < 0.5 THEN 1 ELSE 0 END)
            FROM absa_annotations
            WHERE restaurant_id = ? AND review_month = ?
            """,
            [restaurant_id, review_month],
        ).fetchone()
    total = int(row[0] or 0)
    low_confidence = int(row[1] or 0)
    return {
        "low_confidence_annotation_count": low_confidence,
        "low_confidence_annotation_rate": low_confidence / total if total else 0.0,
    }


def latest_crawl_run(db_path: str | Path, review_month: str) -> dict[str, Any] | None:
    init_db(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT crawl_run_id, source, target_month, area_id, started_at, finished_at,
                   status, num_restaurants, num_reviews_fetched, num_reviews_inserted,
                   num_duplicates, error_message
            FROM crawl_runs
            WHERE target_month = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            [review_month],
        ).fetchone()
    if row is None:
        return None
    columns = [
        "crawl_run_id",
        "source",
        "target_month",
        "area_id",
        "started_at",
        "finished_at",
        "status",
        "num_restaurants",
        "num_reviews_fetched",
        "num_reviews_inserted",
        "num_duplicates",
        "error_message",
    ]
    return dict(zip(columns, row, strict=True))


def aspect_history(db_path: str | Path, restaurant_id: str, aspect: str) -> list[dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT review_month, priority_score, priority_confidence, severity,
                   negative_rate_smoothed, trend_score, benchmark_gap
            FROM priority_items
            WHERE restaurant_id = ? AND aspect = ?
            ORDER BY review_month
            """,
            [restaurant_id, aspect],
        ).fetchall()
    columns = [
        "review_month",
        "priority_score",
        "priority_confidence",
        "severity",
        "negative_rate_smoothed",
        "trend_score",
        "benchmark_gap",
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def get_priority_run(db_path: str | Path, priority_run_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT priority_run_id, restaurant_id, review_month, generated_at,
                   crawl_run_id, absa_model_version, scoring_config_hash, status, output_json
            FROM priority_runs
            WHERE priority_run_id = ?
            """,
            [priority_run_id],
        ).fetchone()
    if row is None:
        return None
    columns = [
        "priority_run_id",
        "restaurant_id",
        "review_month",
        "generated_at",
        "crawl_run_id",
        "absa_model_version",
        "scoring_config_hash",
        "status",
        "output_json",
    ]
    result = dict(zip(columns, row, strict=True))
    result["output"] = json.loads(result["output_json"])
    return result


def _connect(db_path: str | Path):
    return duckdb.connect(str(db_path))


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> datetime:
    return datetime.now(timezone.utc)
