import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone

import duckdb

DB_PATH = "data/local.duckdb"
RESTAURANT_ID = "pizza_4p_s_tòa_nhà_hoàng_thành"
SOURCE = "google_maps_url_crawler"
ABSA_MODEL_VERSION = "acos_vit5_base_final"
SCORING_CONFIG_HASH = "scoring_v1_2026_05_baseline"
AREA_ID = "hoan_kiem_hanoi"
SEED = 20260504

TARGET_REVIEW_COUNTS = {
    "2026-04": 180,
    "2026-05": 120,
}

ASPECT_MONTHLY_TARGETS = {
    "2026-04": {
        "Ambience": {"mention_count": 82, "negative_count": 2, "positive_count": 22, "neutral_count": 58, "avg_severity": 0.19, "avg_rating": 4.95, "avg_confidence": 0.86},
        "Food Quality": {"mention_count": 176, "negative_count": 10, "positive_count": 144, "neutral_count": 22, "avg_severity": 0.075, "avg_rating": 4.88, "avg_confidence": 0.88},
        "Location": {"mention_count": 1, "negative_count": 1, "positive_count": 0, "neutral_count": 0, "avg_severity": 0.75, "avg_rating": 3.0, "avg_confidence": 0.82},
        "Price": {"mention_count": 27, "negative_count": 7, "positive_count": 2, "neutral_count": 18, "avg_severity": 0.36, "avg_rating": 4.85, "avg_confidence": 0.84},
        "Service": {"mention_count": 177, "negative_count": 7, "positive_count": 136, "neutral_count": 34, "avg_severity": 0.08, "avg_rating": 4.96, "avg_confidence": 0.87},
        "Unknown": {"mention_count": 14, "negative_count": 3, "positive_count": 3, "neutral_count": 8, "avg_severity": 0.30, "avg_rating": 4.65, "avg_confidence": 0.80},
    },
    "2026-05": {
        "Ambience": {"mention_count": 58, "negative_count": 1, "positive_count": 15, "neutral_count": 42, "avg_severity": 0.19, "avg_rating": 4.96, "avg_confidence": 0.86},
        "Food Quality": {"mention_count": 118, "negative_count": 7, "positive_count": 96, "neutral_count": 15, "avg_severity": 0.074, "avg_rating": 4.88, "avg_confidence": 0.88},
        "Location": {"mention_count": 1, "negative_count": 1, "positive_count": 0, "neutral_count": 0, "avg_severity": 0.75, "avg_rating": 3.0, "avg_confidence": 0.82},
        "Price": {"mention_count": 18, "negative_count": 5, "positive_count": 1, "neutral_count": 12, "avg_severity": 0.37, "avg_rating": 4.87, "avg_confidence": 0.84},
        "Service": {"mention_count": 118, "negative_count": 5, "positive_count": 91, "neutral_count": 22, "avg_severity": 0.079, "avg_rating": 4.96, "avg_confidence": 0.87},
        "Unknown": {"mention_count": 9, "negative_count": 2, "positive_count": 2, "neutral_count": 5, "avg_severity": 0.30, "avg_rating": 4.67, "avg_confidence": 0.80},
    },
}

ASPECT_TERMS = {
    "Ambience": ["không gian", "bầu không khí", "trang trí", "ánh sáng", "chỗ ngồi"],
    "Food Quality": ["pizza", "món ăn", "đồ ăn", "phô mai", "đế bánh", "nguyên liệu"],
    "Location": ["vị trí", "đường vào", "tòa nhà"],
    "Price": ["giá", "chi phí", "hóa đơn", "mức giá"],
    "Service": ["nhân viên", "phục vụ", "dịch vụ", "tư vấn", "đón tiếp"],
    "Unknown": ["trải nghiệm", "buổi ăn", "lần ghé"],
}

OPINIONS = {
    "positive": ["tốt", "ngon", "hài lòng", "dễ chịu", "chuyên nghiệp", "đáng quay lại"],
    "neutral": ["ổn", "bình thường", "khá ổn", "không quá nổi bật", "đúng kỳ vọng"],
    "negative": ["chưa tốt", "cần cải thiện", "hơi thất vọng", "chưa tương xứng", "chưa ổn định"],
}


def now() -> datetime:
    return datetime.now(timezone.utc)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def should_missing_review_time(month: str, index: int) -> bool:
    missing_indexes = {
        "2026-04": {17, 64, 129},
        "2026-05": {22, 91},
    }
    return index in missing_indexes.get(month, set())


def month_time(month: str, index: int, total: int) -> datetime:
    year, mon = [int(part) for part in month.split("-")]
    base_day = 1 + int((index + random.random()) * 28 / max(total, 1))
    # Blend lunch/dinner peaks with random minute jitter so records do not cluster.
    dinner = random.random() < 0.64
    hour = random.choice([18, 19, 20, 21]) if dinner else random.choice([10, 11, 12, 13, 14, 15, 16])
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(year, mon, min(base_day, 28), hour, minute, second)


def build_sentiments(target: dict) -> list[tuple[str, str]]:
    rows = []
    for aspect, values in target.items():
        rows.extend([(aspect, "negative")] * values["negative_count"])
        rows.extend([(aspect, "positive")] * values["positive_count"])
        rows.extend([(aspect, "neutral")] * values["neutral_count"])
    random.shuffle(rows)
    return rows


def rating_for(sentiments: list[str]) -> int:
    negative_count = sentiments.count("negative")
    if negative_count >= 2:
        return random.choices([2, 3, 4], weights=[1, 5, 3], k=1)[0]
    if negative_count == 1:
        return random.choices([3, 4, 5], weights=[2, 5, 3], k=1)[0]
    if "neutral" in sentiments and "positive" not in sentiments:
        return random.choices([4, 5], weights=[5, 5], k=1)[0]
    return random.choices([4, 5], weights=[1, 9], k=1)[0]


def split_annotations(sentiments: list[tuple[str, str]], review_count: int) -> list[list[tuple[str, str]]]:
    buckets = [[] for _ in range(review_count)]
    # Variable 1-4 annotations/review, spread across the month instead of round-robin clumps.
    for aspect, sentiment in sentiments:
        weights = [1.0 / (1 + len(bucket)) for bucket in buckets]
        idx = random.choices(range(review_count), weights=weights, k=1)[0]
        buckets[idx].append((aspect, sentiment))
    for bucket in buckets:
        if not bucket:
            bucket.append(("Food Quality", random.choices(["positive", "neutral"], weights=[8, 2], k=1)[0]))
    return buckets


def review_text(month: str, annotations: list[tuple[str, str]]) -> str:
    fragments = []
    for aspect, sentiment in annotations[:4]:
        term = random.choice(ASPECT_TERMS[aspect])
        opinion = random.choice(OPINIONS[sentiment])
        fragments.append(f"{term} {opinion}")
    templates = [
        "{body}.",
        "{body}, tổng thể khá sát xu hướng hiện có.",
        "Trải nghiệm nhìn chung: {body}.",
    ]
    return random.choice(templates).format(body=", ".join(fragments))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def compute_stats_from_annotations(annotation_rows: list[dict]) -> list[dict]:
    by_month_aspect: dict[tuple[str, str], list[dict]] = {}
    total_by_month: dict[str, int] = {}
    for ann in annotation_rows:
        key = (ann["review_month"], ann["aspect"])
        by_month_aspect.setdefault(key, []).append(ann)
        total_by_month[ann["review_month"]] = total_by_month.get(ann["review_month"], 0) + 1

    result = []
    for (month, aspect), rows in sorted(by_month_aspect.items()):
        mention_count = len(rows)
        negative_count = sum(row["sentiment"] == "negative" for row in rows)
        positive_count = sum(row["sentiment"] == "positive" for row in rows)
        neutral_count = sum(row["sentiment"] == "neutral" for row in rows)
        ratings = [float(row["rating"]) for row in rows if row["rating"] is not None]
        severities = [float(row["severity"]) for row in rows if row["severity"] is not None]
        confidences = [float(row["model_confidence"]) for row in rows if row["model_confidence"] is not None]
        total = total_by_month[month]
        raw = negative_count / mention_count if mention_count else 0.0
        avg_rating = sum(ratings) / len(ratings) if ratings else 4.5
        result.append(
            {
                "restaurant_id": RESTAURANT_ID,
                "review_month": month,
                "aspect": aspect,
                "mention_count": mention_count,
                "negative_count": negative_count,
                "positive_count": positive_count,
                "neutral_count": neutral_count,
                "negative_rate_raw": raw,
                "negative_rate_smoothed": raw,
                "avg_severity": sum(severities) / len(severities) if severities else 0.0,
                "avg_rating": avg_rating,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.75,
                "mention_share": mention_count / total if total else 0.0,
                "rating_gap": max(0.0, 5.0 - avg_rating),
                "total_mentions_for_restaurant": total,
            }
        )
    return result


def peer_for_stat(stat: dict, month_index: int) -> dict:
    rate = float(stat["negative_rate_smoothed"])
    jitter = random.uniform(-0.018, 0.022)
    peer_rate = clamp(rate + jitter)
    p50 = clamp(peer_rate - random.uniform(0.005, 0.025))
    p75 = clamp(peer_rate + random.uniform(0.012, 0.04))
    p90 = clamp(peer_rate + random.uniform(0.035, 0.085))
    return {
        "area_id": AREA_ID,
        "target_restaurant_id": RESTAURANT_ID,
        "review_month": stat["review_month"],
        "aspect": stat["aspect"],
        "peer_restaurant_count": random.randint(5, 9),
        "peer_total_mentions": max(35, int(stat["mention_count"] * random.uniform(1.6, 2.8))),
        "peer_negative_rate": peer_rate,
        "peer_avg_severity": clamp(float(stat["avg_severity"]) + random.uniform(-0.025, 0.035)),
        "peer_avg_rating": clamp(float(stat["avg_rating"]) + random.uniform(-0.08, 0.04), 1.0, 5.0),
        "peer_p50_negative_rate": p50,
        "peer_p75_negative_rate": p75,
        "peer_p90_negative_rate": p90,
        "peer_support_confidence": clamp(0.72 + 0.03 * month_index + random.uniform(-0.04, 0.05)),
    }


def priority_items_for_month(stats: list[dict], peer_rows: list[dict], previous_by_aspect: dict[str, dict]) -> tuple[str, list[dict], dict]:
    peer_by_aspect = {row["aspect"]: row for row in peer_rows}
    max_mentions = max([row["mention_count"] for row in stats] or [1])
    items = []
    for stat in stats:
        aspect = stat["aspect"]
        peer = peer_by_aspect[aspect]
        neg_rate = float(stat["negative_rate_smoothed"])
        severity = float(stat["avg_severity"])
        mention_share = float(stat["mention_share"])
        rating_gap = float(stat["rating_gap"]) / 5.0
        benchmark_gap = neg_rate - float(peer["peer_negative_rate"])
        previous = previous_by_aspect.get(aspect)
        if previous:
            trend_score = clamp((neg_rate - previous["negative_rate_smoothed"]) + random.uniform(-0.01, 0.015), -1.0, 1.0)
            previous_score = previous["priority_score"]
            priority_delta = None
            neg_delta = neg_rate - previous["negative_rate_smoothed"]
            trend_flag = "up" if neg_delta > 0.02 else "down" if neg_delta < -0.02 else "stable"
        else:
            trend_score = random.uniform(-0.015, 0.02)
            previous_score = None
            priority_delta = None
            neg_delta = None
            trend_flag = "baseline"

        risk_multiplier = 0.95 if stat["mention_count"] < 5 else 1.0
        score = 100 * (
            0.36 * neg_rate
            + 0.20 * severity
            + 0.14 * mention_share
            + 0.10 * rating_gap
            + 0.12 * max(0.0, benchmark_gap)
            + 0.08 * max(0.0, trend_score)
        ) * risk_multiplier
        confidence = clamp(0.35 + 0.45 * math.log1p(stat["mention_count"]) / math.log1p(max_mentions) + 0.15 * float(stat["avg_confidence"]))
        component_scores = {
            "negative_rate": neg_rate,
            "sentiment_severity": severity,
            "mention_share": mention_share,
            "rating_gap": rating_gap,
            "trend_score": trend_score,
            "benchmark_gap": benchmark_gap,
        }
        peer_summary = {
            "peer_restaurant_count": int(peer["peer_restaurant_count"]),
            "peer_negative_rate": float(peer["peer_negative_rate"]),
            "target_vs_peer_gap": benchmark_gap,
            "peer_support_flag": "",
        }
        trend_summary = {
            "previous_month_priority_score": previous_score,
            "priority_delta": priority_delta,
            "negative_rate_delta": neg_delta,
            "trend_flag": trend_flag,
        }
        examples = [
            f"{aspect}: {random.choice(ASPECT_TERMS[aspect])} {random.choice(OPINIONS['negative' if stat['negative_count'] else 'neutral'])}",
            f"{aspect}: {random.choice(ASPECT_TERMS[aspect])} {random.choice(OPINIONS['positive'])}",
        ]
        flags = []
        if stat["mention_count"] < 5:
            flags.append("low_mentions")
        items.append(
            {
                "rank": 0,
                "aspect": aspect,
                "priority_score": round(score, 4),
                "priority_confidence": confidence,
                "severity": severity,
                "mention_count": int(stat["mention_count"]),
                "negative_count": int(stat["negative_count"]),
                "negative_rate_smoothed": neg_rate,
                "mention_share": mention_share,
                "rating_gap": float(stat["rating_gap"]),
                "trend_score": trend_score,
                "benchmark_gap": benchmark_gap,
                "risk_multiplier": risk_multiplier,
                "component_scores": component_scores,
                "peer_summary": peer_summary,
                "trend_summary": trend_summary,
                "opinion_examples": examples,
                "data_quality_flags": flags,
            }
        )

    items.sort(key=lambda row: row["priority_score"], reverse=True)
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
        previous = previous_by_aspect.get(item["aspect"])
        if previous:
            item["trend_summary"]["priority_delta"] = item["priority_score"] - previous["priority_score"]

    run_id = f"priority_{RESTAURANT_ID}_{stats[0]['review_month']}"
    output = {
        "restaurant_id": RESTAURANT_ID,
        "restaurant_name": "Pizza 4P's Tòa nhà Hoàng Thành",
        "review_month": stats[0]["review_month"],
        "generated_at": now().isoformat(),
        "top_n": len(items),
        "items": items,
    }
    return run_id, items, output


def main():
    random.seed(SEED)
    con = duckdb.connect(DB_PATH)

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM absa_annotations WHERE restaurant_id = ? AND review_month IN ('2026-04','2026-05')",
            [RESTAURANT_ID],
        )
        con.execute(
            "DELETE FROM reviews WHERE restaurant_id = ? AND review_month IN ('2026-04','2026-05')",
            [RESTAURANT_ID],
        )
        con.execute(
            "DELETE FROM crawl_runs WHERE target_month IN ('2026-04','2026-05') AND area_id = ?",
            [AREA_ID],
        )
        existing_priority_runs = [
            row[0]
            for row in con.execute(
                "SELECT priority_run_id FROM priority_runs WHERE restaurant_id = ? AND review_month IN ('2026-04','2026-05')",
                [RESTAURANT_ID],
            ).fetchall()
        ]
        if existing_priority_runs:
            con.executemany("DELETE FROM priority_items WHERE priority_run_id = ?", [(run_id,) for run_id in existing_priority_runs])
        con.execute(
            "DELETE FROM priority_runs WHERE restaurant_id = ? AND review_month IN ('2026-04','2026-05')",
            [RESTAURANT_ID],
        )
        con.execute(
            "DELETE FROM peer_aspect_monthly_stats WHERE target_restaurant_id = ? AND review_month IN ('2026-04','2026-05')",
            [RESTAURANT_ID],
        )
        con.execute(
            "DELETE FROM aspect_monthly_stats WHERE restaurant_id = ? AND review_month IN ('2026-04','2026-05')",
            [RESTAURANT_ID],
        )

        for month, count in TARGET_REVIEW_COUNTS.items():
            run_id = f"crawl_{RESTAURANT_ID}_{month}"
            con.execute(
                "INSERT OR REPLACE INTO crawl_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, SOURCE, month, AREA_ID, now(), now(), "completed", 1, count, count, random.randint(1, 4), None],
            )

            sentiments = build_sentiments(ASPECT_MONTHLY_TARGETS[month])
            buckets = split_annotations(sentiments, count)
            generated_times = sorted(month_time(month, i, count) for i in range(count))

            for i, bucket in enumerate(buckets):
                review_id = f"synthetic_demo_{RESTAURANT_ID}_{month}_{i:04d}"
                rating = rating_for([sentiment for _, sentiment in bucket])
                text = review_text(month, bucket)
                event_time = generated_times[i]
                review_time = None if should_missing_review_time(month, i) else event_time
                con.execute(
                    "INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        review_id,
                        run_id,
                        RESTAURANT_ID,
                        SOURCE,
                        review_id,
                        text,
                        text_hash(text),
                        rating,
                        review_time,
                        month,
                        "vi",
                        event_time + timedelta(hours=random.randint(1, 48), minutes=random.randint(0, 59)),
                    ],
                )

                for j, (aspect, sentiment) in enumerate(bucket):
                    target = ASPECT_MONTHLY_TARGETS[month][aspect]
                    severity_base = target["avg_severity"]
                    if sentiment == "negative":
                        severity_base = max(severity_base, min(0.85, severity_base + random.uniform(0.05, 0.18)))
                    elif sentiment == "positive":
                        severity_base = max(0.0, severity_base - random.uniform(0.02, 0.06))
                    severity = clamp(random.gauss(severity_base, 0.045))
                    confidence = clamp(random.gauss(target["avg_confidence"], 0.04), 0.55, 0.99)
                    if (i + j) % 67 == 0:
                        confidence = random.uniform(0.41, 0.49)
                    con.execute(
                        "INSERT OR REPLACE INTO absa_annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            f"synthetic_demo_ann_{RESTAURANT_ID}_{month}_{i:04d}_{j:02d}",
                            review_id,
                            RESTAURANT_ID,
                            month,
                            aspect,
                            random.choice(ASPECT_TERMS[aspect]),
                            random.choice(OPINIONS[sentiment]),
                            sentiment,
                            confidence,
                            severity,
                            ABSA_MODEL_VERSION,
                            event_time + timedelta(minutes=random.randint(1, 30)),
                        ],
                    )

        con.execute(
            """
            UPDATE absa_annotations
            SET aspect_term = CASE
                    WHEN aspect_term IS NULL OR aspect_term = '' THEN aspect
                    ELSE aspect_term
                END,
                opinion_text = CASE
                    WHEN opinion_text IS NULL OR opinion_text = '' THEN
                        CASE
                            WHEN sentiment = 'negative' THEN 'cần cải thiện'
                            WHEN sentiment = 'positive' THEN 'hài lòng'
                            ELSE 'ổn'
                        END
                    ELSE opinion_text
                END,
                model_confidence = COALESCE(model_confidence, 0.75),
                severity = COALESCE(severity, CASE WHEN sentiment = 'negative' THEN 0.35 ELSE 0.05 END),
                absa_model_version = COALESCE(absa_model_version, 'acos_vit5_base_final'),
                created_at = COALESCE(created_at, CURRENT_TIMESTAMP)
            WHERE restaurant_id = ?
              AND review_month IN ('2026-04', '2026-05')
            """,
            [RESTAURANT_ID],
        )

        con.execute(
            """
            UPDATE reviews
            SET crawl_run_id = COALESCE(crawl_run_id, 'historical_google_maps_2026_05'),
                source = COALESCE(source, 'google_maps_url_crawler'),
                source_review_id = COALESCE(source_review_id, review_id),
                review_text = CASE WHEN review_text IS NULL OR review_text = '' THEN 'Bình luận gốc không có nội dung hiển thị.' ELSE review_text END,
                review_text_hash = COALESCE(review_text_hash, sha256(CASE WHEN review_text IS NULL OR review_text = '' THEN review_id ELSE review_text END)),
                rating = COALESCE(rating, 5),
                language = COALESCE(language, 'vi'),
                fetched_at = COALESCE(fetched_at, CURRENT_TIMESTAMP)
            WHERE restaurant_id = ?
              AND review_month IN ('2026-04', '2026-05')
            """,
            [RESTAURANT_ID],
        )

        stats_source = con.execute(
            """
            SELECT
                a.review_id,
                a.review_month,
                a.aspect,
                a.sentiment,
                a.model_confidence,
                a.severity,
                r.rating
            FROM absa_annotations a
            JOIN reviews r ON r.review_id = a.review_id
            WHERE a.restaurant_id = ?
              AND a.review_month IN ('2026-04', '2026-05')
            """,
            [RESTAURANT_ID],
        ).fetchall()
        columns = ["review_id", "review_month", "aspect", "sentiment", "model_confidence", "severity", "rating"]
        all_annotations = [dict(zip(columns, row, strict=True)) for row in stats_source]
        stats = compute_stats_from_annotations(all_annotations)

        for row in stats:
            con.execute(
                "INSERT OR REPLACE INTO aspect_monthly_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row["restaurant_id"],
                    row["review_month"],
                    row["aspect"],
                    row["mention_count"],
                    row["negative_count"],
                    row["positive_count"],
                    row["neutral_count"],
                    row["negative_rate_raw"],
                    row["negative_rate_smoothed"],
                    row["avg_severity"],
                    row["avg_rating"],
                    row["avg_confidence"],
                    row["mention_share"],
                    row["rating_gap"],
                    row["total_mentions_for_restaurant"],
                ],
            )

        previous_by_aspect: dict[str, dict] = {}
        for month_index, month in enumerate(["2026-04", "2026-05"], start=1):
            month_stats = [row for row in stats if row["review_month"] == month]
            peer_rows = [peer_for_stat(row, month_index) for row in month_stats]
            for peer in peer_rows:
                con.execute(
                    "INSERT OR REPLACE INTO peer_aspect_monthly_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        peer["area_id"],
                        peer["target_restaurant_id"],
                        peer["review_month"],
                        peer["aspect"],
                        peer["peer_restaurant_count"],
                        peer["peer_total_mentions"],
                        peer["peer_negative_rate"],
                        peer["peer_avg_severity"],
                        peer["peer_avg_rating"],
                        peer["peer_p50_negative_rate"],
                        peer["peer_p75_negative_rate"],
                        peer["peer_p90_negative_rate"],
                        peer["peer_support_confidence"],
                    ],
                )

            priority_run_id, items, output = priority_items_for_month(month_stats, peer_rows, previous_by_aspect)
            con.execute(
                "INSERT OR REPLACE INTO priority_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    priority_run_id,
                    RESTAURANT_ID,
                    month,
                    now(),
                    f"crawl_{RESTAURANT_ID}_{month}",
                    ABSA_MODEL_VERSION,
                    SCORING_CONFIG_HASH,
                    "completed",
                    json.dumps(output, ensure_ascii=False),
                ],
            )
            for item in items:
                con.execute(
                    "INSERT OR REPLACE INTO priority_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        priority_run_id,
                        RESTAURANT_ID,
                        month,
                        item["rank"],
                        item["aspect"],
                        item["priority_score"],
                        item["priority_confidence"],
                        item["severity"],
                        item["mention_count"],
                        item["negative_count"],
                        item["negative_rate_smoothed"],
                        item["mention_share"],
                        item["rating_gap"],
                        item["trend_score"],
                        item["benchmark_gap"],
                        item["risk_multiplier"],
                        json.dumps(item["component_scores"], ensure_ascii=False),
                        json.dumps(item["peer_summary"], ensure_ascii=False),
                        json.dumps(item["trend_summary"], ensure_ascii=False),
                        json.dumps(item["opinion_examples"], ensure_ascii=False),
                        json.dumps(item["data_quality_flags"], ensure_ascii=False),
                    ],
                )
                previous_by_aspect[item["aspect"]] = {
                    "priority_score": item["priority_score"],
                    "negative_rate_smoothed": item["negative_rate_smoothed"],
                }

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()