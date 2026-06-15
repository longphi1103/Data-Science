from absa_recommender.monitoring import build_monitoring_snapshot


def test_build_monitoring_snapshot_tracks_data_quality_metrics() -> None:
    snapshot = build_monitoring_snapshot(
        crawl_runs=[
            {
                "status": "failed",
                "num_reviews_fetched": 10,
                "num_reviews_inserted": 7,
                "num_duplicates": 3,
            }
        ],
        reviews=[
            {"review_id": "1", "review_time": None},
            {"review_id": "2", "review_time": "2026-06-01T00:00:00"},
        ],
        annotations=[
            {"model_confidence": 0.4},
            {"model_confidence": 0.9},
        ],
        priority_items=[
            {
                "aspect": "Cleanliness",
                "mention_count": 10,
                "peer_summary": {"peer_support_flag": "low_peer_support"},
                "data_quality_flags": ["low_peer_support"],
            }
        ],
    )

    assert snapshot["crawl_success_rate"] == 0.0
    assert snapshot["reviews_fetched_count"] == 10
    assert snapshot["new_review_count"] == 7
    assert snapshot["duplicate_rate"] == 0.3
    assert snapshot["missing_review_time_rate"] == 0.5
    assert snapshot["low_confidence_annotation_rate"] == 0.5
    assert snapshot["peer_support_rate"] == 0.0
    assert snapshot["alerts"]
