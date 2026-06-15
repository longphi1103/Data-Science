from absa_recommender.evaluation import (
    aspect_coverage,
    peer_support_rate,
    priority_score_stability,
)


def test_aspect_coverage() -> None:
    coverage = aspect_coverage(
        [
            {"aspect": "Service", "mention_count": 5},
            {"aspect": "Menu", "mention_count": 0},
        ],
        minimum_mentions=1,
    )

    assert coverage["aspect_count"] == 1
    assert coverage["aspects"] == ["Service"]


def test_peer_support_rate() -> None:
    rate = peer_support_rate(
        [
            {"peer_summary": {}, "data_quality_flags": []},
            {"peer_summary": {"peer_support_flag": "low_peer_support"}, "data_quality_flags": []},
        ]
    )

    assert rate == 0.5


def test_priority_score_stability() -> None:
    stability = priority_score_stability(
        [{"aspect": "Service", "priority_score": 50}],
        [{"aspect": "Service", "priority_score": 85}],
    )

    assert stability["max_delta"] == 35
    assert stability["unstable_count"] == 1
