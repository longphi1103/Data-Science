from pathlib import Path

from absa_recommender.severity import compute_severity, load_severity_config


CONFIG = load_severity_config(Path("configs/severity_lexicon.yaml"))


def test_strong_negative_pattern_scores_at_least_0_9() -> None:
    assert compute_severity("negative", "bàn rất bẩn", config=CONFIG) >= 0.9


def test_mild_negative_pattern_scores_0_6() -> None:
    assert compute_severity("negative", "giá hơi cao", config=CONFIG) == 0.6


def test_positive_scores_0() -> None:
    assert compute_severity("positive", "nhân viên thân thiện", config=CONFIG) == 0.0


def test_safety_pattern_scores_at_least_0_95() -> None:
    assert compute_severity("negative", "ăn xong bị ốm nặng", config=CONFIG) >= 0.95


def test_location_negative_defaults_at_least_0_75() -> None:
    assert compute_severity("negative", "khó tìm", aspect="Location", config=CONFIG) >= 0.75


def test_menu_negative_defaults_at_least_0_75() -> None:
    assert compute_severity("negative", "menu khó hiểu", aspect="Menu", config=CONFIG) >= 0.75
