from pathlib import Path

import pytest

from absa_recommender.config import (
    load_label_schema,
    load_yaml,
    normalize_aspect_label,
    normalize_sentiment_label,
    validate_aspect_label,
    validate_sentiment_label,
)


SCHEMA_PATH = Path("configs/label_schema.yaml")


def test_all_config_files_are_valid_yaml() -> None:
    for path in Path("configs").glob("*.yaml"):
        assert load_yaml(path) is not None


def test_official_aspect_labels_pass() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    for label in [
        "Food Quality",
        "Food Safety",
        "Service",
        "Price",
        "Cleanliness",
        "Ambience",
        "Location",
        "Menu",
    ]:
        assert validate_aspect_label(label, schema) == label


def test_food_quality_alias_maps_to_canonical_label() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    assert normalize_aspect_label("food_quality", schema) == "Food Quality"
    assert validate_aspect_label("food_quality", schema) == "Food Quality"


def test_sentiment_labels_and_permissive_fallback() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    assert validate_sentiment_label("positive", schema) == "positive"
    assert validate_sentiment_label("neutral", schema) == "neutral"
    assert validate_sentiment_label("negative", schema) == "negative"
    assert normalize_sentiment_label("mixed", schema) == "neutral"
    assert validate_sentiment_label("mixed", schema, strict=False) == "neutral"


def test_unknown_sentiment_fails_in_strict_mode() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    with pytest.raises(ValueError):
        validate_sentiment_label("mixed", schema, strict=True)


def test_location_passes() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    assert validate_aspect_label("Location", schema) == "Location"


def test_menu_passes() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    assert validate_aspect_label("Menu", schema) == "Menu"


def test_unknown_aspect_fails_in_strict_mode() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    with pytest.raises(ValueError):
        validate_aspect_label("Parking", schema, strict=True)


def test_unknown_aspect_maps_to_unknown_in_permissive_mode() -> None:
    schema = load_label_schema(SCHEMA_PATH)

    assert validate_aspect_label("Parking", schema, strict=False) == "Unknown"
