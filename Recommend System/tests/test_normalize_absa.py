from pathlib import Path

import pytest

from absa_recommender.config import load_label_schema
from absa_recommender.normalize_absa import flatten_reviews, load_absa_jsonl
from absa_recommender.schemas import AspectExtraction


SCHEMA = load_label_schema(Path("configs/label_schema.yaml"))
SAMPLE_PATH = Path("data/samples/absa_outputs.jsonl")


def test_sample_records_parse() -> None:
    reviews = load_absa_jsonl(SAMPLE_PATH)

    assert len(reviews) == 3
    assert reviews[0].review_id == "rv_001"
    assert reviews[0].annotations[0].aspect_category == "food_quality"


def test_location_and_menu_records_parse() -> None:
    reviews = load_absa_jsonl(SAMPLE_PATH)
    aspect_categories = {
        annotation.aspect_category
        for review in reviews
        for annotation in review.annotations
    }

    assert "Location" in aspect_categories
    assert "Menu" in aspect_categories


def test_flatten_creates_one_extraction_per_annotation() -> None:
    reviews = load_absa_jsonl(SAMPLE_PATH)
    extractions = flatten_reviews(reviews, SCHEMA)
    annotation_count = sum(len(review.annotations) for review in reviews)

    assert len(extractions) == annotation_count


def test_flatten_maps_annotation_fields_to_internal_names() -> None:
    review = load_absa_jsonl(SAMPLE_PATH)[0]
    extraction = flatten_reviews([review], SCHEMA)[0]

    assert extraction.extraction_id == "rv_001_0"
    assert extraction.aspect == "Food Quality"
    assert extraction.aspect_term == "phở bò"
    assert extraction.opinion_text == "không hề đậm đà không có vị thịt chỉ toàn nước muối"
    assert extraction.sentiment == "negative"
    assert extraction.severity >= 0.9
    assert extraction.review_month == "2026-05"


def test_unknown_aspect_strict_mode_fails_during_flatten() -> None:
    review = load_absa_jsonl(SAMPLE_PATH)[0].model_copy(deep=True)
    review.annotations[0].aspect_category = "Parking"

    with pytest.raises(ValueError):
        flatten_reviews([review], SCHEMA, strict=True)


def test_unknown_aspect_permissive_mode_maps_to_unknown() -> None:
    review = load_absa_jsonl(SAMPLE_PATH)[0].model_copy(deep=True)
    review.annotations[0].aspect_category = "Parking"

    extraction = flatten_reviews([review], SCHEMA, strict=False)[0]

    assert extraction.aspect == "Unknown"


def test_aspect_extraction_has_no_internal_evidence_field() -> None:
    assert "evidence" not in AspectExtraction.model_fields
