import json
from pathlib import Path
from typing import Any

from absa_recommender.config import normalize_aspect_label, validate_aspect_label
from absa_recommender.config import normalize_sentiment_label, validate_sentiment_label
from absa_recommender.schemas import ABSAReview, AspectExtraction
from absa_recommender.severity import compute_severity, load_severity_config


def load_absa_jsonl(path: str | Path) -> list[ABSAReview]:
    reviews: list[ABSAReview] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                reviews.append(ABSAReview.model_validate(json.loads(line)))
    return reviews


def flatten_reviews(
    reviews: list[ABSAReview],
    label_schema: dict[str, Any],
    default_restaurant_id: str = "unknown",
    strict: bool = True,
    severity_config: dict[str, Any] | None = None,
) -> list[AspectExtraction]:
    extractions: list[AspectExtraction] = []
    severity_rules = severity_config or load_severity_config("configs/severity_lexicon.yaml")
    for review in reviews:
        restaurant_id = review.restaurant_id or default_restaurant_id
        for annotation_index, annotation in enumerate(review.annotations):
            if strict:
                aspect = validate_aspect_label(annotation.aspect_category, label_schema, strict=True)
                sentiment = validate_sentiment_label(annotation.sentiment, label_schema, strict=True)
            else:
                aspect = normalize_aspect_label(annotation.aspect_category, label_schema)
                sentiment = normalize_sentiment_label(annotation.sentiment, label_schema)

            extractions.append(
                AspectExtraction(
                    extraction_id=f"{review.review_id}_{annotation_index}",
                    review_id=review.review_id,
                    restaurant_id=restaurant_id,
                    restaurant_name=review.restaurant_name,
                    aspect=aspect,
                    aspect_term=annotation.aspect_expression,
                    opinion_text=annotation.opinion_expression,
                    sentiment=sentiment,
                    severity=compute_severity(
                        sentiment,
                        annotation.opinion_expression,
                        aspect=aspect,
                        config=severity_rules,
                    ),
                    model_confidence=annotation.model_confidence,
                    review_text=review.review_text,
                    rating=review.rating,
                    review_time=review.review_time,
                    review_month=_review_month(review),
                )
            )
    return extractions


def _review_month(review: ABSAReview) -> str:
    if review.review_month:
        return review.review_month
    if review.review_time is None:
        return "unknown"
    return review.review_time.strftime("%Y-%m")
