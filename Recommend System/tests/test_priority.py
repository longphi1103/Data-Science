from pathlib import Path

from absa_recommender.normalize_absa import load_absa_jsonl
from absa_recommender.recommender import generate_priority_ranking
from absa_recommender.schemas import PriorityItem


SAMPLE_PATH = Path("data/samples/absa_outputs.jsonl")


def test_end_to_end_sample_returns_priority_items() -> None:
    reviews = load_absa_jsonl(SAMPLE_PATH)

    response = generate_priority_ranking(reviews, top_n=5)

    assert response.restaurant_id == "multiple"
    assert response.items
    assert response.items[0].rank == 1
    assert response.top_n == 5


def test_output_is_aspect_only_without_subproblem_or_actions() -> None:
    reviews = load_absa_jsonl(SAMPLE_PATH)

    response = generate_priority_ranking(reviews, top_n=5)
    item = response.items[0]

    assert "sub_problem_id" not in PriorityItem.model_fields
    assert "recommended_actions" not in PriorityItem.model_fields
    assert "monitoring_kpis" not in PriorityItem.model_fields
    assert item.aspect
    assert 0 <= item.priority_score <= 100
    assert 0 <= item.priority_confidence <= 1


def test_opinion_examples_come_from_opinion_expression() -> None:
    reviews = load_absa_jsonl(SAMPLE_PATH)

    response = generate_priority_ranking(reviews, top_n=5)
    examples = {example for item in response.items for example in item.opinion_examples}

    assert any("ghi m" in example and "h" in example and "t m" in example for example in examples)


def test_priority_response_is_json_serializable() -> None:
    reviews = load_absa_jsonl(SAMPLE_PATH)

    response = generate_priority_ranking(reviews, top_n=5)

    assert response.model_dump(mode="json")
    assert response.model_dump_json()
