from absa_recommender.schemas import PriorityItem


def rank_priority_items(
    items: list[PriorityItem],
    top_n: int,
    force_food_safety_top3: bool = True,
    food_safety_negative_threshold: float = 0.10,
) -> list[PriorityItem]:
    ranked = sorted(items, key=lambda item: item.priority_score, reverse=True)
    if force_food_safety_top3 and top_n >= 3:
        index = next(
            (
                position
                for position, item in enumerate(ranked)
                if item.aspect == "Food Safety"
                and item.negative_rate_smoothed >= food_safety_negative_threshold
            ),
            None,
        )
        if index is not None and index >= 3:
            food_safety = ranked.pop(index)
            ranked.insert(2, food_safety)
    return [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(ranked[:top_n], start=1)
    ]
