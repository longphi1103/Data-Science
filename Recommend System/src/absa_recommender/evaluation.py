from typing import Any


def aspect_coverage(items: list[dict[str, Any]], minimum_mentions: int = 1) -> dict[str, Any]:
    covered = [
        item.get("aspect")
        for item in items
        if int(item.get("mention_count", 0)) >= minimum_mentions
    ]
    return {
        "aspect_count": len(set(covered)),
        "aspects": sorted(set(covered)),
    }


def peer_support_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    supported = sum(
        not bool(item.get("peer_summary", {}).get("peer_support_flag"))
        and "low_peer_support" not in item.get("data_quality_flags", [])
        for item in items
    )
    return supported / len(items)


def priority_score_stability(
    original_items: list[dict[str, Any]],
    rerun_items: list[dict[str, Any]],
    threshold: float = 30.0,
) -> dict[str, Any]:
    rerun_by_aspect = {item.get("aspect"): item for item in rerun_items}
    deltas = []
    unstable = []
    for item in original_items:
        aspect = item.get("aspect")
        if aspect not in rerun_by_aspect:
            continue
        delta = abs(
            float(item.get("priority_score", 0.0))
            - float(rerun_by_aspect[aspect].get("priority_score", 0.0))
        )
        deltas.append(delta)
        if delta > threshold:
            unstable.append({"aspect": aspect, "priority_delta": delta})
    return {
        "max_delta": max(deltas) if deltas else 0.0,
        "unstable_count": len(unstable),
        "unstable_items": unstable,
    }
