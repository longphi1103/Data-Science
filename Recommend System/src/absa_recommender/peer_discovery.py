from math import asin, cos, radians, sin, sqrt
from typing import Any


def discover_peer_restaurants(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    radius_meters: int,
    max_peers: int,
    included_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    included = set(included_types or ["restaurant"])
    peers = [
        candidate
        for candidate in candidates
        if candidate.get("restaurant_id") != target.get("restaurant_id")
        and candidate.get("status", "active") == "active"
        and included.intersection(set(candidate.get("types", ["restaurant"])))
        and _distance_meters(target, candidate) <= radius_meters
    ]
    return sorted(peers, key=lambda item: _distance_meters(target, item))[:max_peers]


def _distance_meters(left: dict[str, Any], right: dict[str, Any]) -> float:
    lat1, lng1 = radians(float(left["lat"])), radians(float(left["lng"]))
    lat2, lng2 = radians(float(right["lat"])), radians(float(right["lng"]))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 6371000 * 2 * asin(sqrt(a))
