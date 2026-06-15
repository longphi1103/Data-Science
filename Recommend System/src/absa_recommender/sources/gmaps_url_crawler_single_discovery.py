#!/usr/bin/env python3
"""
gmaps_url_crawler_single_discovery.py

Single-file Google Maps URL crawler pipeline for restaurant review JSONL.
Supports two modes plus optional area discovery:
- collection: crawl any restaurants/candidates without requiring a target.
- benchmark: require a target restaurant and validate peer coverage.

What this file includes:
- URL canonicalization and stable source_place_id extraction from Google Maps URLs.
- Stable restaurant_id and review_id generation.
- Relative review time parsing: Vietnamese + English common forms.
- Offline HTML/snapshot parser for test fixtures and saved snapshots.
- Normalization to LocalJsonlAdapter-compatible JSONL.
- Dedup and validation.
- Optional bbox/polygon area filtering using coordinates parsed from Google Maps URLs.
- Automatic --area-name resolution to GeoJSON polygon+bbox via cached Nominatim result.
- Optional area discovery from Google Maps search UI/saved search snapshot.
- Optional Playwright live crawler skeleton, without CAPTCHA/proxy/anti-bot bypass logic.

Important:
- The offline mode is the reliable/tested core.
- The live Google Maps UI mode requires selector tuning against real snapshots because Google Maps DOM changes often.
- Run only where you are allowed to collect the data and where doing so complies with relevant terms/laws.

Examples:
  python gmaps_url_crawler_single.py --self-test
  python gmaps_url_crawler_single.py --offline-demo --output raw_reviews_2026-05_area_x.jsonl

Optional live mode, after installing Playwright:
  pip install playwright
  playwright install chromium
  python gmaps_url_crawler_single.py --live --input-urls configs/google_maps_urls.jsonl --crawl-month 2026-05 --area-name "Phường Hàng Trống, Hoàn Kiếm, Hà Nội, Việt Nam" --output raw_reviews_2026-05_area_x.jsonl

Input URL JSONL format for live mode:
  # collection mode: role/restaurant_id may be omitted; role defaults to candidate.
  {"restaurant_name":"Restaurant A","google_maps_url":"https://www.google.com/maps/place/..."}

  # benchmark mode: mark one restaurant as target, others as peer.
  {"role":"target","restaurant_id":"res_demo","restaurant_name":"May Tre Dan Restaurant","google_maps_url":"https://www.google.com/maps/place/..."}
  {"role":"peer","restaurant_id":"res_peer_01","restaurant_name":"Peer A","google_maps_url":"https://www.google.com/maps/place/..."}
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_lib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import URLError, HTTPError
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

# -----------------------------------------------------------------------------
# URL identity
# -----------------------------------------------------------------------------

TRACKING_QUERY_KEYS = {
    "entry",
    "g_ep",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "hl",  # locale is better configured explicitly by crawler
}

GOOGLE_FEATURE_RE = re.compile(r"0x[0-9a-fA-F]+:0x[0-9a-fA-F]+")
CID_RE = re.compile(r"[?&]cid=(\d+)")
COORD_RE = re.compile(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)z")
PLACE_COORD_RE = re.compile(r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)")
PLACE_NAME_RE = re.compile(r"/maps/place/([^/@?]+)")
PLACE_SHORT_ID_RE = re.compile(r"!16s([^!?&/]+)")
WHITESPACE_RE = re.compile(r"\s+")
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
RATING_RE = re.compile(r"(\d)(?:[,.]\d+)?")


def sha1_short(value: str, n: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = html_lib.unescape(value)
    value = unicodedata.normalize("NFC", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value


def normalize_for_hash(value: Optional[str]) -> str:
    value = clean_text(value).lower()
    value = unicodedata.normalize("NFKC", value)
    return value


@dataclass(frozen=True)
class UrlIdentity:
    canonical_url: str
    source_place_id: str
    source_place_id_type: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    zoom: Optional[float] = None
    place_name_hint: Optional[str] = None


def canonicalize_google_maps_url(url: str) -> str:
    """Remove volatile query params while preserving the path/data identity."""
    url = url.strip()
    parts = urlsplit(url)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k not in TRACKING_QUERY_KEYS
    ]
    query.sort(key=lambda kv: kv[0])
    normalized_path = quote(unquote(parts.path), safe="/:!@,+%-._~")
    scheme = parts.scheme or "https"
    netloc = parts.netloc or "www.google.com"
    return urlunsplit((scheme, netloc, normalized_path, urlencode(query), ""))


def extract_coords(url: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract best available coordinates from a Google Maps URL.

    Place URLs often contain both:
    - @lat,lng,zoom: map viewport center, not always exact place coordinate.
    - !3dLAT!4dLNG: place coordinate embedded in data section, usually better.
    Prefer !3d/!4d when present, and keep zoom from @... if available.
    """
    viewport = COORD_RE.search(url)
    place = PLACE_COORD_RE.search(url)
    zoom = float(viewport.group(3)) if viewport else None
    if place:
        return float(place.group(1)), float(place.group(2)), zoom
    if viewport:
        return float(viewport.group(1)), float(viewport.group(2)), zoom
    return None, None, None


def extract_place_name_hint(url: str) -> Optional[str]:
    match = PLACE_NAME_RE.search(url)
    if not match:
        return None
    return unquote(match.group(1)).replace("+", " ").strip() or None


def derive_url_identity(url: str) -> UrlIdentity:
    canonical_url = canonicalize_google_maps_url(url)
    feature = GOOGLE_FEATURE_RE.search(canonical_url)
    if feature:
        source_place_id = "google_feature_" + feature.group(0).replace(":", "_")
        source_place_id_type = "google_feature_id_from_url"
    else:
        cid = CID_RE.search(canonical_url)
        if cid:
            source_place_id = "google_cid_" + cid.group(1)
            source_place_id_type = "google_cid_from_url"
        else:
            source_place_id = "gmapurl_" + sha1_short(canonical_url, 16)
            source_place_id_type = "synthetic_url_hash"
    lat, lng, zoom = extract_coords(canonical_url)
    return UrlIdentity(
        canonical_url=canonical_url,
        source_place_id=source_place_id,
        source_place_id_type=source_place_id_type,
        lat=lat,
        lng=lng,
        zoom=zoom,
        place_name_hint=extract_place_name_hint(canonical_url),
    )




# -----------------------------------------------------------------------------
# Area resolving and filtering
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AreaFilter:
    """Optional geographic filter for discovered/crawled places.

    bbox format: (min_lat, min_lng, max_lat, max_lng)
    polygons format: list of polygons, where each polygon is a list of (lat, lng)
    points. GeoJSON coordinates are converted from [lng, lat] to (lat, lng).
    """

    bbox: Optional[tuple[float, float, float, float]] = None
    polygons: Optional[list[list[tuple[float, float]]]] = None
    area_name: Optional[str] = None
    source: Optional[str] = None
    cache_path: Optional[str] = None

    @property
    def polygon(self) -> Optional[list[tuple[float, float]]]:
        """Backward-compatible first polygon accessor."""
        if not self.polygons:
            return None
        return self.polygons[0]


def point_in_bbox(lat: float, lng: float, bbox: tuple[float, float, float, float]) -> bool:
    min_lat, min_lng, max_lat, max_lng = bbox
    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng


def point_in_polygon(lat: float, lng: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon. Points are (lat, lng)."""
    if len(polygon) < 3:
        return False
    x = lng
    y = lat
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def identity_matches_area(
    identity: UrlIdentity,
    area: Optional[AreaFilter],
    *,
    allow_unknown_coordinates: bool = False,
) -> bool:
    if area is None:
        return True
    if identity.lat is None or identity.lng is None:
        # Google Maps search-result hrefs often omit coordinates.
        # During live discovery we keep these candidates and validate after opening the place page.
        return allow_unknown_coordinates
    if area.bbox and not point_in_bbox(identity.lat, identity.lng, area.bbox):
        return False
    if area.polygons:
        return any(point_in_polygon(identity.lat, identity.lng, polygon) for polygon in area.polygons)
    return True


def parse_bbox(value: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if not value:
        return None
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be min_lat,min_lng,max_lat,max_lng")
    min_lat, min_lng, max_lat, max_lng = parts
    if min_lat > max_lat or min_lng > max_lng:
        raise ValueError("--bbox min values must be <= max values")
    return min_lat, min_lng, max_lat, max_lng


def slugify_area_name(area_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", area_name)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return ascii_text[:120] or "area"


def _geometry_from_geojson(data: dict) -> dict:
    geom = data.get("geometry", data)
    if geom.get("type") == "FeatureCollection":
        features = geom.get("features") or []
        if not features:
            raise ValueError("GeoJSON FeatureCollection has no features")
        geom = features[0].get("geometry", {})
    elif geom.get("type") == "Feature":
        geom = geom.get("geometry", {})
    return geom


def polygons_from_geojson(data: dict) -> list[list[tuple[float, float]]]:
    """Return exterior rings as polygons. Supports Polygon and MultiPolygon."""
    geom = _geometry_from_geojson(data)
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return []

    polygons: list[list[tuple[float, float]]] = []
    if gtype == "Polygon":
        exterior = coords[0]
        polygons.append([(float(lat), float(lng)) for lng, lat in exterior])
    elif gtype == "MultiPolygon":
        for polygon_coords in coords:
            if polygon_coords:
                exterior = polygon_coords[0]
                polygons.append([(float(lat), float(lng)) for lng, lat in exterior])
    else:
        raise ValueError(f"Unsupported GeoJSON geometry type for area polygon: {gtype}")
    return polygons


def safe_polygons_from_geojson(data: dict) -> list[list[tuple[float, float]]]:
    geom = _geometry_from_geojson(data)
    if geom.get("type") not in {"Polygon", "MultiPolygon"}:
        return []
    return polygons_from_geojson(data)


def load_polygons_geojson(path: Optional[Path]) -> list[list[tuple[float, float]]]:
    if not path:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return polygons_from_geojson(data)


def compute_bbox_from_polygons(polygons: list[list[tuple[float, float]]]) -> Optional[tuple[float, float, float, float]]:
    points = [pt for polygon in polygons for pt in polygon]
    if not points:
        return None
    lats = [pt[0] for pt in points]
    lngs = [pt[1] for pt in points]
    return min(lats), min(lngs), max(lats), max(lngs)


def _bbox_from_nominatim_result(result: dict) -> Optional[tuple[float, float, float, float]]:
    raw = result.get("boundingbox")
    if not raw or len(raw) != 4:
        return None
    # Nominatim order: [south, north, west, east]
    south, north, west, east = [float(x) for x in raw]
    return south, west, north, east


def _bbox_from_cached_geojson(data: dict, polygons: list[list[tuple[float, float]]]) -> Optional[tuple[float, float, float, float]]:
    props = data.get("properties", {}) if isinstance(data, dict) else {}
    raw_bbox = props.get("bbox") or data.get("bbox") if isinstance(data, dict) else None
    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        return tuple(float(x) for x in raw_bbox)  # type: ignore[return-value]
    return compute_bbox_from_polygons(polygons)


def nominatim_search_area(area_name: str, *, timeout_s: int = 20, user_agent: str = "absa-restaurant-crawler/0.1") -> dict:
    """Resolve an area name using Nominatim and return the first result.

    This function is intentionally small and cache-oriented. For repeated crawls,
    use --area-cache so the public geocoder is hit at most once per area name.
    """
    query = urlencode({
        "format": "jsonv2",
        "q": area_name,
        "limit": "1",
        "polygon_geojson": "1",
        "addressdetails": "1",
    })
    url = f"https://nominatim.openstreetmap.org/search?{query}"
    req = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not resolve --area-name via Nominatim. Provide --bbox, --area-polygon, "
            "or pre-populate --area-cache with a cached GeoJSON file. "
            f"Original error: {exc}"
        ) from exc
    results = json.loads(payload)
    if not results:
        raise RuntimeError(f"Nominatim returned no result for area_name={area_name!r}")
    return results[0]


def resolve_area_name(
    area_name: str,
    *,
    area_cache_dir: Path,
    allow_network: bool = True,
    user_agent: str = "absa-restaurant-crawler/0.1",
) -> AreaFilter:
    """Resolve area name to polygon+bbox and cache it as GeoJSON Feature.

    Cache filename is a slug of the area name, e.g.
    data/area_cache/phuong_hang_trong_hoan_kiem_ha_noi_viet_nam.geojson
    """
    area_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = area_cache_dir / f"{slugify_area_name(area_name)}.geojson"

    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        polygons = safe_polygons_from_geojson(data)
        bbox = _bbox_from_cached_geojson(data, polygons)
        return AreaFilter(
            bbox=bbox,
            polygons=polygons or None,
            area_name=area_name,
            source="cache_geojson",
            cache_path=str(cache_path),
        )

    if not allow_network:
        raise RuntimeError(
            f"Area cache not found for {area_name!r}: {cache_path}. "
            "Network resolution is disabled. Provide --bbox, --area-polygon, or create this cache file."
        )

    result = nominatim_search_area(area_name, user_agent=user_agent)
    geom = result.get("geojson")
    bbox = _bbox_from_nominatim_result(result)
    if geom:
        polygons = safe_polygons_from_geojson(geom)
        if bbox is None:
            bbox = compute_bbox_from_polygons(polygons)
    else:
        polygons = []

    feature = {
        "type": "Feature",
        "properties": {
            "area_name": area_name,
            "display_name": result.get("display_name"),
            "osm_type": result.get("osm_type"),
            "osm_id": result.get("osm_id"),
            "class": result.get("class"),
            "type": result.get("type"),
            "bbox": list(bbox) if bbox else None,
            "source": "openstreetmap_nominatim",
        },
        "geometry": geom or {"type": "GeometryCollection", "geometries": []},
    }
    cache_path.write_text(json.dumps(feature, ensure_ascii=False, indent=2), encoding="utf-8")

    return AreaFilter(
        bbox=bbox,
        polygons=polygons or None,
        area_name=area_name,
        source="openstreetmap_nominatim",
        cache_path=str(cache_path),
    )


def build_area_filter(
    bbox_value: Optional[str] = None,
    polygon_path: Optional[Path] = None,
    area_name: Optional[str] = None,
    area_cache_dir: Path = Path("data/area_cache"),
    allow_area_network: bool = True,
    user_agent: str = "absa-restaurant-crawler/0.1",
) -> Optional[AreaFilter]:
    explicit_bbox = parse_bbox(bbox_value)
    polygons: list[list[tuple[float, float]]] = []
    area_source: Optional[str] = None
    cache_path: Optional[str] = None

    auto_bbox: Optional[tuple[float, float, float, float]] = None
    if area_name:
        resolved = resolve_area_name(
            area_name,
            area_cache_dir=area_cache_dir,
            allow_network=allow_area_network,
            user_agent=user_agent,
        )
        auto_bbox = resolved.bbox
        if resolved.polygons:
            polygons.extend(resolved.polygons)
        area_source = resolved.source
        cache_path = resolved.cache_path

    if polygon_path:
        file_polygons = load_polygons_geojson(polygon_path)
        polygons.extend(file_polygons)
        area_source = (area_source + "+file_polygon") if area_source else "file_polygon"

    bbox = explicit_bbox or auto_bbox or compute_bbox_from_polygons(polygons)
    if bbox or polygons:
        return AreaFilter(
            bbox=bbox,
            polygons=polygons or None,
            area_name=area_name,
            source=area_source,
            cache_path=cache_path,
        )
    return None

VALID_MODES = {"collection", "benchmark"}
VALID_ROLES = {"target", "peer", "candidate"}


def normalize_role(role: Optional[str], *, mode: str = "collection", default_role: Optional[str] = None) -> str:
    """Resolve missing/invalid role according to crawler mode.

    collection mode defaults to candidate.
    benchmark mode defaults to peer, while explicit target remains target.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    resolved = clean_text(role).lower() if role else ""
    if not resolved:
        resolved = default_role or ("candidate" if mode == "collection" else "peer")
    if resolved not in VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {role!r}")
    return resolved


def make_restaurant_id(role: str, source_place_id: str) -> str:
    role = normalize_role(role, mode="collection", default_role="candidate")
    if role == "target":
        return "res_target_" + sha1_short(source_place_id, 8)
    if role == "peer":
        return "res_peer_" + sha1_short(source_place_id, 8)
    return "res_candidate_" + sha1_short(source_place_id, 8)


def choose_restaurant_id(
    *,
    role: str,
    source_place_id: str,
    provided_restaurant_id: Optional[str] = None,
    target_restaurant_id: Optional[str] = None,
) -> str:
    """Choose a stable restaurant_id.

    - Explicit restaurant_id always wins.
    - In benchmark mode, a target row may inherit --target-restaurant-id.
    - Otherwise generate a deterministic ID from role + source_place_id.
    """
    if provided_restaurant_id:
        return provided_restaurant_id
    if role == "target" and target_restaurant_id:
        return target_restaurant_id
    return make_restaurant_id(role, source_place_id)


# -----------------------------------------------------------------------------
# Relative time parser
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedReviewTime:
    review_time: datetime
    review_month: str
    confidence: str  # high | medium | low
    original_text: str


VI_NUMBERS = {
    "một": 1,
    "mot": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "bon": 4,
    "năm": 5,
    "nam": 5,
    "sáu": 6,
    "sau": 6,
    "bảy": 7,
    "bay": 7,
    "tám": 8,
    "tam": 8,
    "chín": 9,
    "chin": 9,
    "mười": 10,
    "muoi": 10,
}

EN_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

UNIT_DAYS = {
    "ngày": 1,
    "ngay": 1,
    "tuần": 7,
    "tuan": 7,
    "tháng": 30,
    "thang": 30,
    "năm": 365,
    "nam": 365,
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30,
    "months": 30,
    "year": 365,
    "years": 365,
}


def _number_from_token(token: str) -> Optional[int]:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return VI_NUMBERS.get(token) or EN_NUMBERS.get(token)


def _ensure_tz(dt: datetime, default_tz: timezone = timezone.utc) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=default_tz)
    return dt


def parse_relative_review_time(text: str, crawl_time: datetime) -> ParsedReviewTime:
    """Parse Google Maps relative review time.

    Live Google Maps cards often contain many numbers before the actual date
    (for example reviewer stats like "12 bài đánh giá").  Older versions of
    this crawler used the first "number + word" pair and therefore missed the
    real review date. This parser searches for known time units only and scans
    all matches in the card text.
    """
    crawl_time = _ensure_tz(crawl_time)
    original = text or ""
    normalized = clean_text(original).lower()

    if not normalized:
        review_time = crawl_time
        return ParsedReviewTime(review_time, review_time.strftime("%Y-%m"), "unknown", original)

    # Absolute dates first when present.
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"]:
        try:
            review_time = datetime.strptime(normalized.strip(), fmt).replace(tzinfo=crawl_time.tzinfo)
            return ParsedReviewTime(review_time, review_time.strftime("%Y-%m"), "high", original)
        except ValueError:
            pass

    if any(token in normalized for token in ["hôm qua", "hom qua", "yesterday"]):
        review_time = crawl_time - timedelta(days=1)
        return ParsedReviewTime(review_time, review_time.strftime("%Y-%m"), "medium", original)

    if any(token in normalized for token in ["vừa xong", "vua xong", "just now"]):
        review_time = crawl_time
        return ParsedReviewTime(review_time, review_time.strftime("%Y-%m"), "medium", original)

    num_pattern = (
        r"\d+|một|mot|hai|ba|bốn|bon|năm|nam|sáu|sau|bảy|bay|tám|tam|"
        r"chín|chin|mười|muoi|a|an|one|two|three|four|five|six|seven|eight|nine|ten"
    )
    # Include minute/hour units, but treat them as same-day.
    unit_days = dict(UNIT_DAYS)
    unit_days.update({
        "phút": 0, "phut": 0, "minute": 0, "minutes": 0,
        "giờ": 0, "gio": 0, "hour": 0, "hours": 0,
    })
    unit_pattern = "|".join(sorted((re.escape(u) for u in unit_days), key=len, reverse=True))
    rel_re = re.compile(
        rf"(?P<num>{num_pattern})\s+(?P<unit>{unit_pattern})(?:\s*(?:trước|ago))?",
        re.IGNORECASE,
    )

    for match in rel_re.finditer(normalized):
        n = _number_from_token(match.group("num")) or 1
        unit = match.group("unit").lower()
        days_per_unit = unit_days.get(unit)
        if days_per_unit is None:
            continue
        review_time = crawl_time - timedelta(days=n * days_per_unit)
        if unit in {"ngày", "ngay", "day", "days", "tuần", "tuan", "week", "weeks"}:
            confidence = "medium"
        elif days_per_unit == 0:
            confidence = "medium"
        else:
            confidence = "low"
        return ParsedReviewTime(review_time, review_time.strftime("%Y-%m"), confidence, original)

    review_time = crawl_time
    return ParsedReviewTime(review_time, review_time.strftime("%Y-%m"), "unknown", original)


# -----------------------------------------------------------------------------
# Static/snapshot HTML parser
# -----------------------------------------------------------------------------

class _ReviewCardParser(HTMLParser):
    """Parser for saved/synthetic snapshots.

    Expected simple shape:
      <div class="review-card" data-review-id="...">
        <span class="reviewer">...</span>
        <span class="rating">2 sao</span>
        <span class="time">3 tuần trước</span>
        <span class="text">...</span>
      </div>

    This is deliberately test-friendly. Tune live Playwright selectors separately.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cards: List[Dict[str, str]] = []
        self.current: Optional[Dict[str, str]] = None
        self.capture_field: Optional[str] = None
        self.buffer: List[str] = []
        self.card_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_d = {k: v or "" for k, v in attrs}
        classes = attrs_d.get("class", "").split()
        if tag == "div" and "review-card" in classes:
            self.current = {}
            self.card_depth = 1
            if attrs_d.get("data-review-id"):
                self.current["native_review_id"] = attrs_d["data-review-id"]
            return

        if self.current is not None:
            if tag == "div":
                self.card_depth += 1
            for field_name in ["reviewer", "rating", "time", "text"]:
                if field_name in classes:
                    self.capture_field = field_name
                    self.buffer = []
                    return

    def handle_data(self, data: str) -> None:
        if self.capture_field:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is not None and self.capture_field:
            self.current[self.capture_field] = clean_text(" ".join(self.buffer))
            self.capture_field = None
            self.buffer = []
            return

        if self.current is not None and tag == "div":
            self.card_depth -= 1
            if self.card_depth <= 0:
                if any(k in self.current for k in ["text", "rating", "time"]):
                    self.cards.append(self.current)
                self.current = None
                self.card_depth = 0


def parse_rating(value: str) -> Optional[int]:
    value = value or ""
    match = RATING_RE.search(value)
    if not match:
        return None
    rating = int(match.group(1))
    if 1 <= rating <= 5:
        return rating
    return None


def parse_review_cards_from_html(html: str) -> List[dict]:
    parser = _ReviewCardParser()
    parser.feed(html)
    reviews = []
    for card in parser.cards:
        reviews.append(
            {
                "native_review_id": card.get("native_review_id"),
                "reviewer_name": card.get("reviewer"),
                "rating": parse_rating(card.get("rating", "")),
                "relative_time_text": card.get("time"),
                "review_text": card.get("text"),
            }
        )
    return reviews


# -----------------------------------------------------------------------------
# Normalize, dedup, validate
# -----------------------------------------------------------------------------

def make_source_review_id(
    *,
    source_place_id: str,
    review_text: str,
    rating: Optional[int],
    reviewer_name: Optional[str] = None,
    review_month: Optional[str] = None,
    native_review_id: Optional[str] = None,
) -> tuple[str, str]:
    if native_review_id:
        return native_review_id, "native"
    payload = "|".join(
        [
            source_place_id,
            normalize_for_hash(reviewer_name),
            normalize_for_hash(review_text),
            "" if rating is None else str(rating),
            review_month or "",
        ]
    )
    return sha1_short(payload, 16), "synthetic_hash"


def make_review_id(source: str, restaurant_id: str, source_review_id: str) -> str:
    source_prefix = re.sub(r"[^a-zA-Z0-9]+", "_", source).strip("_") or "source"
    return f"{source_prefix}_{restaurant_id}_{source_review_id}"


def normalize_review(raw: Dict[str, Any]) -> Dict[str, Any]:
    review_text = clean_text(raw.get("review_text"))
    rating = raw.get("rating")
    if rating is not None:
        rating = int(rating)

    review_time = raw.get("review_time")
    if isinstance(review_time, datetime):
        review_time_str = review_time.isoformat()
    elif review_time:
        review_time_str = str(review_time)
    else:
        review_time_str = None

    review_month = raw.get("review_month") or (review_time_str[:7] if review_time_str else None)
    source = raw.get("source") or "google_maps_url_crawler"
    source_place_id = raw["source_place_id"]

    source_review_id, source_review_id_type = make_source_review_id(
        source_place_id=source_place_id,
        review_text=review_text,
        rating=rating,
        reviewer_name=raw.get("reviewer_name"),
        review_month=review_month,
        native_review_id=raw.get("native_review_id") or raw.get("source_review_id"),
    )

    return {
        "review_id": make_review_id(source, raw["restaurant_id"], source_review_id),
        "review_text": review_text,
        "restaurant_id": raw["restaurant_id"],
        "restaurant_name": clean_text(raw.get("restaurant_name")),
        "rating": rating,
        "review_time": review_time_str,
        "review_month": review_month,
        "source": source,
        "source_place_id": source_place_id,
        "source_review_id": source_review_id,
        "language": raw.get("language") or "vi",
        # Audit-only fields. write_jsonl drops fields starting with "_" by default.
        "_source_review_id_type": source_review_id_type,
        "_time_confidence": raw.get("time_confidence"),
    }


def fallback_dedup_key(review: dict) -> str:
    payload = "|".join(
        [
            review.get("source_place_id", ""),
            normalize_for_hash(review.get("review_text")),
            "" if review.get("rating") is None else str(review.get("rating")),
            review.get("review_time") or review.get("review_month") or "",
        ]
    )
    return sha1_short(payload, 20)


def dedup_reviews(reviews: Iterable[dict]) -> List[dict]:
    seen: Dict[str, dict] = {}
    for review in reviews:
        if review.get("source_place_id") and review.get("source_review_id"):
            key = f"primary::{review['source_place_id']}::{review['source_review_id']}"
        else:
            key = "fallback::" + fallback_dedup_key(review)
        if key not in seen:
            seen[key] = review
    return list(seen.values())


class ValidationError(Exception):
    pass


def validate_reviews_jsonl_objects(
    reviews: Iterable[dict],
    *,
    crawl_month: Optional[str] = None,
    mode: str = "collection",
    require_target_restaurant_id: Optional[str] = None,
    crawled_restaurant_ids: Optional[Iterable[str]] = None,
    min_peers: int = 0,
    min_restaurants: int = 1,
) -> None:
    """Validate LocalJsonlAdapter rows with mode-aware checks.

    collection mode:
      - no target restaurant required.
      - validates schema, uniqueness, month, and minimum restaurant count.

    benchmark mode:
      - target restaurant required.
      - peers are all restaurant_ids different from the target.
      - validates min_peers.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    reviews = list(reviews)
    errors: List[str] = []
    review_ids = set()
    source_keys = set()
    restaurant_ids = set()
    crawled_ids = {rid for rid in (crawled_restaurant_ids or []) if rid}

    for i, row in enumerate(reviews, start=1):
        prefix = f"review[{i}]"
        for field_name in [
            "review_id",
            "review_text",
            "restaurant_id",
            "source",
            "source_place_id",
            "source_review_id",
            "review_month",
        ]:
            if not row.get(field_name):
                errors.append(f"{prefix}: missing/empty {field_name}")

        if row.get("review_id") in review_ids:
            errors.append(f"{prefix}: duplicate review_id {row.get('review_id')}")
        review_ids.add(row.get("review_id"))

        source_key = (row.get("source"), row.get("source_place_id"), row.get("source_review_id"))
        if source_key in source_keys:
            errors.append(f"{prefix}: duplicate source key {source_key}")
        source_keys.add(source_key)

        rating = row.get("rating")
        if rating is not None and rating not in {1, 2, 3, 4, 5}:
            errors.append(f"{prefix}: rating must be null or 1..5, got {rating!r}")

        month = row.get("review_month")
        if month and not MONTH_RE.match(month):
            errors.append(f"{prefix}: review_month must be YYYY-MM, got {month!r}")
        if crawl_month and month != crawl_month:
            errors.append(f"{prefix}: review_month {month!r} != crawl_month {crawl_month!r}")

        restaurant_id = row.get("restaurant_id")
        if restaurant_id:
            restaurant_ids.add(restaurant_id)

    validation_restaurant_ids = restaurant_ids | crawled_ids

    if min_restaurants and len(validation_restaurant_ids) < min_restaurants:
        errors.append(f"need at least {min_restaurants} restaurants, got {len(validation_restaurant_ids)}")

    if mode == "benchmark":
        if not require_target_restaurant_id:
            errors.append("benchmark mode requires a target restaurant_id; pass --target-restaurant-id or mark one input row as role=target")
        elif require_target_restaurant_id not in validation_restaurant_ids:
            errors.append(f"missing target restaurant_id {require_target_restaurant_id!r}")
        if require_target_restaurant_id:
            peer_ids = {rid for rid in validation_restaurant_ids if rid != require_target_restaurant_id}
        else:
            peer_ids = validation_restaurant_ids
        if min_peers and len(peer_ids) < min_peers:
            errors.append(f"need at least {min_peers} peer restaurants, got {len(peer_ids)}")

    if errors:
        raise ValidationError("\n".join(errors))


# -----------------------------------------------------------------------------
# Offline snapshot pipeline
# -----------------------------------------------------------------------------

def parse_snapshot_to_adapter_reviews(
    *,
    html: str,
    google_maps_url: str,
    role: str,
    crawl_time: datetime,
    crawl_month: str,
    restaurant_id: Optional[str] = None,
    restaurant_name: Optional[str] = None,
    target_restaurant_id: Optional[str] = None,
    area_filter: Optional[AreaFilter] = None,
) -> List[dict]:
    identity = derive_url_identity(google_maps_url)
    if not identity_matches_area(identity, area_filter):
        return []
    rid = choose_restaurant_id(
        role=role,
        source_place_id=identity.source_place_id,
        provided_restaurant_id=restaurant_id,
        target_restaurant_id=target_restaurant_id,
    )
    name = restaurant_name or identity.place_name_hint or ""

    rows = []
    for raw_card in parse_review_cards_from_html(html):
        parsed_time = parse_relative_review_time(raw_card.get("relative_time_text") or "", crawl_time)
        if parsed_time.review_month != crawl_month:
            continue
        rows.append(
            normalize_review(
                {
                    "restaurant_id": rid,
                    "restaurant_name": name,
                    "source": "google_maps_url_crawler",
                    "source_place_id": identity.source_place_id,
                    "native_review_id": raw_card.get("native_review_id"),
                    "reviewer_name": raw_card.get("reviewer_name"),
                    "rating": raw_card.get("rating"),
                    "review_text": raw_card.get("review_text"),
                    "review_time": parsed_time.review_time,
                    "review_month": parsed_time.review_month,
                    "time_confidence": parsed_time.confidence,
                    "language": "vi",
                }
            )
        )
    return rows


def infer_target_restaurant_id_from_items(
    items: Iterable[dict],
    *,
    mode: str,
    default_role: Optional[str] = None,
    explicit_target_restaurant_id: Optional[str] = None,
) -> Optional[str]:
    if explicit_target_restaurant_id:
        return explicit_target_restaurant_id
    if mode != "benchmark":
        return None
    for item in items:
        role = normalize_role(item.get("role"), mode=mode, default_role=default_role)
        if role == "target":
            identity = derive_url_identity(item["google_maps_url"])
            return choose_restaurant_id(
                role=role,
                source_place_id=identity.source_place_id,
                provided_restaurant_id=item.get("restaurant_id"),
                target_restaurant_id=None,
            )
    return None


def offline_build_jsonl_objects(
    snapshots: Iterable[dict],
    *,
    crawl_time: datetime,
    crawl_month: str,
    mode: str = "collection",
    default_role: Optional[str] = None,
    target_restaurant_id: Optional[str] = None,
    min_peers: int = 0,
    min_restaurants: int = 1,
    area_filter: Optional[AreaFilter] = None,
) -> List[dict]:
    snapshots = list(snapshots)
    effective_target_id = infer_target_restaurant_id_from_items(
        snapshots,
        mode=mode,
        default_role=default_role,
        explicit_target_restaurant_id=target_restaurant_id,
    )

    all_rows: List[dict] = []
    for snap in snapshots:
        role = normalize_role(snap.get("role"), mode=mode, default_role=default_role)
        all_rows.extend(
            parse_snapshot_to_adapter_reviews(
                html=snap["html"],
                google_maps_url=snap["google_maps_url"],
                role=role,
                crawl_time=crawl_time,
                crawl_month=crawl_month,
                restaurant_id=snap.get("restaurant_id"),
                restaurant_name=snap.get("restaurant_name"),
                target_restaurant_id=effective_target_id,
                area_filter=area_filter,
            )
        )
    rows = dedup_reviews(all_rows)
    validate_reviews_jsonl_objects(
        rows,
        crawl_month=crawl_month,
        mode=mode,
        require_target_restaurant_id=effective_target_id,
        min_peers=min_peers,
        min_restaurants=min_restaurants,
    )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict], *, drop_audit_fields: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if drop_audit_fields:
                row = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Optional Playwright live crawler skeleton
# -----------------------------------------------------------------------------

@dataclass
class CrawlInput:
    google_maps_url: str
    role: Optional[str] = None
    restaurant_id: Optional[str] = None
    restaurant_name: Optional[str] = None
    # Internal hint captured during area discovery. Useful because some Google Maps
    # result hrefs contain a good feature id but later redirect to a viewport
    # placeholder URL when opened directly. Input JSONL can omit this.
    source_place_id_hint: Optional[str] = None


@dataclass
class DiscoveryConfig:
    enabled: bool = False
    search_queries: list[str] = field(default_factory=lambda: ["nhà hàng", "quán ăn", "restaurant"])
    max_places: int = 80
    target_url: Optional[str] = None
    target_restaurant_name: Optional[str] = None
    click_search_this_area: bool = True


@dataclass
class LiveCrawlerConfig:
    crawl_month: str
    crawl_time: datetime
    input_urls_jsonl: Optional[Path]
    output_jsonl: Path
    mode: str = "collection"
    default_role: Optional[str] = None
    target_restaurant_id: Optional[str] = None
    min_peers: int = 0
    min_restaurants: int = 1
    locale: str = "vi-VN"
    headless: bool = True
    max_reviews_per_restaurant: int = 200
    stop_after_old_reviews: int = 20
    include_unknown_time: bool = False
    include_nonmatching_month: bool = False
    debug_expand_dom: bool = False
    debug_expand_dom_limit: int = 5
    area_filter: Optional[AreaFilter] = None
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    selectors: dict[str, list[str]] = field(
        default_factory=lambda: {
            "reviews_button": [
                'button[role="tab"]:has-text("Bài đánh giá")',
                'button[role="tab"]:has-text("Reviews")',
                'div[role="tab"]:has-text("Bài đánh giá")',
                'div[role="tab"]:has-text("Reviews")',
                'button[aria-label*="Bài đánh giá"]',
                'button[aria-label*="Reviews"]',
                'button:has-text("Bài đánh giá"):not(:has-text("Viết"))',
                'button:has-text("reviews"):not(:has-text("Write"))',
                'button:has-text("Reviews"):not(:has-text("Write"))',
            ],
            "sort_button": [
                'button[aria-label*="Sắp xếp bài đánh giá"]',
                'button[aria-label*="Sort reviews"]',
                'button[aria-label*="Sắp xếp"]',
                'button[aria-label*="Sort"]',
                'button:has-text("Sắp xếp")',
                'button:has-text("Sort")',
                'button[aria-label*="Liên quan nhất"]',
                'button[aria-label*="Có liên quan nhất"]',
                'button[aria-label*="Most relevant"]',
                'button[aria-label*="Mới nhất"]',
                'button[aria-label*="Newest"]',
                'div[role="button"][aria-label*="Sắp xếp"]',
                'div[role="button"][aria-label*="Sort"]',
                'div[role="button"]:has-text("Sắp xếp")',
                'div[role="button"]:has-text("Sort")',
                'div[role="combobox"]:has-text("Liên quan nhất")',
                'div[role="combobox"]:has-text("Most relevant")',
            ],
            "newest_option": [
                'text="Mới nhất"',
                'text="Newest"',
                'div[role="menuitemradio"]:has-text("Mới nhất")',
                'div[role="menuitemradio"]:has-text("Newest")',
                'div[role="menuitem"]:has-text("Mới nhất")',
                'div[role="menuitem"]:has-text("Newest")',
                'div[aria-label*="Mới nhất"]',
                'div[aria-label*="Newest"]',
            ],
            # Common Google Maps review-card selectors. These are still UI-dependent.
            "review_card": [
                'div[data-review-id]',
                'div[data-reviewer-id]',
                'div.jftiEf',
                'div.jJc9Ad',
                'div[role="article"]:has(span[role="img"][aria-label*="sao"])',
                'div[role="article"]:has(span[role="img"][aria-label*="star"])',
                'div[jscontroller="e6Mltc"]',
                'div[jsaction*="review"]',
                'xpath=//div[@data-review-id or @data-reviewer-id]',
                'xpath=//div[contains(concat(" ", normalize-space(@class), " "), " jftiEf ") or contains(concat(" ", normalize-space(@class), " "), " jJc9Ad ")]',
                'xpath=//div[.//span[@role="img" and (contains(@aria-label,"sao") or contains(@aria-label,"star"))] and .//*[contains(text(),"trước") or contains(text(),"ago")]][not(.//div[.//span[@role="img" and (contains(@aria-label,"sao") or contains(@aria-label,"star"))] and .//*[contains(text(),"trước") or contains(text(),"ago")]])]',
                'xpath=//div[.//span[@role="img" and (contains(@aria-label,"sao") or contains(@aria-label,"star"))] and (.//span[contains(@class,"rsqaWe")] or .//*[contains(text(),"trước") or contains(text(),"ago")])]',
            ],
            "review_text": [
                'span.wiI7pd',
                'span[lang]',
                'div[class*="review"] span',
            ],
            "review_time": [
                'span.rsqaWe',
                'span[class*="rsqaWe"]',
            ],
            "reviewer_name": [
                'div.d4r55',
                'div[class*="d4r55"]',
            ],
            "rating_node": [
                'span.kvMYJc[role="img"]',
                'span[role="img"][aria-label*="sao"]',
                'span[role="img"][aria-label*="star"]',
            ],
        }
    )


def read_input_urls(path: Path) -> list[CrawlInput]:
    items: list[CrawlInput] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            items.append(CrawlInput(**data))
    return items


# -----------------------------------------------------------------------------
# Area discovery: search result URL extraction
# -----------------------------------------------------------------------------

class _PlaceResultParser(HTMLParser):
    """Small parser for saved/synthetic Google Maps search snapshots.

    Test-friendly fixture formats supported:
      <a class="place-result" href="https://www.google.com/maps/place/..." data-name="Peer A">Peer A</a>
      <div class="place-result" data-url="https://www.google.com/maps/place/..." data-name="Peer A"></div>

    Live Google Maps search uses Playwright selectors separately because the DOM is
    highly dynamic and changes frequently.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self.current: Optional[dict] = None
        self.capture_text = False
        self.buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_d = {k: v or "" for k, v in attrs}
        classes = attrs_d.get("class", "").split()
        href = attrs_d.get("href") or attrs_d.get("data-url") or ""
        if "place-result" in classes or (tag == "a" and "/maps/place/" in href):
            url = href
            if not url:
                return
            self.current = {
                "google_maps_url": url,
                "restaurant_name": attrs_d.get("data-name") or attrs_d.get("aria-label") or "",
            }
            self.capture_text = tag == "a"
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.current is not None and self.capture_text:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is not None and self.capture_text and tag == "a":
            if not self.current.get("restaurant_name"):
                self.current["restaurant_name"] = clean_text(" ".join(self.buffer))
            self.results.append(self.current)
            self.current = None
            self.capture_text = False
            self.buffer = []
        elif self.current is not None and tag == "div":
            self.results.append(self.current)
            self.current = None
            self.capture_text = False
            self.buffer = []


def parse_place_results_from_html(html: str, *, area_filter: Optional[AreaFilter] = None, max_places: int = 80) -> list[CrawlInput]:
    parser = _PlaceResultParser()
    parser.feed(html)
    out: list[CrawlInput] = []
    seen: set[str] = set()
    for raw in parser.results:
        url = clean_text(raw.get("google_maps_url"))
        if not url:
            continue
        identity = derive_url_identity(url)
        if not identity_matches_area(identity, area_filter):
            continue
        if identity.source_place_id in seen:
            continue
        seen.add(identity.source_place_id)
        out.append(CrawlInput(
            google_maps_url=identity.canonical_url,
            restaurant_name=clean_text(raw.get("restaurant_name")) or identity.place_name_hint,
        ))
        if len(out) >= max_places:
            break
    return out


def build_google_maps_search_url(query: str, area_filter: Optional[AreaFilter] = None, *, zoom: int = 16) -> str:
    q = quote(query.strip())
    if area_filter and area_filter.bbox:
        min_lat, min_lng, max_lat, max_lng = area_filter.bbox
        lat = (min_lat + max_lat) / 2
        lng = (min_lng + max_lng) / 2
        return f"https://www.google.com/maps/search/{q}/@{lat:.7f},{lng:.7f},{zoom}z?hl=vi&gl=VN"
    return f"https://www.google.com/maps/search/{q}?hl=vi&gl=VN"


def merge_crawl_inputs(*groups: Iterable[CrawlInput]) -> list[CrawlInput]:
    """Deduplicate inputs by source_place_id while preserving order.

    If discovery captured a feature id before a URL redirects to a placeholder,
    source_place_id_hint is preferred as the dedup key.
    """
    seen: set[str] = set()
    merged: list[CrawlInput] = []
    for group in groups:
        for item in group:
            identity = derive_url_identity(item.google_maps_url)
            key = item.source_place_id_hint or identity.source_place_id
            if key in seen:
                continue
            seen.add(key)
            item.google_maps_url = identity.canonical_url
            if item.source_place_id_hint is None:
                item.source_place_id_hint = identity.source_place_id
            merged.append(item)
    return merged


def snapshots_from_discovered_inputs(inputs: Iterable[CrawlInput], *, mode: str, target_restaurant_id: Optional[str] = None) -> list[dict]:
    """Build offline review snapshots from discovered inputs.

    This exists only for deterministic offline testing/demo. Live mode opens each URL.
    """
    snaps: list[dict] = []
    for idx, item in enumerate(inputs):
        role = normalize_role(item.role, mode=mode)
        restaurant_id = item.restaurant_id
        if role == "target" and target_restaurant_id:
            restaurant_id = restaurant_id or target_restaurant_id
        snaps.append({
            "html": SAMPLE_HTML,
            "google_maps_url": item.google_maps_url,
            "role": role,
            "restaurant_id": restaurant_id,
            "restaurant_name": item.restaurant_name,
        })
    return snaps



def is_probably_placeholder_place_url(url: str) -> bool:
    """Return True for Google Maps placeholder URLs such as /maps/place//,@lat,lng.

    Those URLs represent the current map viewport, not a restaurant. In your screenshot,
    the browser opened exactly this kind of URL, which means discovery accepted a bad href.
    """
    if not url:
        return True
    parts = urlsplit(url)
    path = unquote(parts.path)
    if "/maps/place//" in path or re.search(r"/maps/place/\s*[,/@]", path):
        return True
    identity = derive_url_identity(url)
    if identity.source_place_id_type == "synthetic_url_hash" and not identity.place_name_hint:
        return True
    hint = (identity.place_name_hint or "").strip(" ,/@")
    if hint in {"", ","}:
        return True
    return False


def is_usable_discovered_place_url(url: str) -> bool:
    """Keep real place URLs and reject viewport/category placeholder links."""
    if not url or "/maps/place/" not in url:
        return False
    if is_probably_placeholder_place_url(url):
        return False
    identity = derive_url_identity(url)
    # Prefer URLs with Google feature id/cid. A non-empty place-name hint is acceptable
    # because clicking/opening the place may later resolve to a richer URL.
    if identity.source_place_id_type in {"google_feature_id_from_url", "google_cid_from_url"}:
        return True
    return bool(identity.place_name_hint)


async def _first_attr_available(root, selectors: list[str], attr_name: str) -> Optional[str]:
    for selector in selectors:
        try:
            locator = root.locator(selector).first
            value = await locator.get_attribute(attr_name, timeout=1000)
            if value:
                return clean_text(value)
        except Exception:
            continue
    return None


async def _locator_count_first_available(page, selectors: list[str]):
    """Return (locator, count, selector) for the first selector with matches."""
    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = await locator.count()
            if count > 0:
                return locator, count, selector
        except Exception:
            continue
    return page.locator("__never_matches__"), 0, ""


async def _scroll_results_panel_or_page(page) -> None:
    """Scroll the active Google Maps result/review panel when present; otherwise scroll the page."""
    for selector in [
        'div[role="feed"]',
        'div[aria-label*="Kết quả"]',
        'div[aria-label*="Results"]',
        'div[aria-label*="Bài đánh giá"]',
        'div[aria-label*="bài đánh giá"]',
        'div[aria-label*="Reviews"]',
        'div.m6QErb[tabindex="-1"]',
        'div.m6QErb',
    ]:
        try:
            loc = page.locator(selector).first
            await loc.wait_for(timeout=800)
            await loc.evaluate("el => { el.scrollTop = el.scrollTop + 2500; }")
            return
        except Exception:
            continue
    await page.mouse.wheel(0, 2500)


async def _resolve_place_href_by_click(page, anchor, search_url: str) -> Optional[tuple[str, str]]:
    """Click a search-result anchor and read the resolved Maps place URL.

    Google Maps often exposes placeholder hrefs in the DOM. The real place URL is only
    available after clicking. This helper clicks, waits briefly, captures page.url,
    then navigates back to the search page.
    """
    try:
        before = page.url
        await anchor.click(timeout=2500)
        await page.wait_for_timeout(1800)
        resolved_url = page.url
        name = ""
        try:
            name = clean_text(await page.locator('h1').first.inner_text(timeout=1000))
        except Exception:
            try:
                name = clean_text(await anchor.inner_text(timeout=1000))
            except Exception:
                name = ""
        if before != resolved_url:
            try:
                await page.go_back(wait_until="domcontentloaded", timeout=10000)
                await page.wait_for_timeout(1000)
            except Exception:
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(1000)
        if is_usable_discovered_place_url(resolved_url):
            return resolved_url, name
    except Exception:
        try:
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1000)
        except Exception:
            pass
    return None


async def discover_live_area_inputs(page, *, area_filter: Optional[AreaFilter], search_queries: list[str], max_places: int, click_search_this_area: bool = True) -> list[CrawlInput]:
    """Discover Google Maps place URLs from search result UI.

    V3 fixes two practical issues:
    - reject placeholder URLs like /maps/place//,@lat,lng (the weird page in your screenshot);
    - when an anchor href is only a placeholder, click it and capture the resolved real place URL.
    """
    discovered: list[CrawlInput] = []
    seen: set[str] = set()
    search_this_area_selectors = [
        'button:has-text("Tìm kiếm khu vực này")',
        'button:has-text("Search this area")',
        'text="Tìm kiếm khu vực này"',
        'text="Search this area"',
    ]

    for query in search_queries:
        if len(discovered) >= max_places:
            break
        search_url = build_google_maps_search_url(query, area_filter)
        print(f"[gmaps-crawler] discovery query={query!r} url={search_url}", file=sys.stderr)
        await page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(3500)
        if click_search_this_area:
            clicked = await _click_first_available(page, search_this_area_selectors, timeout_ms=1500)
            if clicked:
                await page.wait_for_timeout(1800)

        no_progress = 0
        last_count = 0
        for scroll_idx in range(35):
            anchors = page.locator('a[href*="/maps/place/"]')
            count = await anchors.count()
            print(f"[gmaps-crawler] discovery scroll={scroll_idx} anchors={count} kept={len(discovered)}", file=sys.stderr)

            for i in range(count):
                if len(discovered) >= max_places:
                    break
                a = anchors.nth(i)
                try:
                    href = await a.get_attribute("href", timeout=1000)
                except Exception:
                    href = None
                try:
                    anchor_text = clean_text(await a.inner_text(timeout=1000))
                except Exception:
                    anchor_text = ""

                candidate_url: Optional[str] = href
                candidate_name: str = anchor_text

                # Many Google Maps anchors are viewport placeholders. Click to resolve them.
                if not candidate_url or not is_usable_discovered_place_url(candidate_url):
                    resolved = await _resolve_place_href_by_click(page, a, search_url)
                    if not resolved:
                        continue
                    candidate_url, resolved_name = resolved
                    candidate_name = resolved_name or candidate_name

                if not is_usable_discovered_place_url(candidate_url):
                    continue
                identity = derive_url_identity(candidate_url)
                if not identity_matches_area(identity, area_filter, allow_unknown_coordinates=True):
                    continue
                if identity.source_place_id in seen:
                    continue
                seen.add(identity.source_place_id)
                name = candidate_name or identity.place_name_hint or ""
                discovered.append(CrawlInput(
                    google_maps_url=identity.canonical_url,
                    restaurant_name=name,
                    source_place_id_hint=identity.source_place_id,
                ))
                print(f"[gmaps-crawler] discovered name={name!r} id={identity.source_place_id}", file=sys.stderr)

            if len(discovered) >= max_places:
                break
            if len(discovered) == last_count:
                no_progress += 1
                if no_progress >= 6:
                    break
            else:
                no_progress = 0
                last_count = len(discovered)

            await _scroll_results_panel_or_page(page)
            await page.wait_for_timeout(1000)

    return discovered

async def _click_first_available(page, selectors: list[str], timeout_ms: int = 2500) -> bool:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            await locator.wait_for(timeout=timeout_ms)
            try:
                await locator.scroll_into_view_if_needed(timeout=timeout_ms)
            except Exception:
                pass
            await locator.click(timeout=timeout_ms)
            return True
        except Exception:
            try:
                box = await locator.bounding_box(timeout=timeout_ms)
                if box:
                    await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    return True
            except Exception:
                continue
    return False


async def _click_reviews_tab_safely(page, selectors: list[str], timeout_ms: int = 2500) -> bool:
    """Click the real Reviews tab/button, avoiding "Write a review" controls.

    A broad `:has-text("bài đánh giá")` selector can hit "Viết bài đánh giá".
    That click returns success but does not open the review list, which leads to
    sort controls being absent and `visible_cards=0`.
    """
    banned = ["viết", "write", "rating", "xếp hạng của bạn", "your rating"]
    preferred = [
        'button[role="tab"]',
        'div[role="tab"]',
        'button[aria-label]',
        'div[role="button"][aria-label]',
        'button',
        'div[role="button"]',
    ]
    keywords = ["bài đánh giá", "đánh giá", "reviews", "review"]
    for selector in preferred:
        try:
            loc = page.locator(selector)
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 140)):
            el = loc.nth(i)
            try:
                if not await el.is_visible(timeout=250):
                    continue
                text = clean_text(await el.inner_text(timeout=250) or "")
                aria = clean_text(await el.get_attribute("aria-label", timeout=250) or "")
                title = clean_text(await el.get_attribute("title", timeout=250) or "")
            except Exception:
                continue
            haystack = f"{text} {aria} {title}".lower()
            if not any(k in haystack for k in keywords):
                continue
            if any(b in haystack for b in banned):
                continue
            try:
                await el.scroll_into_view_if_needed(timeout=timeout_ms)
            except Exception:
                pass
            try:
                await el.click(timeout=timeout_ms)
                return True
            except Exception:
                try:
                    box = await el.bounding_box(timeout=timeout_ms)
                    if box:
                        await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                        return True
                except Exception:
                    continue
    return await _click_first_available(page, selectors, timeout_ms=timeout_ms)


def _extract_place_short_id(url: str) -> Optional[str]:
    """Extract Google Maps short place id from `!16s...` when present."""
    match = PLACE_SHORT_ID_RE.search(url)
    if not match:
        return None
    return match.group(1)


def _build_minimal_place_url(
    identity: UrlIdentity,
    fallback_name: str = "",
    *,
    reviews: bool = False,
    source_url: str = "",
) -> Optional[str]:
    """Build a Maps place URL from feature id + coordinates.

    Google Maps is sensitive to the `/data=` arity markers. A review URL should
    use `!4m8!3m7!...!9m1!1b1!16s...`, not `!4m6!...!9m1!1b1`; the latter was
    being rewritten by Maps to `/maps/place//@lat,lng`, which loses the place
    detail panel and makes review-card selectors return zero cards.
    """
    feature_id = _google_feature_id_for_url(identity)
    if not feature_id or not identity.lat or not identity.lng:
        return None
    name = fallback_name or identity.place_name_hint or "Place"
    place_slug = quote(name.replace(" ", "+"), safe="+")
    zoom = identity.zoom or 16
    short_id = _extract_place_short_id(source_url or identity.canonical_url)
    if reviews:
        data = f"!4m8!3m7!1s{feature_id}!8m2!3d{identity.lat}!4d{identity.lng}!9m1!1b1"
        if short_id:
            data += f"!16s{short_id}"
    else:
        data = f"!4m6!3m5!1s{feature_id}!8m2!3d{identity.lat}!4d{identity.lng}"
        if short_id:
            data += f"!16s{short_id}"
    return (
        f"https://www.google.com/maps/place/{place_slug}/@{identity.lat},{identity.lng},{zoom}z/"
        f"data={data}?hl=vi"
    )


def _with_reviews_url(url: str) -> str:
    """Return a stable Google Maps place URL variant that opens the Reviews tab."""
    identity = derive_url_identity(url)
    reviews_url = _build_minimal_place_url(identity, identity.place_name_hint or "", reviews=True, source_url=url)
    if reviews_url:
        return reviews_url

    clean_url = _clean_place_url_from_identity(identity, identity.place_name_hint or "") or url
    if "!9m1!1b1" in clean_url:
        return clean_url
    parts = urlsplit(clean_url)
    path = parts.path
    if "/data=!" in path:
        path = path + "!9m1!1b1"
    elif "/data=" in path:
        path = path + "!9m1!1b1"
    elif path.endswith("/"):
        path = path + "data=!9m1!1b1"
    else:
        path = path + "/data=!9m1!1b1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _dedup_preserve_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


async def _reviews_panel_ready(page, selectors: dict[str, list[str]], *, timeout_ms: int = 700) -> tuple[bool, str]:
    """Return True when the visible UI looks like the Google Maps reviews panel."""
    _, count, card_selector = await _locator_count_first_available(page, selectors.get("review_card", []))
    if count > 0:
        return True, f"cards:{card_selector}"

    if await _visible_text_exists(
        page,
        ["Mới nhất", "Newest", "Liên quan nhất", "Có liên quan nhất", "Most relevant"],
        timeout_ms=timeout_ms,
    ):
        return True, "reviews-sort-ui"

    for selector in [
        'div[role="feed"]',
        'div[aria-label*="Bài đánh giá"]',
        'div[aria-label*="Reviews"]',
        '[data-review-id]',
    ]:
        try:
            loc = page.locator(selector).first
            await loc.wait_for(timeout=timeout_ms)
            if await loc.is_visible(timeout=250):
                return True, f"panel:{selector}"
        except Exception:
            continue
    return False, "not-ready"


async def _open_reviews_panel(page, selectors: dict[str, list[str]], *source_urls: str | None) -> tuple[bool, str]:
    """Open the review list without losing the real place URL.

    Prefer navigating to the stable `!9m1!1b1` reviews URL derived from the real
    input/opened URL. Some Google Maps builds rewrite broad review-tab clicks to
    `/maps/place//@lat,lng`, which makes the crawler see zero review cards.
    """
    candidate_review_urls = []
    for candidate in _dedup_preserve_order([*[u for u in source_urls if u], page.url]):
        if is_probably_placeholder_place_url(candidate):
            continue

        # If the incoming/detail URL already contains Google's review-tab marker,
        # try it as-is first. Rebuilding a minimal data payload can make Maps
        # rewrite some Vietnamese place URLs to `/maps/place//@...`.
        if "!9m1!1b1" in candidate and candidate not in candidate_review_urls:
            candidate_review_urls.append(candidate)

        review_url = _with_reviews_url(candidate)
        if not is_probably_placeholder_place_url(review_url) and review_url not in candidate_review_urls:
            candidate_review_urls.append(review_url)

    for idx, reviews_url in enumerate(candidate_review_urls):
        try:
            before_url = page.url
            if page.url != reviews_url:
                await page.goto(reviews_url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(4500)
            if is_probably_placeholder_place_url(page.url):
                print(
                    f"[gmaps-crawler] review_url_rewritten_to_placeholder idx={idx} requested={reviews_url} actual={page.url}",
                    file=sys.stderr,
                )
                if not is_probably_placeholder_place_url(before_url):
                    try:
                        await page.goto(before_url, wait_until="domcontentloaded", timeout=45000)
                        await page.wait_for_timeout(2500)
                    except Exception:
                        pass
                continue
            ready, method = await _reviews_panel_ready(page, selectors, timeout_ms=900)
            if ready:
                return True, f"url-9m1-1b1[{idx}]+{method}"

            clicked_after_url = await _click_reviews_tab_safely(page, selectors.get("reviews_button", []), timeout_ms=1800)
            if clicked_after_url:
                await page.wait_for_timeout(2500)
                ready, method = await _reviews_panel_ready(page, selectors, timeout_ms=900)
                if ready:
                    return True, f"url-9m1-1b1[{idx}]+safe-click+{method}"
        except Exception:
            pass

    before_click_url = page.url
    clicked = await _click_reviews_tab_safely(page, selectors.get("reviews_button", []), timeout_ms=2500)
    await page.wait_for_timeout(2200)
    ready, method = await _reviews_panel_ready(page, selectors, timeout_ms=900)
    if ready:
        return True, f"safe-click+{method}"

    if is_probably_placeholder_place_url(page.url) and not is_probably_placeholder_place_url(before_click_url):
        try:
            reviews_url = _with_reviews_url(before_click_url)
            await page.goto(reviews_url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(3500)
            ready, method = await _reviews_panel_ready(page, selectors, timeout_ms=900)
            if ready:
                return True, f"restore-after-placeholder+{method}"
        except Exception:
            pass

    return clicked, "reviews-panel-not-ready"


async def _click_button_by_keywords(page, keywords: list[str], *, timeout_ms: int = 1200) -> bool:
    """Click a visible button/div role=button whose text/aria/title contains any keyword."""
    lowered_keywords = [k.lower() for k in keywords]
    for selector in ['button', 'div[role="button"]', '[role="combobox"]', '[aria-haspopup="menu"]', '[aria-haspopup="listbox"]']:
        try:
            loc = page.locator(selector)
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 120)):
            btn = loc.nth(i)
            try:
                if not await btn.is_visible(timeout=300):
                    continue
                text = clean_text(await btn.inner_text(timeout=300) or "")
                aria = clean_text(await btn.get_attribute("aria-label", timeout=300) or "")
                title = clean_text(await btn.get_attribute("title", timeout=300) or "")
            except Exception:
                continue
            haystack = f"{text} {aria} {title}".lower()
            if any(k in haystack for k in lowered_keywords):
                try:
                    await btn.scroll_into_view_if_needed(timeout=timeout_ms)
                except Exception:
                    pass
                try:
                    await btn.click(timeout=timeout_ms)
                    return True
                except Exception:
                    try:
                        box = await btn.bounding_box(timeout=timeout_ms)
                        if box:
                            await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                            return True
                    except Exception:
                        continue
    return False


async def _click_menu_option_by_keywords(page, keywords: list[str], *, timeout_ms: int = 1800) -> bool:
    """Click a visible dropdown/menu option containing one of the provided keywords."""
    lowered_keywords = [k.lower() for k in keywords]
    for keyword in keywords:
        for exact in [True, False]:
            try:
                loc = page.get_by_text(keyword, exact=exact).first
                await loc.wait_for(timeout=timeout_ms)
                if await loc.is_visible(timeout=300):
                    try:
                        await loc.click(timeout=timeout_ms)
                    except Exception:
                        target = loc.locator(
                            'xpath=ancestor-or-self::*[@role="menuitemradio" or @role="menuitem" or @role="option" or @role="button"][1]'
                        ).first
                        await target.click(timeout=timeout_ms)
                    return True
            except Exception:
                continue
    for selector in ['div[role="menuitemradio"]', 'div[role="menuitem"]', 'div[role="option"]', '[role="menuitemradio"]', '[role="option"]', 'div[aria-checked]', 'span']:
        try:
            loc = page.locator(selector)
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 160)):
            item = loc.nth(i)
            try:
                if not await item.is_visible(timeout=300):
                    continue
                text = clean_text(await item.inner_text(timeout=300) or "")
                aria = clean_text(await item.get_attribute("aria-label", timeout=300) or "")
            except Exception:
                continue
            haystack = f"{text} {aria}".lower()
            if any(k in haystack for k in lowered_keywords):
                try:
                    await item.click(timeout=timeout_ms)
                    return True
                except Exception:
                    try:
                        box = await item.bounding_box(timeout=timeout_ms)
                        if box:
                            await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                            return True
                    except Exception:
                        continue
    return False


async def _visible_text_exists(page, keywords: list[str], *, timeout_ms: int = 400) -> bool:
    """Return True if any keyword is visible on the current page/panel."""
    for keyword in keywords:
        try:
            loc = page.get_by_text(keyword, exact=True).first
            await loc.wait_for(timeout=timeout_ms)
            if await loc.is_visible(timeout=250):
                return True
        except Exception:
            pass
    lowered = [k.lower() for k in keywords]
    for selector in ['div[role="button"]', 'button', 'div[aria-haspopup]', '[role="combobox"]', 'span']:
        try:
            loc = page.locator(selector)
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 120)):
            el = loc.nth(i)
            try:
                if not await el.is_visible(timeout=200):
                    continue
                text = clean_text(await el.inner_text(timeout=200) or "")
                aria = clean_text(await el.get_attribute("aria-label", timeout=200) or "")
            except Exception:
                continue
            haystack = f"{text} {aria}".lower()
            if any(k == haystack.strip() or k in haystack for k in lowered):
                return True
    return False


async def _click_text_or_ancestor_by_keywords(page, keywords: list[str], *, timeout_ms: int = 1000) -> tuple[bool, str]:
    """Click visible text matching current sort value, or its clickable ancestor."""
    for keyword in keywords:
        try:
            txt = page.get_by_text(keyword, exact=True).first
            await txt.wait_for(timeout=timeout_ms)
            if not await txt.is_visible(timeout=250):
                continue
            for ancestor_xpath in [
                'xpath=ancestor-or-self::*[@role="button" or @role="combobox" or @aria-haspopup="listbox" or @aria-haspopup="menu"][1]',
                'xpath=ancestor::*[@role="button" or @role="combobox" or @aria-haspopup="listbox" or @aria-haspopup="menu"][1]',
                'xpath=..',
            ]:
                try:
                    target = txt.locator(ancestor_xpath).first
                    if await target.is_visible(timeout=250):
                        await target.click(timeout=timeout_ms)
                        return True, f"text-ancestor:{keyword}"
                except Exception:
                    continue
            try:
                await txt.click(timeout=timeout_ms)
                return True, f"text:{keyword}"
            except Exception:
                continue
        except Exception:
            continue
    return False, "text-sort-control-not-found"


async def _click_select_like_sort_control(page, *, timeout_ms: int = 1000) -> tuple[bool, str]:
    """Click the select-like sort control shown in the Google Maps Reviews tab."""
    clicked, method = await _click_text_or_ancestor_by_keywords(
        page,
        [
            "Liên quan nhất", "Có liên quan nhất", "Phù hợp nhất",
            "Mới nhất", "Xếp hạng cao nhất", "Xếp hạng thấp nhất",
            "Most relevant", "Newest", "Highest rating", "Lowest rating",
        ],
        timeout_ms=timeout_ms,
    )
    if clicked:
        return True, method

    selectors = [
        'button[aria-label*="Sắp xếp bài đánh giá"]',
        'button[aria-label*="Sort reviews"]',
        'button[aria-label*="Sắp xếp"]',
        'button[aria-label*="Sort"]',
        'div[role="combobox"]',
        '[role="combobox"]',
        'div[role="button"][aria-haspopup="listbox"]',
        'div[role="button"][aria-haspopup="menu"]',
        'button[aria-haspopup="listbox"]',
        'button[aria-haspopup="menu"]',
        'div[aria-haspopup="listbox"]',
        'div[aria-haspopup="menu"]',
    ]
    sort_keywords = [
        "mới nhất", "liên quan", "phù hợp", "xếp hạng", "sắp xếp", "bài đánh giá",
        "newest", "relevant", "rating", "sort", "reviews",
    ]
    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 100)):
            el = loc.nth(i)
            try:
                if not await el.is_visible(timeout=250):
                    continue
                text = clean_text(await el.inner_text(timeout=250) or "")
                aria = clean_text(await el.get_attribute("aria-label", timeout=250) or "")
                title = clean_text(await el.get_attribute("title", timeout=250) or "")
            except Exception:
                continue
            haystack = f"{text} {aria} {title}".lower()
            if haystack.strip() and not any(k in haystack for k in sort_keywords):
                continue
            try:
                await el.scroll_into_view_if_needed(timeout=timeout_ms)
            except Exception:
                pass
            try:
                await el.click(timeout=timeout_ms)
                return True, f"select-like:{selector}"
            except Exception:
                try:
                    box = await el.bounding_box(timeout=timeout_ms)
                    if box:
                        await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                        return True, f"select-like-mouse:{selector}"
                except Exception:
                    continue
    return False, "select-like-sort-not-found"


async def _sort_reviews_by_newest(page, selectors: dict[str, list[str]]) -> tuple[bool, str]:
    """Best-effort sort Google Maps review panel by newest."""
    attempts: list[tuple[bool, str]] = []

    opened, method = await _click_select_like_sort_control(page, timeout_ms=1500)
    attempts.append((opened, method))
    if not opened:
        sort_opened = await _click_first_available(page, selectors.get("sort_button", []), timeout_ms=1200)
        method = "selector" if sort_opened else ""
        if not sort_opened:
            sort_opened = await _click_button_by_keywords(
                page,
                [
                    "sắp xếp bài đánh giá", "sắp xếp", "sap xep", "sort reviews", "sort",
                    "liên quan nhất", "có liên quan nhất", "lien quan", "most relevant", "relevant",
                ],
                timeout_ms=1200,
            )
            method = "keyword-button" if sort_opened else ""
        opened = sort_opened
        attempts.append((opened, method))

    if opened:
        await page.wait_for_timeout(900)
        newest_clicked = await _click_first_available(page, selectors.get("newest_option", []), timeout_ms=1200)
        if newest_clicked:
            await page.wait_for_timeout(2200)
            return True, method + "+selector-newest"
        newest_clicked = await _click_menu_option_by_keywords(page, ["Mới nhất", "Newest"], timeout_ms=1600)
        if newest_clicked:
            await page.wait_for_timeout(2200)
            return True, method + "+keyword-newest"
        if await _visible_text_exists(page, ["Mới nhất", "Newest"], timeout_ms=500):
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass
            return True, method + "+already-newest"
        attempts.append((False, method + "+newest-option-not-found"))

    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass
    return False, ";".join(m for _, m in attempts if m) or "sort-control-not-found"

async def _text_first_available(root, selectors: list[str]) -> Optional[str]:
    for selector in selectors:
        try:
            locator = root.locator(selector).first
            value = await locator.inner_text(timeout=1000)
            value = value.strip()
            if value:
                return value
        except Exception:
            continue
    return None


async def _debug_expand_dom_candidates(card, *, label: str = "", max_candidates: int = 30) -> None:
    """Print DOM candidates that may correspond to Google Maps' inline expand control.

    This is intentionally diagnostic. It does not click anything. It dumps tag,
    text, aria, class, jsaction, candidate rect, and nearest clickable ancestor
    info for nodes that mention Xem thêm/Thêm/More or look like Maps expand
    controls such as w8nwRe.
    """
    try:
        payload = await card.evaluate(
            """
            (el, args) => {
              const maxCandidates = args.maxCandidates || 30;
              const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
              const lowerIncludes = value => /xem\\s*thêm|\\bthêm\\b|see\\s*more|read\\s*more|\\bmore\\b|expand|w8nwRe/i.test(value || '');

              const ownerMarkers = [
                'thông tin phản hồi từ chủ sở hữu',
                'thông tin phản hồi của chủ sở hữu',
                'phản hồi từ chủ sở hữu',
                'phản hồi của chủ sở hữu',
                'response from the owner',
                'owner response'
              ].map(s => s.toLowerCase());

              const isInsideOwnerReply = node => {
                let cur = node;
                while (cur && cur !== el) {
                  const text = normalize(cur.innerText || cur.textContent || '').toLowerCase();
                  const aria = normalize((cur.getAttribute && cur.getAttribute('aria-label')) || '').toLowerCase();
                  if (ownerMarkers.some(marker => text.includes(marker) || aria.includes(marker))) return true;
                  cur = cur.parentElement;
                }
                return false;
              };

              const rectObj = node => {
                try {
                  const r = node.getBoundingClientRect();
                  return {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)};
                } catch (_) {
                  return null;
                }
              };

              const nodeInfo = node => {
                if (!node) return null;
                return {
                  tag: node.tagName || '',
                  text: normalize(node.innerText || node.textContent || '').slice(0, 160),
                  aria: normalize((node.getAttribute && node.getAttribute('aria-label')) || '').slice(0, 160),
                  title: normalize((node.getAttribute && node.getAttribute('title')) || '').slice(0, 160),
                  role: normalize((node.getAttribute && node.getAttribute('role')) || ''),
                  cls: normalize((node.getAttribute && node.getAttribute('class')) || '').slice(0, 240),
                  jsaction: normalize((node.getAttribute && node.getAttribute('jsaction')) || '').slice(0, 240),
                  tabindex: normalize((node.getAttribute && node.getAttribute('tabindex')) || ''),
                  rect: rectObj(node)
                };
              };

              const candidates = [];
              const nodes = [el, ...Array.from(el.querySelectorAll('*'))];
              for (const node of nodes) {
                if (candidates.length >= maxCandidates) break;
                const text = normalize(node.innerText || node.textContent || '');
                const ownText = normalize(Array.from(node.childNodes || [])
                  .filter(ch => ch.nodeType === Node.TEXT_NODE)
                  .map(ch => ch.textContent || '')
                  .join(' '));
                const aria = normalize((node.getAttribute && node.getAttribute('aria-label')) || '');
                const title = normalize((node.getAttribute && node.getAttribute('title')) || '');
                const cls = normalize((node.getAttribute && node.getAttribute('class')) || '');
                const jsaction = normalize((node.getAttribute && node.getAttribute('jsaction')) || '');
                const haystack = [ownText, text, aria, title, cls, jsaction].join(' ');
                if (!lowerIncludes(haystack)) continue;

                let clickable = node.closest && node.closest('button, [role="button"], a, [jsaction], [tabindex]');
                if (!clickable || !el.contains(clickable)) {
                  clickable = node;
                }

                candidates.push({
                  node: nodeInfo(node),
                  own_text: ownText.slice(0, 160),
                  inside_owner_reply: isInsideOwnerReply(node),
                  clickable: nodeInfo(clickable),
                });
              }

              return {
                label: args.label || '',
                card_text: normalize(el.innerText || '').slice(0, 600),
                card_rect: rectObj(el),
                candidate_count: candidates.length,
                candidates,
              };
            }
            """,
            {"label": label, "maxCandidates": max_candidates},
            timeout=1500,
        )
        print(
            "[gmaps-crawler] expand_dom_candidates "
            + json.dumps(payload, ensure_ascii=False),
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"[gmaps-crawler] expand_dom_debug_failed label={label!r} error={exc}", file=sys.stderr)


async def _expand_review_text_if_collapsed(card) -> bool:
    """Expand a Google Maps review card's collapsed customer text when possible.

    Google Maps currently renders the customer "Xem thêm" control as a real
    button like:
      <button class="w8nwRe kyuRq" jsaction="...review.expandReview">Xem thêm</button>

    The previous implementation skipped this button because it walked up to the
    whole review-card ancestor; that ancestor also contains the owner response
    text, so every descendant was incorrectly classified as "inside owner reply".
    This version distinguishes customer expandReview from owner expandOwnerResponse
    using the button's jsaction first, and only uses owner-container checks as a
    fallback.
    """

    async def customer_body_text() -> str:
        try:
            value = await card.evaluate(
                """
                el => {
                  const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
                  const ownerContainers = new Set(Array.from(el.querySelectorAll('.CDe7pd')));
                  const isInsideOwner = node => {
                    let cur = node;
                    while (cur && cur !== el) {
                      if (ownerContainers.has(cur)) return true;
                      cur = cur.parentElement;
                    }
                    return false;
                  };
                  const candidates = [];
                  for (const node of Array.from(el.querySelectorAll('span.wiI7pd, span[lang], div.MyEned span, div.MyEned'))) {
                    if (isInsideOwner(node)) continue;
                    const text = normalize(node.innerText || node.textContent || '');
                    if (!text) continue;
                    if (/^(xem thêm|thêm|see more|more|read more|mở rộng|expand|thích|like|chia sẻ|share|mới|new)$/i.test(text)) continue;
                    candidates.push(text);
                  }
                  return candidates.sort((a, b) => b.length - a.length)[0] || '';
                }
                """,
                timeout=900,
            )
            return clean_text(value)
        except Exception:
            return ""

    async def should_skip_candidate(candidate) -> tuple[bool, str]:
        """Return whether this candidate is owner-response/control noise."""
        try:
            info = await candidate.evaluate(
                """
                node => {
                  const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
                  const jsaction = (node.getAttribute('jsaction') || '').toLowerCase();
                  const cls = node.getAttribute('class') || '';
                  const text = normalize(node.innerText || node.textContent || '');
                  const aria = node.getAttribute('aria-label') || '';

                  // The strongest signal. Customer review expander and owner-response
                  // expander are separate jsactions in Google Maps.
                  if (jsaction.includes('expandreview')) {
                    return {skip: false, reason: 'customer-expandReview'};
                  }
                  if (jsaction.includes('expandownerresponse')) {
                    return {skip: true, reason: 'owner-expandOwnerResponse'};
                  }

                  // A nested button may be under the node selected by text/span.
                  const customerButton = node.querySelector && node.querySelector('button[jsaction*="expandReview"], button[jsaction*="expandreview"]');
                  if (customerButton) return {skip: false, reason: 'contains-customer-expandReview'};
                  const ownerButton = node.querySelector && node.querySelector('button[jsaction*="expandOwnerResponse"], button[jsaction*="expandownerresponse"]');
                  if (ownerButton) return {skip: true, reason: 'contains-owner-expandOwnerResponse'};

                  // Owner response content is usually under .CDe7pd. Do NOT walk up to
                  // the entire root review card and inspect all text; that causes false
                  // positives when a normal review card also contains an owner response.
                  if (node.closest && node.closest('.CDe7pd')) {
                    return {skip: true, reason: 'inside-owner-container-CDe7pd'};
                  }

                  // Avoid broad whole-card nodes; they are not clickable expand controls.
                  if ((cls || '').includes('jftiEf') || (cls || '').includes('jJc9Ad')) {
                    return {skip: true, reason: 'whole-card-node'};
                  }

                  const haystack = `${text} ${aria} ${cls} ${jsaction}`.toLowerCase();
                  if (!/(xem thêm|thêm|see more|more|read more|expandreview|w8nwre)/i.test(haystack)) {
                    return {skip: true, reason: 'not-expand-like'};
                  }
                  return {skip: false, reason: 'expand-like'};
                }
                """,
                timeout=700,
            )
            return bool(info.get("skip")), str(info.get("reason") or "")
        except Exception as exc:
            return True, f"inspect-failed:{exc}"

    # Prefer the actual Google Maps button/action over text spans. The span text
    # "Xem thêm" is often inside div.MyEned, but the real event is on the sibling
    #/overlay button with jsaction="...review.expandReview".
    selectors = [
        'button[jsaction*="expandReview"]',
        'button[jsaction*="expandreview"]',
        'button.w8nwRe[jsaction*="expandReview"]',
        'button.w8nwRe:has-text("Xem thêm")',
        'button.w8nwRe:has-text("Thêm")',
        'button:has-text("Xem thêm")',
        'button:has-text("Thêm")',
        'button:has-text("See more")',
        'button:has-text("More")',
        'button:has-text("Read more")',
        'button[aria-label*="Xem thêm"]',
        'button[aria-label*="Thêm"]',
        'button[aria-label*="See more"]',
        'button[aria-label*="More"]',
        'button[aria-label*="Read more"]',
        'div.MyEned:has-text("Xem thêm") button[jsaction*="expandReview"]',
        'div.MyEned:has-text("Thêm") button[jsaction*="expandReview"]',
        'span:has-text("Xem thêm")',
        'span:has-text("Thêm")',
        'span:has-text("See more")',
        'span:has-text("More")',
    ]

    changed_any = False
    for pass_idx in range(4):
        before_body = await customer_body_text()
        try:
            before_card = clean_text(await card.inner_text(timeout=900))
        except Exception:
            before_card = before_body
        before_len = len(before_body or before_card)
        clicked_this_pass = False

        for selector in selectors:
            try:
                loc = card.locator(selector)
                count = await loc.count()
            except Exception:
                continue
            for i in range(min(count, 12)):
                candidate = loc.nth(i)
                try:
                    if not await candidate.is_visible(timeout=300):
                        continue
                    skip, reason = await should_skip_candidate(candidate)
                    if skip:
                        continue

                    # If selector hit the text span/div, prefer the real customer
                    # expandReview button in the same card/body.
                    target = candidate
                    try:
                        jsaction = (await candidate.get_attribute("jsaction", timeout=250) or "").lower()
                    except Exception:
                        jsaction = ""
                    if "expandreview" not in jsaction:
                        try:
                            candidate_target = candidate.locator(
                                'xpath=ancestor-or-self::*[contains(@jsaction,"expandReview") or contains(@jsaction,"expandreview")][1]'
                            ).first
                            if await candidate_target.count() > 0 and await candidate_target.is_visible(timeout=250):
                                target = candidate_target
                        except Exception:
                            pass
                        if target == candidate:
                            try:
                                button_target = card.locator('button[jsaction*="expandReview"], button[jsaction*="expandreview"]').first
                                if await button_target.count() > 0 and await button_target.is_visible(timeout=250):
                                    target = button_target
                            except Exception:
                                pass

                    try:
                        await target.scroll_into_view_if_needed(timeout=900)
                    except Exception:
                        pass

                    clicked_by = "playwright"
                    try:
                        await target.click(timeout=1200, force=True)
                    except Exception:
                        box = await target.bounding_box(timeout=800)
                        if not box:
                            continue
                        await card.page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                        clicked_by = "mouse"

                    await card.page.wait_for_timeout(550)
                    after_body = await customer_body_text()
                    try:
                        after_card = clean_text(await card.inner_text(timeout=900))
                    except Exception:
                        after_card = after_body
                    after_len = len(after_body or after_card)
                    changed = after_len > before_len or ("xem thêm" not in (after_body or after_card).lower() and "see more" not in (after_body or after_card).lower())
                    print(
                        f"[gmaps-crawler] expand_click selector={selector!r} reason={reason} "
                        f"method={clicked_by} changed={changed} before_len={before_len} after_len={after_len}",
                        file=sys.stderr,
                    )
                    clicked_this_pass = True
                    changed_any = changed_any or changed
                    break
                except Exception:
                    continue
            if clicked_this_pass:
                break

        if not clicked_this_pass:
            break

    return changed_any


OWNER_REPLY_MARKER_RE = re.compile(
    r"\b("
    r"Thông tin phản hồi từ chủ sở hữu|"
    r"Thông tin phản hồi của chủ sở hữu|"
    r"Phản hồi từ chủ sở hữu|"
    r"Phản hồi của chủ sở hữu|"
    r"Response from the owner|"
    r"Owner response"
    r")\b",
    re.IGNORECASE,
)


def strip_owner_reply_from_review_text(value: Optional[str]) -> str:
    """Remove owner response text that Google Maps may render inside a review card."""
    text = clean_text(value)
    if not text:
        return ""
    control_only = {
        "mới",
        "new",
        "thông tin",
        "info",
        "information",
        "xem thêm",
        "see more",
        "more",
        "mở rộng",
        "expand",
        "read more",
    }
    if text.lower() in control_only:
        return ""
    match = OWNER_REPLY_MARKER_RE.search(text)
    if match:
        text = text[: match.start()]
    # Remove Google Maps controls/actions that can leak into broad card text.
    text = re.sub(r"\s*(?:Xem thêm|See more|More|Mở rộng|Expand|Read more)\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(?:Bản dịch của Google\s*・\s*Xem bản gốc|Google translation\s*・\s*Show original)\s*\([^)]*\)\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[\ue000-\uf8ff]?\s*(?:Thích|Like)\s*[\ue000-\uf8ff]?\s*(?:Chia sẻ|Share)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[\ue000-\uf8ff]?\s*(?:Thích|Like|Chia sẻ|Share)\s*$", "", text, flags=re.IGNORECASE)
    return clean_text(text).strip(" …")


async def _extract_google_review_text_without_owner_reply(card, selectors: list[str]) -> Optional[str]:
    """Extract only the customer review text from a Google Maps card.

    Google Maps cards also contain reviewer metadata, star glyphs, timestamps and
    action labels. Never return the whole card as review_text. Prefer dedicated
    review-body nodes, and if Maps only exposes broad nodes, trim metadata with a
    conservative DOM/text heuristic.
    """
    try:
        value = await card.evaluate(
            """
            el => {
              const ownerMarkers = [
                'th\\u00f4ng tin ph\\u1ea3n h\\u1ed3i t\\u1eeb ch\\u1ee7 s\\u1edf h\\u1eefu',
                'th\\u00f4ng tin ph\\u1ea3n h\\u1ed3i c\\u1ee7a ch\\u1ee7 s\\u1edf h\\u1eefu',
                'ph\\u1ea3n h\\u1ed3i t\\u1eeb ch\\u1ee7 s\\u1edf h\\u1eefu',
                'ph\\u1ea3n h\\u1ed3i c\\u1ee7a ch\\u1ee7 s\\u1edf h\\u1eefu',
                'response from the owner',
                'owner response'
              ].map(s => s.toLowerCase());

              const isInsideOwnerReply = node => {
                let cur = node;
                while (cur && cur !== el) {
                  const text = (cur.innerText || cur.textContent || '').toLowerCase();
                  const aria = ((cur.getAttribute && cur.getAttribute('aria-label')) || '').toLowerCase();
                  if (ownerMarkers.some(marker => text.includes(marker) || aria.includes(marker))) return true;
                  cur = cur.parentElement;
                }
                return false;
              };

              const badExact = new Set([
                'xem th\\u00eam', 'see more', 'more', 'm\\u1edf r\\u1ed9ng', 'expand', 'read more',
                'th\\u00edch', 'like', 'chia s\\u1ebb', 'share',
                'm\\u1edbi', 'new', 'th\\u00f4ng tin', 'info', 'information'
              ]);

              const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();

              const candidates = [];
              for (const node of Array.from(el.querySelectorAll('span.wiI7pd, span[lang], div.MyEned span, div.MyEned'))) {
                if (isInsideOwnerReply(node)) continue;
                const text = normalize(node.innerText || node.textContent || '');
                if (!text) continue;
                const lower = text.toLowerCase();
                if (badExact.has(lower)) continue;
                candidates.push(text);
              }

              if (candidates.length) {
                return candidates.sort((a, b) => b.length - a.length)[0];
              }

              // Last DOM-local fallback: find text after the M\\u1edaI/NEW time marker
              // and before control/owner markers, instead of returning full card text.
              let fullText = normalize(el.innerText || '');
              if (!fullText) return '';
              const lower = fullText.toLowerCase();
              let start = -1;
              const timeMarkers = [' tr\\u01b0\\u1edbc m\\u1edbi ', ' ago new ', ' tr\\u01b0\\u1edbc ', ' ago '];
              for (const marker of timeMarkers) {
                const idx = lower.indexOf(marker);
                if (idx >= 0) {
                  start = idx + marker.length;
                  break;
                }
              }
              let review = start >= 0 ? fullText.slice(start).trim() : '';

              const cutMarkers = [
                ' xem th\\u00eam', ' see more', ' th\\u00edch', ' like', ' chia s\\u1ebb', ' share',
                ' th\\u00f4ng tin ph\\u1ea3n h\\u1ed3i t\\u1eeb ch\\u1ee7 s\\u1edf h\\u1eefu',
                ' th\\u00f4ng tin ph\\u1ea3n h\\u1ed3i c\\u1ee7a ch\\u1ee7 s\\u1edf h\\u1eefu',
                ' ph\\u1ea3n h\\u1ed3i t\\u1eeb ch\\u1ee7 s\\u1edf h\\u1eefu',
                ' ph\\u1ea3n h\\u1ed3i c\\u1ee7a ch\\u1ee7 s\\u1edf h\\u1eefu',
                ' response from the owner', ' owner response'
              ];
              const reviewLower = review.toLowerCase();
              let cut = review.length;
              for (const marker of cutMarkers) {
                const idx = reviewLower.indexOf(marker);
                if (idx >= 0 && idx < cut) cut = idx;
              }
              return review.slice(0, cut).trim();
            }
            """,
            timeout=1200,
        )
        value = strip_owner_reply_from_review_text(value)
        if value and not _looks_like_google_review_card_dump(value):
            return value
    except Exception:
        pass

    fallback = await _text_first_available(card, selectors)
    fallback = strip_owner_reply_from_review_text(fallback)
    if fallback and not _looks_like_google_review_card_dump(fallback):
        return fallback
    return None


def _looks_like_google_review_card_dump(text: Optional[str]) -> bool:
    """Heuristic guard against using an entire Google Maps card as review_text."""
    value = clean_text(text)
    if not value:
        return False
    lowered = value.lower()
    metadata_markers = [
        "local guide",
        "b\\u00e0i \\u0111\\u00e1nh gi\\u00e1",
        "review",
        "\\u1ea3nh",
        "photo",
        "th\\u00edch",
        "like",
        "chia s\\u1ebb",
        "share",
        "xem th\\u00eam",
        "see more",
    ]
    if sum(1 for marker in metadata_markers if marker.encode("utf-8").decode("unicode_escape") in lowered) >= 2:
        return True
    normalized_ascii = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    if re.search(r"^(?:.{0,80})(?:local guide|\d+\s+bai\s+danh\s+gia|\d+\s+reviews)", normalized_ascii, re.IGNORECASE):
        return True
    if re.search(r"^[^.!?\n]{1,80}\s+.{0,12}\s+\d+\s+(?:gio|phut|ngay|tuan|thang|nam|hour|minute|day|week|month|year)", normalized_ascii, re.IGNORECASE):
        return True
    return False


async def _is_original_google_review_card(card) -> bool:
    """Return True only for the root/original Google Maps review card.

    Google Maps review panels can expose nested/auxiliary nodes through broad
    selectors such as `div[jsaction*="review"]`. Those nodes include expanded
    text containers, owner replies, and action/reply controls. Crawling them as
    cards creates duplicate/non-review rows. Keep only top-level review cards
    that contain reviewer + rating + time metadata and are not nested inside
    another review card.
    """
    try:
        return bool(
            await card.evaluate(
                """
                el => {
                  const rootSelectors = [
                    '[data-review-id]',
                    '[data-reviewer-id]',
                    '.jftiEf',
                    '.jJc9Ad',
                    '[jscontroller="e6Mltc"]'
                  ];

                  let rootMatches = false;
                  for (const selector of rootSelectors) {
                    try {
                      if (el.matches(selector)) {
                        rootMatches = true;
                        break;
                      }
                    } catch (_) {}
                  }
                  if (!rootMatches) return false;

                  const parentReview = el.parentElement && el.parentElement.closest(
                    '[data-review-id], [data-reviewer-id], .jftiEf, .jJc9Ad, [jscontroller="e6Mltc"]'
                  );
                  if (parentReview) return false;

                  const text = (el.innerText || '').toLowerCase();
                  const ownerReplyMarkers = [
                    'response from the owner',
                    'owner response',
                    'phản hồi từ chủ sở hữu',
                    'phản hồi của chủ sở hữu',
                    'chủ sở hữu'
                  ];
                  const hasReviewer = !!el.querySelector('.d4r55, [class*="d4r55"]');
                  const hasTime = !!el.querySelector('.rsqaWe, [class*="rsqaWe"]');
                  const hasRating = !!el.querySelector('[aria-label*=" sao"], [aria-label*=" star"], [aria-label*="stars"]');
                  // Textless/star-only reviews are still original review cards, and
                  // some Google Maps builds only mount .wiI7pd after expansion or
                  // translation rendering. Do not require a body-text node here;
                  // extraction can decide whether to keep/skip the card later.
                  if (ownerReplyMarkers.some(marker => text.includes(marker)) && !hasRating) {
                    return false;
                  }
                  return hasReviewer && hasTime && hasRating;
                }
                """
            )
        )
    except Exception:
        return False


async def _current_place_title(page) -> str:
    """Best-effort place title from an opened Google Maps detail panel."""
    for selector in [
        'h1.DUwDvf',
        'h1',
        'div[role="main"] h1',
    ]:
        try:
            value = clean_text(await page.locator(selector).first.inner_text(timeout=1200))
            if value:
                return value
        except Exception:
            continue
    return ""


def _name_tokens(value: str) -> set[str]:
    value = unicodedata.normalize("NFKD", value or "").lower()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    tokens = {t for t in re.split(r"[^a-z0-9]+", value) if len(t) >= 3}
    return tokens


def _names_roughly_match(expected: str, actual: str) -> bool:
    exp = _name_tokens(expected)
    act = _name_tokens(actual)
    if not exp or not act:
        return False
    return len(exp & act) >= max(1, min(2, len(exp)))


async def _looks_like_place_detail_page(page, expected_name: Optional[str] = None) -> bool:
    """Return True when the current Maps page appears to be a real place page.

    This rejects the problematic /maps/place//,@lat,lng viewport page shown in your
    screenshots, even if it contains toolbar buttons like Directions/Save.
    """
    if is_probably_placeholder_place_url(page.url):
        return False
    current_url = page.url or ""
    if "/maps/search/" in current_url or "!1m2!2m1" in current_url:
        # Google Maps sometimes keeps a search-results panel while the URL still looks
        # like /maps/place/... (for example URLs containing !1m2!2m1 search metadata).
        # In that state the visible "Bài đánh giá" button belongs to the results UI,
        # not to the target place review tab, so opening/crawling reviews yields 0 cards.
        return False
    title = await _current_place_title(page)
    if not title:
        return False
    if expected_name and not _names_roughly_match(expected_name, title):
        # Do not fail too aggressively for translated/shortened names; a real h1 is
        # still a better signal than the placeholder page.
        return True
    return True


def _google_feature_id_for_url(identity: UrlIdentity) -> Optional[str]:
    """Return raw Google feature id (`0x...:0x...`) from an identity when available."""
    if identity.source_place_id_type not in {"google_feature_id_from_url", "google_feature_id_from_discovery_hint"}:
        return None
    if identity.source_place_id.startswith("google_feature_"):
        raw = identity.source_place_id[len("google_feature_"):].replace("_", ":", 1)
        if GOOGLE_FEATURE_RE.fullmatch(raw):
            return raw
    if GOOGLE_FEATURE_RE.fullmatch(identity.source_place_id):
        return identity.source_place_id
    return None


def _clean_place_url_from_identity(identity: UrlIdentity, fallback_name: str = "") -> Optional[str]:
    """Build a Maps place URL without embedded search-result metadata.

    Google Maps URLs captured from search result cards often contain `!1m2!2m1`
    in the data segment. Navigating those URLs can render a search-results panel
    even though the path is `/maps/place/...`, causing the crawler to click
    "Viết bài đánh giá" instead of a real Reviews tab. Rebuilding a minimal
    feature-id URL avoids that mixed UI state.
    """
    return _build_minimal_place_url(identity, fallback_name, reviews=False, source_url=identity.canonical_url)


async def _click_best_search_result_for_name(page, expected_name: str) -> bool:
    """Click a Google Maps search result by visible name when possible.

    Google Maps often exposes many placeholder anchors. Clicking the visible result
    card is more reliable than opening the href directly.
    """
    candidates = [
        'a.hfpxzc',
        'div[role="article"] a[href*="/maps/place/"]',
        'a[aria-label][href*="/maps/place/"]',
        'a[href*="/maps/place/"]',
    ]
    best = None
    best_score = -1
    exp_tokens = _name_tokens(expected_name)
    for selector in candidates:
        try:
            loc = page.locator(selector)
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 30)):
            a = loc.nth(i)
            try:
                label = clean_text(await a.get_attribute("aria-label", timeout=500) or "")
                text = clean_text(await a.inner_text(timeout=500) or "")
                href = await a.get_attribute("href", timeout=500)
                article_text = ""
                try:
                    article_text = clean_text(await a.locator('xpath=ancestor::*[@role="article"][1]').inner_text(timeout=500) or "")
                except Exception:
                    pass
            except Exception:
                continue
            candidate_name = label or text or article_text
            cand_tokens = _name_tokens(candidate_name)
            score = len(exp_tokens & cand_tokens)
            if article_text:
                score = max(score, len(exp_tokens & _name_tokens(article_text)))
            if href and is_usable_discovered_place_url(href):
                score += 1
            if expected_name and _names_roughly_match(expected_name, candidate_name):
                score += 3
            if score > best_score:
                best_score = score
                best = a
    if best is None or best_score <= 0:
        return False
    try:
        await best.click(timeout=3500)
        await page.wait_for_timeout(2500)
        return True
    except Exception:
        return False


async def _open_place_for_crawl(page, item: CrawlInput, area_filter: Optional[AreaFilter]) -> tuple[UrlIdentity, str]:
    """Open a real Google Maps place detail page for an item.

    V4 fix: if direct navigation opens /maps/place//,@... placeholder, fall back to
    a name-based Maps search and click the visible result card.
    """
    initial_identity = derive_url_identity(item.google_maps_url)
    expected_name = item.restaurant_name or initial_identity.place_name_hint or ""

    open_url = initial_identity.canonical_url
    if "!1m2!2m1" in open_url or "/maps/search/" in open_url:
        clean_url = _clean_place_url_from_identity(initial_identity, expected_name)
        if clean_url:
            open_url = clean_url

    await page.goto(open_url, wait_until="domcontentloaded", timeout=45000)
    await page.wait_for_timeout(2500)

    if not await _looks_like_place_detail_page(page, expected_name):
        if expected_name:
            search_query = expected_name
            if area_filter and area_filter.area_name:
                search_query = f"{expected_name} {area_filter.area_name}"
            search_url = build_google_maps_search_url(search_query, area_filter)
            print(f"[gmaps-crawler] direct open gave placeholder; retry search-click query={search_query!r}", file=sys.stderr)
            await page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(2500)
            clicked = await _click_best_search_result_for_name(page, expected_name)
            print(f"[gmaps-crawler] retry_search_click={clicked} url={page.url}", file=sys.stderr)
            await page.wait_for_timeout(2000)

    resolved_identity = derive_url_identity(page.url)
    # If the opened URL is still placeholder/synthetic, keep the feature id found
    # during discovery for stable IDs and dedup.
    if item.source_place_id_hint:
        resolved_identity = UrlIdentity(
            canonical_url=resolved_identity.canonical_url,
            source_place_id=item.source_place_id_hint,
            source_place_id_type="google_feature_id_from_discovery_hint",
            lat=resolved_identity.lat or initial_identity.lat,
            lng=resolved_identity.lng or initial_identity.lng,
            zoom=resolved_identity.zoom or initial_identity.zoom,
            place_name_hint=resolved_identity.place_name_hint or initial_identity.place_name_hint,
        )
    elif resolved_identity.source_place_id_type == "synthetic_url_hash" and initial_identity.source_place_id_type != "synthetic_url_hash":
        resolved_identity = initial_identity

    title = await _current_place_title(page)
    return resolved_identity, title or expected_name


async def crawl_live_with_playwright(config: LiveCrawlerConfig) -> list[dict]:
    """Optional live-browser flow.

    This intentionally avoids stealth/proxy/CAPTCHA-bypass behavior. Expect to tune selectors
    using screenshots/snapshots from your environment before production use.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Install with: pip install playwright && playwright install chromium"
        ) from exc

    base_inputs: list[CrawlInput] = []
    if config.input_urls_jsonl:
        base_inputs.extend(read_input_urls(config.input_urls_jsonl))
    if config.discovery.target_url:
        base_inputs.append(CrawlInput(
            google_maps_url=config.discovery.target_url,
            role="target" if config.mode == "benchmark" else "candidate",
            restaurant_id=config.target_restaurant_id if config.mode == "benchmark" else None,
            restaurant_name=config.discovery.target_restaurant_name,
        ))

    all_reviews: list[dict] = []
    crawled_restaurant_ids: set[str] = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=config.headless)
        context = await browser.new_context(locale=config.locale)
        page = await context.new_page()

        discovered_inputs: list[CrawlInput] = []
        if config.discovery.enabled:
            discovered_inputs = await discover_live_area_inputs(
                page,
                area_filter=config.area_filter,
                search_queries=config.discovery.search_queries,
                max_places=config.discovery.max_places,
                click_search_this_area=config.discovery.click_search_this_area,
            )

        inputs = merge_crawl_inputs(base_inputs, discovered_inputs)
        print(
            f"[gmaps-crawler] base_inputs={len(base_inputs)} discovered_inputs={len(discovered_inputs)} merged_inputs={len(inputs)}",
            file=sys.stderr,
        )
        if not inputs:
            raise RuntimeError(
                "No Google Maps place URLs to crawl. Discovery found 0 usable place URLs. "
                "Try a more specific --search-query including the ward/district name, run with --headful, "
                "or provide --input-urls to isolate the review crawler."
            )

        item_dicts = [
            {
                "google_maps_url": item.google_maps_url,
                "role": item.role,
                "restaurant_id": item.restaurant_id,
                "restaurant_name": item.restaurant_name,
                "source_place_id_hint": item.source_place_id_hint,
            }
            for item in inputs
        ]
        effective_target_id = infer_target_restaurant_id_from_items(
            item_dicts,
            mode=config.mode,
            default_role=config.default_role,
            explicit_target_restaurant_id=config.target_restaurant_id,
        )

        for item in inputs:
            item_role = normalize_role(item.role, mode=config.mode, default_role=config.default_role)
            initial_identity = derive_url_identity(item.google_maps_url)
            precheck_identity = initial_identity
            if item.source_place_id_hint:
                precheck_identity = UrlIdentity(
                    canonical_url=initial_identity.canonical_url,
                    source_place_id=item.source_place_id_hint,
                    source_place_id_type="google_feature_id_from_discovery_hint",
                    lat=initial_identity.lat,
                    lng=initial_identity.lng,
                    zoom=initial_identity.zoom,
                    place_name_hint=initial_identity.place_name_hint,
                )
            if not identity_matches_area(precheck_identity, config.area_filter, allow_unknown_coordinates=True):
                print(f"[gmaps-crawler] skip outside area before open: {item.google_maps_url}", file=sys.stderr)
                continue

            identity, opened_title = await _open_place_for_crawl(page, item, config.area_filter)

            if not identity_matches_area(identity, config.area_filter, allow_unknown_coordinates=True):
                print(f"[gmaps-crawler] skip outside area after open: {page.url}", file=sys.stderr)
                continue

            restaurant_id = choose_restaurant_id(
                role=item_role,
                source_place_id=identity.source_place_id,
                provided_restaurant_id=item.restaurant_id,
                target_restaurant_id=effective_target_id,
            )
            crawled_restaurant_ids.add(restaurant_id)
            restaurant_name = item.restaurant_name or opened_title or identity.place_name_hint or initial_identity.place_name_hint or ""
            print(f"[gmaps-crawler] crawling restaurant_id={restaurant_id} name={restaurant_name!r} opened_url={page.url}", file=sys.stderr)

            reviews_clicked, reviews_method = await _open_reviews_panel(page, config.selectors, page.url, item.google_maps_url)
            print(f"[gmaps-crawler] reviews_clicked={reviews_clicked} method={reviews_method} url={page.url}", file=sys.stderr)
            await page.wait_for_timeout(1500)

            sort_clicked, sort_method = await _sort_reviews_by_newest(page, config.selectors)
            print(f"[gmaps-crawler] sort_clicked={sort_clicked} method={sort_method}", file=sys.stderr)

            seen_card_texts: set[str] = set()
            old_streak = 0
            previous_seen_count = 0
            no_progress_scrolls = 0
            place_reviews_before = len(all_reviews)
            debug_expand_dom_count = 0

            for scroll_idx in range(config.max_reviews_per_restaurant):
                cards, count, card_selector = await _locator_count_first_available(page, config.selectors["review_card"])
                if scroll_idx == 0 or count > previous_seen_count:
                    print(f"[gmaps-crawler] review_scroll={scroll_idx} card_selector={card_selector!r} visible_cards={count}", file=sys.stderr)

                for i in range(count):
                    card = cards.nth(i)
                    if not await _is_original_google_review_card(card):
                        continue
                    try:
                        card_text = (await card.inner_text(timeout=1000)).strip()
                    except Exception:
                        continue
                    if config.debug_expand_dom and debug_expand_dom_count < config.debug_expand_dom_limit:
                        await _debug_expand_dom_candidates(
                            card,
                            label=f"scroll{scroll_idx}:card{i + 1}",
                            max_candidates=30,
                        )
                        debug_expand_dom_count += 1

                    expand_clicked = await _expand_review_text_if_collapsed(card)
                    if expand_clicked:
                        try:
                            card_text = (await card.inner_text(timeout=1000)).strip()
                        except Exception:
                            pass
                    try:
                        native_review_id = await card.get_attribute("data-review-id", timeout=500)
                    except Exception as exc:
                        native_review_id = None
                        if len(seen_card_texts) <= 3:
                            print(
                                f"[gmaps-crawler] data_review_id_unavailable card_index={i} "
                                f"reason={type(exc).__name__}",
                                file=sys.stderr,
                            )
                    card_dedup_key = native_review_id or card_text
                    if not card_text or card_dedup_key in seen_card_texts:
                        continue
                    seen_card_texts.add(card_dedup_key)

                    review_text = await _extract_google_review_text_without_owner_reply(card, config.selectors["review_text"])
                    if not review_text:
                        if len(seen_card_texts) <= 3:
                            print(
                                f"[gmaps-crawler] skip_card_no_clean_review_text expand_clicked={expand_clicked} "
                                f"text={clean_text(card_text)[:120]!r}",
                                file=sys.stderr,
                            )
                        continue
                    review_text = strip_owner_reply_from_review_text(review_text)
                    relative_time_text = await _text_first_available(card, config.selectors["review_time"]) or card_text
                    reviewer_name = await _text_first_available(card, config.selectors["reviewer_name"])
                    rating_label = await _first_attr_available(card, config.selectors["rating_node"], "aria-label")
                    rating = parse_rating(rating_label or "")

                    parsed_time = parse_relative_review_time(relative_time_text, config.crawl_time)
                    if parsed_time.confidence == "unknown" and relative_time_text != card_text:
                        # Some Google Maps builds do not expose the date in span.rsqaWe.
                        # Fall back to scanning the full card text for known time expressions.
                        parsed_time = parse_relative_review_time(card_text, config.crawl_time)

                    if len(seen_card_texts) <= 3:
                        snippet = clean_text(review_text)[:90]
                        print(
                            f"[gmaps-crawler] sample_review time={relative_time_text!r} "
                            f"parsed_month={parsed_time.review_month} confidence={parsed_time.confidence} "
                            f"rating={rating} text={snippet!r}",
                            file=sys.stderr,
                        )

                    if parsed_time.review_month < config.crawl_month:
                        old_streak += 1
                        if old_streak >= config.stop_after_old_reviews:
                            break
                        continue
                    old_streak = 0

                    if parsed_time.review_month != config.crawl_month:
                        continue

                    if parsed_time.confidence == "unknown" and config.include_unknown_time:
                        parsed_time = ParsedReviewTime(
                            parsed_time.review_time,
                            config.crawl_month,
                            "unknown_assigned_to_crawl_month",
                            parsed_time.original_text,
                        )

                    if parsed_time.review_month == config.crawl_month:
                        all_reviews.append(
                            normalize_review(
                                {
                                    "restaurant_id": restaurant_id,
                                    "restaurant_name": restaurant_name,
                                    "source": "google_maps_url_crawler",
                                    "source_place_id": identity.source_place_id,
                                    "native_review_id": native_review_id,
                                    "reviewer_name": reviewer_name,
                                    "rating": rating,
                                    "review_text": review_text,
                                    "review_time": parsed_time.review_time,
                                    "review_month": parsed_time.review_month,
                                    "time_confidence": parsed_time.confidence,
                                    "language": "vi",
                                }
                            )
                        )

                if old_streak >= config.stop_after_old_reviews:
                    break

                if len(seen_card_texts) == previous_seen_count:
                    no_progress_scrolls += 1
                    if no_progress_scrolls >= 5:
                        break
                else:
                    no_progress_scrolls = 0
                    previous_seen_count = len(seen_card_texts)

                await _scroll_results_panel_or_page(page)
                await page.wait_for_timeout(900)

            print(
                f"[gmaps-crawler] restaurant_done restaurant_id={restaurant_id} cards_seen={len(seen_card_texts)} reviews_kept={len(all_reviews) - place_reviews_before}",
                file=sys.stderr,
            )

        await browser.close()

    print(f"[gmaps-crawler] raw_reviews_collected={len(all_reviews)}", file=sys.stderr)
    deduped = dedup_reviews(all_reviews)
    print(f"[gmaps-crawler] deduped_reviews={len(deduped)}", file=sys.stderr)
    validate_reviews_jsonl_objects(
        deduped,
        crawl_month=config.crawl_month,
        mode=config.mode,
        require_target_restaurant_id=effective_target_id,
        crawled_restaurant_ids=crawled_restaurant_ids,
        min_peers=config.min_peers,
        min_restaurants=config.min_restaurants,
    )
    write_jsonl(config.output_jsonl, deduped)
    return deduped


# -----------------------------------------------------------------------------
# Demo data and self-test
# -----------------------------------------------------------------------------

SAMPLE_HTML = """<!doctype html>
<html>
<body>
  <div class="review-card" data-review-id="native_001">
    <span class="reviewer">Nguyen A</span>
    <span class="rating">2 sao</span>
    <span class="time">3 tuần trước</span>
    <span class="text">Phục vụ chậm, đồ ăn ổn.</span>
  </div>
  <div class="review-card" data-review-id="native_002">
    <span class="reviewer">Tran B</span>
    <span class="rating">5 sao</span>
    <span class="time">2 ngày trước</span>
    <span class="text">Món ăn ngon, nhân viên nhiệt tình.</span>
  </div>
  <div class="review-card" data-review-id="native_old">
    <span class="reviewer">Le C</span>
    <span class="rating">1 sao</span>
    <span class="time">2 tháng trước</span>
    <span class="text">Quá cũ để thuộc tháng crawl.</span>
  </div>
</body>
</html>
"""

SAMPLE_URL = "https://www.google.com/maps/place/May+Tre+Dan+Restaurant+-+Authentic+Vietnamese+Cuisine/@21.0295216,105.8444368,17z/data=!4m10!1m2!2m1!1srestaurants+near+me!3m6!1s0x3135abe82d6c811d:0xdc3b9b71b9ddcc90!8m2!3d21.0293134!4d105.8454768!16s%2Fg%2F11vf33ltyw?entry=ttu&g_ep=abc"
PEER_URL = "https://www.google.com/maps/place/Peer+A/@21.0301,105.8460,17z/data=!1s0x3135abc000000001:0x1111111111111111"
PEER_B_URL = "https://www.google.com/maps/place/Peer+B/@21.0310,105.8470,17z/data=!1s0x3135abc000000002:0x2222222222222222"
OUTSIDE_URL = "https://www.google.com/maps/place/Far+Away/@20.0000,105.0000,17z/data=!1s0x999:0x888"

SAMPLE_SEARCH_HTML = f"""<!doctype html>
<html>
<body>
  <a class="place-result" href="{SAMPLE_URL}" data-name="May Tre Dan Restaurant">May Tre Dan Restaurant</a>
  <a class="place-result" href="{PEER_URL}" data-name="Peer A">Peer A</a>
  <a class="place-result" href="{PEER_B_URL}" data-name="Peer B">Peer B</a>
  <a class="place-result" href="{OUTSIDE_URL}" data-name="Far Away">Far Away</a>
</body>
</html>
"""


def demo_snapshots() -> list[dict]:
    return [
        {
            "html": SAMPLE_HTML,
            "google_maps_url": SAMPLE_URL,
            "role": "target",
            "restaurant_id": "res_demo",
            "restaurant_name": "May Tre Dan Restaurant",
        },
        {
            "html": SAMPLE_HTML,
            "google_maps_url": PEER_URL,
            "role": "peer",
            "restaurant_id": "res_peer_01",
            "restaurant_name": "Peer A",
        },
    ]


def collection_demo_snapshots() -> list[dict]:
    return [
        {
            "html": SAMPLE_HTML,
            "google_maps_url": SAMPLE_URL,
            "restaurant_name": "May Tre Dan Restaurant",
        },
        {
            "html": SAMPLE_HTML,
            "google_maps_url": PEER_URL,
            "restaurant_name": "Peer A",
        },
    ]


def build_demo_rows(mode: str = "benchmark") -> list[dict]:
    if mode == "collection":
        return offline_build_jsonl_objects(
            collection_demo_snapshots(),
            crawl_time=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
            crawl_month="2026-05",
            mode="collection",
            min_restaurants=2,
        )
    return offline_build_jsonl_objects(
        demo_snapshots(),
        crawl_time=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
        crawl_month="2026-05",
        mode="benchmark",
        target_restaurant_id="res_demo",
        min_peers=1,
        min_restaurants=2,
    )


def run_self_test() -> None:
    canonical = canonicalize_google_maps_url(SAMPLE_URL)
    assert "entry=" not in canonical
    assert "g_ep=" not in canonical
    assert "0x3135abe82d6c811d:0xdc3b9b71b9ddcc90" in canonical

    identity = derive_url_identity(SAMPLE_URL)
    assert identity.source_place_id == "google_feature_0x3135abe82d6c811d_0xdc3b9b71b9ddcc90"
    assert abs((identity.lat or 0) - 21.0293134) < 1e-9
    assert abs((identity.lng or 0) - 105.8454768) < 1e-9
    assert make_restaurant_id("peer", identity.source_place_id).startswith("res_peer_")
    assert make_restaurant_id("candidate", identity.source_place_id).startswith("res_candidate_")
    assert normalize_role(None, mode="collection") == "candidate"
    assert normalize_role(None, mode="benchmark") == "peer"
    assert identity_matches_area(identity, AreaFilter(bbox=(21.0, 105.8, 21.1, 105.9)))
    assert not identity_matches_area(identity, AreaFilter(bbox=(20.0, 105.8, 20.5, 105.9)))

    crawl_time = datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc)
    assert parse_relative_review_time("3 tuần trước", crawl_time).review_time.date().isoformat() == "2026-05-13"
    assert parse_relative_review_time("2 ngày trước", crawl_time).review_month == "2026-06"
    assert parse_relative_review_time("1 tháng trước", crawl_time).confidence == "low"
    assert parse_relative_review_time("12 bài đánh giá 3 tuần trước", crawl_time).review_month == "2026-05"
    assert parse_relative_review_time("không có ngày", crawl_time).confidence == "unknown"
    assert strip_owner_reply_from_review_text(
        "Cá nhân tôi thấy phiên bản Hàn Quốc ngon hơn. Xem thêm "
        "Thông tin phản hồi từ chủ sở hữu vừa xong Cảm ơn bạn."
    ) == "Cá nhân tôi thấy phiên bản Hàn Quốc ngon hơn."
    assert strip_owner_reply_from_review_text(
        "Good pizza See more Response from the owner 1 hour ago Thanks."
    ) == "Good pizza"
    assert strip_owner_reply_from_review_text(
        "Đồ ăn: 5 Dịch vụ: 5… Xem thêm \ue8dc Thích \ue80d Chia sẻ"
    ) == "Đồ ăn: 5 Dịch vụ: 5"
    assert strip_owner_reply_from_review_text("MỚI") == ""
    assert strip_owner_reply_from_review_text("Thông tin") == ""
    assert _looks_like_google_review_card_dump(
        "JONGWOOK LEE Local Guide · 3 bài đánh giá · 50 ảnh 53 phút trước MỚI Cá nhân tôi thấy phiên bản Hàn Quốc ngon hơn. Xem thêm Thích Chia sẻ"
    )

    cards = parse_review_cards_from_html(SAMPLE_HTML)
    assert len(cards) == 3
    assert cards[0]["native_review_id"] == "native_001"
    assert cards[0]["rating"] == 2
    assert "Phục vụ chậm" in (cards[0]["review_text"] or "")

    row = normalize_review(
        {
            "restaurant_id": "res_demo",
            "restaurant_name": "May Tre Dan Restaurant",
            "source": "google_maps_url_crawler",
            "source_place_id": "google_feature_abc",
            "native_review_id": "native_001",
            "reviewer_name": "Nguyen A",
            "rating": 2,
            "review_text": "  Phục vụ chậm\n",
            "review_time": datetime(2026, 5, 13, 0, 0, tzinfo=timezone.utc),
            "review_month": "2026-05",
            "language": "vi",
        }
    )
    assert row["review_text"] == "Phục vụ chậm"
    assert row["source_review_id"] == "native_001"
    validate_reviews_jsonl_objects([row], crawl_month="2026-05", mode="benchmark", require_target_restaurant_id="res_demo")
    validate_reviews_jsonl_objects([row], crawl_month="2026-05", mode="collection", min_restaurants=1)
    validate_reviews_jsonl_objects(
        [],
        crawl_month="2026-05",
        mode="benchmark",
        require_target_restaurant_id="res_demo",
        crawled_restaurant_ids=["res_demo", "res_peer_01"],
        min_peers=1,
        min_restaurants=2,
    )

    duplicate = dict(row, review_id="different")
    assert len(dedup_reviews([row, duplicate])) == 1

    # Auto area resolver test via local cache. No network call is made.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cache_path = cache_dir / f"{slugify_area_name('Phường Test, Hà Nội, Việt Nam')}.geojson"
        cached_feature = {
            "type": "Feature",
            "properties": {
                "area_name": "Phường Test, Hà Nội, Việt Nam",
                "bbox": [21.0, 105.8, 21.1, 105.9],
                "source": "unit_test_cache",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[105.8, 21.0], [105.9, 21.0], [105.9, 21.1], [105.8, 21.1], [105.8, 21.0]]],
            },
        }
        cache_path.write_text(json.dumps(cached_feature, ensure_ascii=False), encoding="utf-8")
        auto_area = build_area_filter(area_name="Phường Test, Hà Nội, Việt Nam", area_cache_dir=cache_dir, allow_area_network=False)
        assert auto_area is not None
        assert auto_area.bbox == (21.0, 105.8, 21.1, 105.9)
        assert auto_area.polygons and len(auto_area.polygons[0]) == 5
        assert identity_matches_area(identity, auto_area)

        point_cache_path = cache_dir / f"{slugify_area_name('Ph?????ng Point, H??? Ch?? Minh, Vi???t Nam')}.geojson"
        point_feature = {
            "type": "Feature",
            "properties": {
                "area_name": "Ph?????ng Point, H??? Ch?? Minh, Vi???t Nam",
                "bbox": [10.77, 106.69, 10.79, 106.71],
                "source": "unit_test_point_cache",
            },
            "geometry": {"type": "Point", "coordinates": [106.70, 10.78]},
        }
        point_cache_path.write_text(json.dumps(point_feature, ensure_ascii=False), encoding="utf-8")
        point_area = build_area_filter(
            area_name="Ph?????ng Point, H??? Ch?? Minh, Vi???t Nam",
            area_cache_dir=cache_dir,
            allow_area_network=False,
        )
        assert point_area is not None
        assert point_area.bbox == (10.77, 106.69, 10.79, 106.71)
        assert point_area.polygons is None
        outside_url = "https://www.google.com/maps/place/Far/@20.0,105.0,17z/data=!3d20.0!4d105.0!1s0x1:0x2"
        assert not identity_matches_area(derive_url_identity(outside_url), auto_area)

    rows = build_demo_rows("benchmark")
    assert len(rows) == 2
    assert {r["restaurant_id"] for r in rows} == {"res_demo", "res_peer_01"}

    collection_rows = build_demo_rows("collection")
    assert len(collection_rows) == 2
    assert all(r["restaurant_id"].startswith("res_candidate_") for r in collection_rows)
    validate_reviews_jsonl_objects(collection_rows, crawl_month="2026-05", mode="collection", min_restaurants=2)

    # Area discovery parser test: fixture has 4 places, bbox keeps 3 and filters out OUTSIDE_URL.
    discovered = parse_place_results_from_html(SAMPLE_SEARCH_HTML, area_filter=AreaFilter(bbox=(21.0, 105.8, 21.1, 105.9)), max_places=10)
    assert len(discovered) == 3
    assert {derive_url_identity(d.google_maps_url).source_place_id for d in discovered} == {
        derive_url_identity(SAMPLE_URL).source_place_id,
        derive_url_identity(PEER_URL).source_place_id,
        derive_url_identity(PEER_B_URL).source_place_id,
    }
    search_url = build_google_maps_search_url("nhà hàng", AreaFilter(bbox=(21.0, 105.8, 21.1, 105.9)))
    assert "@21.0500000,105.8500000,16z" in search_url
    assert len(merge_crawl_inputs(discovered, discovered)) == 3
    unknown_identity = derive_url_identity("https://www.google.com/maps/place/Unknown+Coords")
    assert not identity_matches_area(unknown_identity, AreaFilter(bbox=(21.0, 105.8, 21.1, 105.9)))
    assert identity_matches_area(unknown_identity, AreaFilter(bbox=(21.0, 105.8, 21.1, 105.9)), allow_unknown_coordinates=True)
    placeholder_url = "https://www.google.com/maps/place//,@21.0401419,105.8445139,12z/data=!3m1!4b1"
    assert is_probably_placeholder_place_url(placeholder_url)
    assert not is_usable_discovered_place_url(placeholder_url)
    assert is_usable_discovered_place_url(SAMPLE_URL)

    try:
        offline_build_jsonl_objects(collection_demo_snapshots(), crawl_time=crawl_time, crawl_month="2026-05", mode="benchmark", min_peers=1)
        raise AssertionError("benchmark mode without target should fail")
    except ValidationError as exc:
        assert "target restaurant_id" in str(exc)

    hinted = CrawlInput(google_maps_url=placeholder_url, restaurant_name="X", source_place_id_hint="google_feature_hint")
    assert merge_crawl_inputs([hinted])[0].source_place_id_hint == "google_feature_hint"

    assert _dedup_preserve_order(["a", "b", "a", "", "c"]) == ["a", "b", "c"]
    review_url = _with_reviews_url(SAMPLE_URL)
    assert "!9m1!1b1" in review_url
    assert "!4m8!3m7!" in review_url
    assert "!16s%2Fg%2F11vf33ltyw" in review_url
    clean_place_url = _clean_place_url_from_identity(identity, "May Tre Dan Restaurant")
    assert clean_place_url is not None
    assert "!1m2!2m1" not in clean_place_url
    assert "!4m6!3m5!" in clean_place_url
    assert "0x3135abe82d6c811d:0xdc3b9b71b9ddcc90" in clean_place_url

    print("Self-test passed: 44 core checks OK")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_iso_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _ensure_tz(parsed)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Single-file Google Maps URL review crawler pipeline")
    parser.add_argument("--self-test", action="store_true", help="Run built-in self-test checks")
    parser.add_argument("--offline-demo", action="store_true", help="Run embedded offline demo and write JSONL")
    parser.add_argument("--live", action="store_true", help="Run optional Playwright live crawler")
    parser.add_argument("--input-urls", type=Path, help="Input Google Maps URL JSONL for live mode")
    parser.add_argument("--output", type=Path, default=Path("raw_reviews_2026-05_area_x.jsonl"), help="Output JSONL path")
    parser.add_argument("--crawl-month", default="2026-05", help="Month to keep, format YYYY-MM")
    parser.add_argument("--crawl-time", default="2026-06-03T10:00:00+00:00", help="ISO crawl time used for relative date parsing")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="collection", help="collection = no target required; benchmark = require target + peers")
    parser.add_argument("--default-role", choices=sorted(VALID_ROLES), default=None, help="Role for input rows that omit role. Defaults: candidate in collection, peer in benchmark")
    parser.add_argument("--target-restaurant-id", default=None, help="Target restaurant_id for benchmark validation. If omitted, inferred from the first input row with role=target")
    parser.add_argument("--min-peers", type=int, default=0, help="Benchmark mode only: minimum peer restaurant count for validation")
    parser.add_argument("--min-restaurants", type=int, default=1, help="Minimum distinct restaurant_id count for validation")
    parser.add_argument("--headful", action="store_true", help="Live mode only: run browser non-headless")
    parser.add_argument("--bbox", help="Optional area filter: min_lat,min_lng,max_lat,max_lng")
    parser.add_argument("--area-polygon", type=Path, help="Optional GeoJSON Polygon/MultiPolygon file for area filtering")
    parser.add_argument("--area-name", help="Optional administrative area name to auto-resolve to polygon+bbox, e.g. 'Phường Hàng Trống, Hoàn Kiếm, Hà Nội, Việt Nam'")
    parser.add_argument("--area-cache", type=Path, default=Path("data/area_cache"), help="Directory for cached area GeoJSON files used by --area-name")
    parser.add_argument("--no-area-network", action="store_true", help="Do not call Nominatim when --area-name cache is missing")
    parser.add_argument("--discover-from-area", action="store_true", help="Discover Google Maps place URLs from area search before crawling reviews")
    parser.add_argument("--search-query", action="append", dest="search_queries", help="Area discovery search query. Repeatable. Defaults to nhà hàng/quán ăn/restaurant")
    parser.add_argument("--max-discovered-places", type=int, default=80, help="Maximum place URLs to discover from area search")
    parser.add_argument("--target-url", help="Optional target Google Maps URL, useful for benchmark mode with --discover-from-area")
    parser.add_argument("--target-restaurant-name", help="Optional target restaurant display name for --target-url")
    parser.add_argument("--no-search-this-area-click", action="store_true", help="Live discovery: do not click Search this area/Tìm kiếm khu vực này")
    parser.add_argument("--max-reviews-per-restaurant", type=int, default=200, help="Live mode: maximum review scroll iterations per restaurant")
    parser.add_argument("--stop-after-old-reviews", type=int, default=20, help="Live mode: stop after this many old reviews in a row once sorted newest")
    parser.add_argument("--debug-expand-dom", action="store_true", help="Live debug: print DOM candidates for the inline Xem thêm/Thêm review expander")
    parser.add_argument("--debug-expand-dom-limit", type=int, default=5, help="Live debug: maximum number of review cards to dump expand DOM candidates for")
    parser.add_argument("--include-unknown-time", action="store_true", help="Live debug: keep reviews whose time cannot be parsed, assigning review_month=crawl_month")
    parser.add_argument("--include-nonmatching-month", action="store_true", help="Live debug: keep visible parsed reviews even when review_month != crawl_month")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    if args.offline_demo:
        area_filter = build_area_filter(args.bbox, args.area_polygon, args.area_name, args.area_cache, not args.no_area_network)
        if args.discover_from_area:
            discovered = parse_place_results_from_html(
                SAMPLE_SEARCH_HTML,
                area_filter=area_filter,
                max_places=args.max_discovered_places,
            )
            if args.mode == "benchmark":
                target_input = CrawlInput(
                    google_maps_url=args.target_url or SAMPLE_URL,
                    role="target",
                    restaurant_id=args.target_restaurant_id or "res_demo",
                    restaurant_name=args.target_restaurant_name or "May Tre Dan Restaurant",
                )
                # Keep discovered places as peers except the target duplicate.
                peer_inputs = [CrawlInput(google_maps_url=i.google_maps_url, role="peer", restaurant_name=i.restaurant_name) for i in discovered]
                inputs = merge_crawl_inputs([target_input], peer_inputs)
                effective_target_id = args.target_restaurant_id or "res_demo"
            else:
                inputs = [CrawlInput(google_maps_url=i.google_maps_url, role=args.default_role or "candidate", restaurant_name=i.restaurant_name) for i in discovered]
                effective_target_id = args.target_restaurant_id
            snapshots = snapshots_from_discovered_inputs(inputs, mode=args.mode, target_restaurant_id=effective_target_id)
        else:
            snapshots = demo_snapshots() if args.mode == "benchmark" else collection_demo_snapshots()
            effective_target_id = args.target_restaurant_id
        rows = offline_build_jsonl_objects(
            snapshots,
            crawl_time=parse_iso_datetime(args.crawl_time),
            crawl_month=args.crawl_month,
            mode=args.mode,
            default_role=args.default_role,
            target_restaurant_id=effective_target_id,
            min_peers=args.min_peers,
            min_restaurants=args.min_restaurants,
            area_filter=area_filter,
        )
        write_jsonl(args.output, rows)
        source_note = "discovery demo" if args.discover_from_area else "demo"
        print(f"Wrote {len(rows)} {args.mode} {source_note} reviews to {args.output}")
        return 0

    if args.live:
        if not args.input_urls and not args.discover_from_area and not args.target_url:
            parser.error("--live requires --input-urls, --discover-from-area, or --target-url")
        config = LiveCrawlerConfig(
            crawl_month=args.crawl_month,
            crawl_time=parse_iso_datetime(args.crawl_time),
            input_urls_jsonl=args.input_urls,
            output_jsonl=args.output,
            mode=args.mode,
            default_role=args.default_role,
            target_restaurant_id=args.target_restaurant_id,
            min_peers=args.min_peers,
            min_restaurants=args.min_restaurants,
            max_reviews_per_restaurant=args.max_reviews_per_restaurant,
            stop_after_old_reviews=args.stop_after_old_reviews,
            include_unknown_time=args.include_unknown_time,
            include_nonmatching_month=args.include_nonmatching_month,
            debug_expand_dom=args.debug_expand_dom,
            debug_expand_dom_limit=args.debug_expand_dom_limit,
            headless=not args.headful,
            area_filter=build_area_filter(args.bbox, args.area_polygon, args.area_name, args.area_cache, not args.no_area_network),
            discovery=DiscoveryConfig(
                enabled=args.discover_from_area,
                search_queries=args.search_queries or ["nhà hàng", "quán ăn", "restaurant"],
                max_places=args.max_discovered_places,
                target_url=args.target_url,
                target_restaurant_name=args.target_restaurant_name,
                click_search_this_area=not args.no_search_this_area_click,
            ),
        )
        rows = asyncio.run(crawl_live_with_playwright(config))
        print(f"Wrote {len(rows)} live reviews to {args.output}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
