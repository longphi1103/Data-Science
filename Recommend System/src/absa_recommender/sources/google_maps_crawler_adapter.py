import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


class GoogleMapsCrawlerAdapter:
    """Source adapter that invokes the single-file Google Maps crawler and reads its JSONL output."""

    source = "google_maps_url_crawler"

    def __init__(
        self,
        crawler_script: str | Path | None = None,
        output_path: str | Path = "data/gmaps_monthly_raw.jsonl",
        input_urls_path: str | Path | None = None,
        mode: str = "benchmark",
        target_restaurant_id: str | None = None,
        target_url: str | None = None,
        target_restaurant_name: str | None = None,
        area_name: str | None = None,
        bbox: str | None = None,
        discover_from_area: bool = False,
        live: bool = True,
        headful: bool = True,
        max_discovered_places: int = 3,
        max_reviews_per_restaurant: int = 10,
        stop_after_old_reviews: int = 20,
        min_peers: int = 0,
        min_restaurants: int = 1,
        no_area_network: bool = False,
        search_queries: list[str] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.crawler_script = (
            Path(crawler_script)
            if crawler_script is not None
            else Path(__file__).with_name("gmaps_url_crawler_single_discovery.py")
        )
        self.output_path = Path(output_path)
        self.input_urls_path = Path(input_urls_path) if input_urls_path else None
        self.mode = mode
        self.target_restaurant_id = target_restaurant_id
        self.target_url = target_url
        self.target_restaurant_name = target_restaurant_name
        self.area_name = area_name
        self.bbox = bbox
        self.discover_from_area = discover_from_area
        self.live = live
        self.headful = headful
        self.max_discovered_places = max_discovered_places
        self.max_reviews_per_restaurant = max_reviews_per_restaurant
        self.stop_after_old_reviews = stop_after_old_reviews
        self.min_peers = min_peers
        self.min_restaurants = min_restaurants
        self.no_area_network = no_area_network
        self.search_queries = search_queries or []
        self.log_callback = log_callback

    def fetch_reviews(
        self,
        restaurant_id: str | None = None,
        month: str | None = None,
    ) -> list[dict[str, Any]]:
        if month is None:
            raise ValueError("GoogleMapsCrawlerAdapter requires month")
        self._run_crawler(restaurant_id=restaurant_id, month=month)
        return self._read_rows(restaurant_id=restaurant_id, month=month)

    def _run_crawler(self, restaurant_id: str | None, month: str) -> None:
        if not self.crawler_script.exists():
            raise FileNotFoundError(f"Google Maps crawler script not found: {self.crawler_script}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(self.crawler_script),
            "--output",
            str(self.output_path),
            "--crawl-month",
            month,
            "--mode",
            self.mode,
            "--min-peers",
            str(self.min_peers),
            "--min-restaurants",
            str(self.min_restaurants),
            "--max-discovered-places",
            str(self.max_discovered_places),
            "--max-reviews-per-restaurant",
            str(self.max_reviews_per_restaurant),
            "--stop-after-old-reviews",
            str(self.stop_after_old_reviews),
        ]
        if self.live:
            command.append("--live")
        else:
            command.append("--offline-demo")
        if self.input_urls_path is not None:
            command.extend(["--input-urls", str(self.input_urls_path)])
        if restaurant_id or self.target_restaurant_id:
            command.extend(["--target-restaurant-id", restaurant_id or self.target_restaurant_id or ""])
        if self.target_url:
            command.extend(["--target-url", self.target_url])
        if self.target_restaurant_name:
            command.extend(["--target-restaurant-name", self.target_restaurant_name])
        if self.area_name:
            command.extend(["--area-name", self.area_name])
        if self.bbox:
            command.extend(["--bbox", self.bbox])
        if self.discover_from_area:
            command.append("--discover-from-area")
        if self.headful:
            command.append("--headful")
        if self.no_area_network:
            command.append("--no-area-network")
        for query in self.search_queries:
            command.extend(["--search-query", query])

        self._log("command: " + " ".join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            self._log(line.rstrip())
        returncode = process.wait()
        if returncode != 0:
            output = "".join(output_lines)
            raise RuntimeError(
                "Google Maps crawler failed with exit code "
                f"{returncode}\nOUTPUT:\n{output}"
            )
        self._log(f"completed successfully with exit code {returncode}")

    def _log(self, message: str) -> None:
        print(f"[gmaps-crawler] {message}", flush=True)
        if self.log_callback is not None:
            self.log_callback(message)

    def _read_rows(self, restaurant_id: str | None, month: str) -> list[dict[str, Any]]:
        rows = []
        with self.output_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_month = row.get("review_month") or str(row.get("review_time", ""))[:7]
                if row_month != month:
                    continue
                if restaurant_id is not None and self.mode != "benchmark" and row.get("restaurant_id") != restaurant_id:
                    continue
                rows.append(row)
        return rows