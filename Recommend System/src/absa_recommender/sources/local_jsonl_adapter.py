import json
from pathlib import Path
from typing import Any


class LocalJsonlAdapter:
    source = "local_jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch_reviews(self, restaurant_id: str | None = None, month: str | None = None) -> list[dict[str, Any]]:
        rows = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if restaurant_id is not None and row.get("restaurant_id") != restaurant_id:
                    continue
                row_month = row.get("review_month") or str(row.get("review_time", ""))[:7]
                if month is not None and row_month != month:
                    continue
                rows.append(row)
        return rows
