import json
from pathlib import Path

from absa_recommender.models import AbsaOutput


def load_absa_jsonl(path: str | Path) -> list[AbsaOutput]:
    records: list[AbsaOutput] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(AbsaOutput.model_validate(json.loads(line)))
    return records
