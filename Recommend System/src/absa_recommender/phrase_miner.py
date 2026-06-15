from collections import Counter
import re
import unicodedata


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).lower().strip()
    return re.sub(r"\s+", " ", normalized)


def cluster_text(aspect_expression: str, opinion_expression: str) -> str:
    return normalize_text(f"{aspect_expression} | {opinion_expression}")


def top_values(records: list[dict], field: str, limit: int) -> list[str]:
    counts = Counter(
        normalize_text(str(record.get(field, "")))
        for record in records
        if str(record.get(field, "")).strip()
    )
    return [value for value, _ in counts.most_common(limit)]


def top_opinion_phrases(records: list[dict], limit: int) -> list[str]:
    return top_values(records, "opinion_expression", limit)
