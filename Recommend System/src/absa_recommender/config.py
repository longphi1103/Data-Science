from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_label_schema(path: str | Path) -> dict[str, Any]:
    schema = load_yaml(path)
    if "aspects" not in schema or "sentiments" not in schema:
        raise ValueError("Label schema must define aspects and sentiments.")
    return schema


def normalize_aspect_label(label: str, schema: dict[str, Any]) -> str:
    return _normalize_label(label, _label_section(schema, "aspects"))


def normalize_sentiment_label(label: str, schema: dict[str, Any]) -> str:
    return _normalize_label(label, _label_section(schema, "sentiments"))


def validate_aspect_label(label: str, schema: dict[str, Any], strict: bool = True) -> str:
    return _validate_label(label, _label_section(schema, "aspects"), strict=strict)


def validate_sentiment_label(label: str, schema: dict[str, Any], strict: bool = True) -> str:
    return _validate_label(label, _label_section(schema, "sentiments"), strict=strict)


def _label_section(schema: dict[str, Any], name: str) -> dict[str, Any]:
    labels = schema[name]
    if isinstance(labels, dict):
        return labels

    validation = schema.get("validation", {})
    alias_key = "aspect_aliases" if name == "aspects" else "sentiment_aliases"
    unknown_key = "unknown_aspect" if name == "aspects" else "unknown_sentiment"
    return {
        "labels": labels,
        "aliases": schema.get(alias_key, {}),
        "unknown_label": validation.get(unknown_key),
    }


def _validate_label(label: str, section: dict[str, Any], strict: bool) -> str:
    lookup = _label_lookup(section)
    key = _label_key(label)
    if key in lookup:
        return lookup[key]

    if strict:
        allowed = ", ".join(section.get("labels", []))
        raise ValueError(f"Unknown label {label!r}. Allowed labels: {allowed}")
    return _unknown_label(section)


def _normalize_label(label: str, section: dict[str, Any]) -> str:
    lookup = _label_lookup(section)
    key = _label_key(label)
    return lookup.get(key, _unknown_label(section))


def _label_lookup(section: dict[str, Any]) -> dict[str, str]:
    lookup = {_label_key(label): label for label in section.get("labels", [])}
    for alias, canonical in section.get("aliases", {}).items():
        lookup[_label_key(alias)] = canonical
    return lookup


def _unknown_label(section: dict[str, Any]) -> str:
    unknown_label = section.get("unknown_label")
    if unknown_label is None:
        raise ValueError("Label schema section must define an unknown_label.")
    return unknown_label


def _label_key(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").casefold().split())
