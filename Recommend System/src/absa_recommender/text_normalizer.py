import html
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ue000-\uf8ff\ufffc\ufffd]"
)
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_SENTENCE_PUNCT_RE = re.compile(r"([.!?]){2,}")
_DECORATIVE_CHARS_RE = re.compile(r"[~*_=\-]{3,}")
_EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u27bf"
    "\ufe0f"
    "\u200d"
    "]+",
    flags=re.UNICODE,
)
_WORD_RE = re.compile(r"(?<![\wÀ-ỹ]){pattern}(?![\wÀ-ỹ])", flags=re.IGNORECASE)


def normalize_review_text(text: str, config: dict[str, Any] | None = None) -> str:
    """Normalize Vietnamese restaurant review text with lightweight rule-based steps.

    The function intentionally avoids training, word segmentation, and aggressive
    stopword removal so ABSA models still receive natural Vietnamese sentences.
    """

    cfg = config or load_text_normalization_config()
    if not cfg.get("enabled", True):
        return str(text or "")

    normalized = str(text or "")
    normalized = html.unescape(normalized)
    normalized = unicodedata.normalize(str(cfg.get("unicode", {}).get("form", "NFC")), normalized)

    if cfg.get("special_chars", {}).get("remove_control_chars", True):
        normalized = _CONTROL_CHARS_RE.sub(" ", normalized)

    if cfg.get("case", {}).get("lowercase", True):
        normalized = normalized.lower()

    normalized = _collapse_whitespace(normalized)

    if cfg.get("emoji", {}).get("mode", "remove") == "remove":
        normalized = _EMOJI_RE.sub(" ", normalized)

    punctuation_cfg = cfg.get("punctuation", {})
    if punctuation_cfg.get("normalize_repeated", True):
        normalized = _normalize_repeated_punctuation(normalized)
        normalized = _DECORATIVE_CHARS_RE.sub(" ", normalized)
    if punctuation_cfg.get("remove", False):
        if punctuation_cfg.get("keep_sentence_marks", True):
            normalized = re.sub(r"[^\w\sÀ-ỹ.!?]", " ", normalized, flags=re.UNICODE)
        else:
            normalized = re.sub(r"[^\w\sÀ-ỹ]", " ", normalized, flags=re.UNICODE)

    normalized = _collapse_whitespace(normalized)
    normalized = _apply_replacements(normalized, cfg.get("phrase_replacements", {}), phrase=True)
    normalized = _apply_replacements(normalized, cfg.get("replacements", {}), phrase=False)

    stopwords_cfg = cfg.get("stopwords", {})
    if stopwords_cfg.get("enabled", False):
        normalized = _remove_stopwords(normalized, stopwords_cfg)

    normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
    return _collapse_whitespace(normalized)


@lru_cache(maxsize=1)
def load_text_normalization_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "text_normalization.yaml"
    if not config_path.exists():
        return {"enabled": True}
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded if isinstance(loaded, dict) else {"enabled": True}


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_repeated_punctuation(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        chars = match.group(0)
        if "!" in chars:
            return "!"
        if "?" in chars:
            return "?"
        return "."

    return _REPEATED_SENTENCE_PUNCT_RE.sub(replace, text)


def _apply_replacements(text: str, replacements: dict[str, Any], *, phrase: bool) -> str:
    normalized = text
    for source, target in sorted(replacements.items(), key=lambda item: len(str(item[0])), reverse=True):
        source_text = _collapse_whitespace(str(source).lower())
        target_text = str(target)
        if not source_text:
            continue
        if phrase or " " in source_text:
            pattern = re.compile(
                rf"(?<![\wÀ-ỹ]){re.escape(source_text)}(?![\wÀ-ỹ])",
                flags=re.IGNORECASE,
            )
        else:
            pattern = re.compile(
                rf"(?<![\wÀ-ỹ]){re.escape(source_text)}(?![\wÀ-ỹ])",
                flags=re.IGNORECASE,
            )
        normalized = pattern.sub(target_text, normalized)
    return _collapse_whitespace(normalized)


def _remove_stopwords(text: str, stopwords_cfg: dict[str, Any]) -> str:
    stopwords = {str(word).lower() for word in stopwords_cfg.get("words", [])}
    protected = {str(word).lower() for word in stopwords_cfg.get("protected_words", [])}
    removable = stopwords - protected
    if not removable:
        return text
    tokens = [token for token in text.split() if token.lower() not in removable]
    return " ".join(tokens)