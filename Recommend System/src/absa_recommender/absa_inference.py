import hashlib
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Protocol

from absa_recommender.config import load_label_schema, load_yaml
from absa_recommender.schemas import ABSAAnnotation, ABSAReview


class ABSAInferenceAdapter(Protocol):
    model_version: str

    def infer(self, reviews: list[dict[str, Any]]) -> list[ABSAReview]:
        ...


class ExternalABSAInferenceNotConfigured(RuntimeError):
    pass


class PreAnnotatedABSAAdapter:
    model_version = "preannotated-jsonl"

    def infer(self, reviews: list[dict[str, Any]]) -> list[ABSAReview]:
        return [ABSAReview.model_validate(review) for review in reviews]


class ViT5ABSAAdapter:
    """Adapter for the trained ACOS/ABSA ViT5 seq2seq model packaged as a Hugging Face zip."""

    model_version = "acos-vit5-large-final"

    def __init__(
        self,
        model_zip_path: str | Path | None = None,
        config_path: str | Path = "configs/absa_model.yaml",
        label_schema_path: str | Path = "configs/label_schema.yaml",
    ) -> None:
        self.config = load_yaml(config_path)
        model_config = self.config.get("model", {})
        inference_config = self.config.get("inference", {})
        configured_model_zip_path = (
            model_zip_path
            or inference_config.get("model_zip_path")
            or model_config.get("model_zip_path")
            or model_config.get("zip_path")
            or "models/acos_vit5_large_final.zip"
        )
        self.model_zip_path = Path(configured_model_zip_path)
        self.model_version = str(model_config.get("version", self.model_version))
        schema = load_label_schema(label_schema_path)
        self.aspects = set(schema.get("aspects", []))
        self.sentiments = set(schema.get("sentiments", []))
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._model = None
        self._tokenizer = None

    def infer(self, reviews: list[dict[str, Any]]) -> list[ABSAReview]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        inference_config = self.config.get("inference", {})
        batch_size = max(1, int(inference_config.get("batch_size", 8)))
        max_input_length = int(inference_config.get("max_input_length", 512))
        max_new_tokens = int(inference_config.get("max_new_tokens", 256))
        num_beams = max(1, int(inference_config.get("num_beams", 4)))
        fallback_to_cpu = bool(inference_config.get("cuda_oom_fallback_to_cpu", True))
        outputs: list[ABSAReview] = []
        for start in range(0, len(reviews), batch_size):
            batch = reviews[start : start + batch_size]
            generated, decoded, confidences = self._generate_batch(
                batch,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                fallback_to_cpu=fallback_to_cpu,
            )
            outputs.extend(
                self._review_from_generation(review, text, confidence)
                for review, text, confidence in zip(batch, decoded, confidences, strict=True)
            )
        return outputs

    def _generate_batch(
        self,
        batch: list[dict[str, Any]],
        *,
        max_input_length: int,
        max_new_tokens: int,
        num_beams: int,
        fallback_to_cpu: bool,
    ) -> tuple[Any, list[str], list[float]]:
        assert self._model is not None
        assert self._tokenizer is not None

        prompts = [self._prompt(review) for review in batch]
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        try:
            return self._generate_encoded(
                encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        except RuntimeError as error:
            if not fallback_to_cpu or not _is_cuda_oom_error(error):
                raise

            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

            self._model.to("cpu")
            encoded = {key: value.cpu() for key, value in encoded.items()}
            return self._generate_encoded(
                encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

    def _generate_encoded(
        self,
        encoded: dict[str, Any],
        *,
        max_new_tokens: int,
        num_beams: int,
    ) -> tuple[Any, list[str], list[float]]:
        assert self._model is not None
        assert self._tokenizer is not None

        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}

        try:
            import torch
        except ImportError:
            torch = None

        if torch is None:
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
            )
        else:
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        decoded = self._tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
        confidences = self._sequence_confidences(generated)
        return generated, decoded, confidences

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.model_zip_path.exists():
            raise ExternalABSAInferenceNotConfigured(f"Model zip not found: {self.model_zip_path}")

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5TokenizerFast
        except ImportError as error:
            raise ExternalABSAInferenceNotConfigured(
                "The trained ViT5 adapter requires torch, transformers, sentencepiece and safetensors."
            ) from error

        self._tempdir = tempfile.TemporaryDirectory(prefix="absa_vit5_")
        model_dir = Path(self._tempdir.name)
        with zipfile.ZipFile(self.model_zip_path) as archive:
            archive.extractall(model_dir)

        self._tokenizer = self._load_tokenizer(model_dir, AutoTokenizer, T5TokenizerFast)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        configured_device = str(self.config.get("inference", {}).get("device", "auto")).strip().lower()
        if configured_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif configured_device in {"cuda", "cpu"}:
            device = configured_device
        else:
            raise ExternalABSAInferenceNotConfigured(
                f"Unsupported inference.device={configured_device!r}; expected auto, cuda or cpu."
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ExternalABSAInferenceNotConfigured("inference.device is cuda but CUDA is not available.")
        self._model.to(device)
        self._model.eval()

    def _load_tokenizer(self, model_dir: Path, auto_tokenizer: Any, t5_tokenizer_fast: Any) -> Any:
        try:
            return auto_tokenizer.from_pretrained(model_dir)
        except TypeError as error:
            if "argument 'vocab'" not in str(error):
                raise

            tokenizer_json = model_dir / "tokenizer.json"
            if not tokenizer_json.exists():
                raise ExternalABSAInferenceNotConfigured(
                    "The trained ViT5 tokenizer metadata is incompatible with this transformers "
                    "version, and tokenizer.json was not found for fallback loading."
                ) from error

            return t5_tokenizer_fast(tokenizer_file=str(tokenizer_json), extra_ids=100)

    def _sequence_confidences(self, generated: Any) -> list[float]:
        assert self._model is not None
        assert self._tokenizer is not None

        if not getattr(generated, "scores", None):
            return [float(self.config.get("inference", {}).get("default_confidence", 0.80))] * len(
                generated.sequences
            )

        transition_scores = self._model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        pad_token_id = self._tokenizer.pad_token_id
        eos_token_id = self._tokenizer.eos_token_id

        confidences: list[float] = []
        for sequence, token_scores in zip(generated.sequences, transition_scores, strict=True):
            usable_scores = []
            score_offset = len(sequence) - len(token_scores)
            for index, log_probability in enumerate(token_scores):
                token_id = int(sequence[score_offset + index])
                if pad_token_id is not None and token_id == pad_token_id:
                    continue
                if eos_token_id is not None and token_id == eos_token_id:
                    continue
                usable_scores.append(float(log_probability))
            if not usable_scores:
                confidences.append(0.0)
                continue
            mean_log_probability = sum(usable_scores) / len(usable_scores)
            confidences.append(max(0.0, min(1.0, math.exp(mean_log_probability))))
        return confidences

    def _prompt(self, review: dict[str, Any]) -> str:
        template = str(
            self.config.get("inference", {}).get(
                "prompt_template",
                "Trích xuất các bộ (aspect, sentiment, aspect_term, opinion) từ review nhà hàng: {review_text}",
            )
        )
        text = str(review.get("review_text") or review.get("text") or "")
        return template.format(review_text=text)

    def _review_from_generation(
        self,
        review: dict[str, Any],
        generation: str,
        confidence: float,
    ) -> ABSAReview:
        annotations = [
            annotation
            for item in _parse_generation_items(generation)
            if (annotation := self._annotation_from_item(item, confidence)) is not None
        ]
        if not annotations:
            annotations = [
                ABSAAnnotation(
                    aspect_expression="",
                    aspect_category="Unknown" if "Unknown" in self.aspects else next(iter(self.aspects)),
                    opinion_expression=_opinion_snippet(
                        str(review.get("review_text") or review.get("text") or "")
                    ),
                    sentiment="neutral",
                    model_confidence=0.30,
                )
            ]
        text = str(review.get("review_text") or review.get("text") or "")
        return ABSAReview(
            review_id=str(review.get("review_id") or review.get("source_review_id") or _stable_id(text)),
            review_text=text,
            restaurant_id=review.get("restaurant_id"),
            restaurant_name=review.get("restaurant_name") or review.get("name"),
            rating=review.get("rating"),
            review_time=review.get("review_time"),
            review_month=review.get("review_month"),
            annotations=annotations,
        )

    def _annotation_from_item(
        self,
        item: dict[str, Any],
        confidence: float,
    ) -> ABSAAnnotation | None:
        aspect = _normalize_aspect(str(item.get("aspect") or item.get("aspect_category") or ""))
        sentiment = _normalize_sentiment(str(item.get("sentiment") or item.get("polarity") or ""))
        if aspect not in self.aspects:
            aspect = "Unknown" if "Unknown" in self.aspects else ""
        if sentiment not in self.sentiments:
            sentiment = "neutral"
        if not aspect:
            return None
        return ABSAAnnotation(
            aspect_expression=str(item.get("aspect_term") or item.get("aspect_expression") or ""),
            aspect_category=aspect,
            opinion_expression=str(
                item.get("opinion") or item.get("opinion_expression") or item.get("opinion_text") or ""
            ),
            sentiment=sentiment,
            model_confidence=confidence,
        )


class PlaceholderABSAAdapter:
    """Deterministic placeholder adapter with the same contract as a trained ABSA model."""

    model_version = "placeholder-rule-absa-v0"

    def __init__(self, label_schema_path: str = "configs/label_schema.yaml") -> None:
        schema = load_label_schema(label_schema_path)
        self.aspects = set(schema.get("aspects", []))

    def infer(self, reviews: list[dict[str, Any]]) -> list[ABSAReview]:
        return [self._infer_one(review) for review in reviews]

    def _infer_one(self, review: dict[str, Any]) -> ABSAReview:
        text = str(review.get("review_text") or review.get("text") or "")
        rating = review.get("rating")
        annotations = [
            ABSAAnnotation(
                aspect_expression=match["term"],
                aspect_category=match["aspect"],
                opinion_expression=match["opinion"],
                sentiment=_sentiment(text, rating),
                model_confidence=match["confidence"],
            )
            for match in self._aspect_matches(text)
        ]
        if not annotations:
            annotations = [
                ABSAAnnotation(
                    aspect_expression="",
                    aspect_category="Unknown" if "Unknown" in self.aspects else next(iter(self.aspects)),
                    opinion_expression=_opinion_snippet(text),
                    sentiment=_sentiment(text, rating),
                    model_confidence=0.35,
                )
            ]
        return ABSAReview(
            review_id=str(review.get("review_id") or review.get("source_review_id") or _stable_id(text)),
            review_text=text,
            restaurant_id=review.get("restaurant_id"),
            restaurant_name=review.get("restaurant_name") or review.get("name"),
            rating=rating,
            review_time=review.get("review_time"),
            review_month=review.get("review_month"),
            annotations=annotations,
        )

    def _aspect_matches(self, text: str) -> list[dict[str, Any]]:
        normalized = text.lower()
        matches = []
        for aspect, patterns in _ASPECT_PATTERNS.items():
            if aspect not in self.aspects:
                continue
            for pattern in patterns:
                if pattern in normalized:
                    matches.append(
                        {
                            "aspect": aspect,
                            "term": pattern,
                            "opinion": _opinion_snippet(text),
                            "confidence": 0.58,
                        }
                    )
                    break
        if matches:
            return matches
        if "Food Quality" in self.aspects and text.strip():
            return [
                {
                    "aspect": "Food Quality",
                    "term": "",
                    "opinion": _opinion_snippet(text),
                    "confidence": 0.42,
                }
            ]
        return []


def infer_absa_with_adapter(
    reviews: list[dict[str, Any]],
    adapter: ABSAInferenceAdapter | None = None,
) -> list[ABSAReview]:
    if adapter is None:
        adapter = PreAnnotatedABSAAdapter()
    return adapter.infer(reviews)


def build_absa_adapter(name: str) -> ABSAInferenceAdapter:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"preannotated", "preannotated-jsonl"}:
        return PreAnnotatedABSAAdapter()
    if normalized in {"placeholder", "placeholder-rule", "rule", "stub"}:
        return PlaceholderABSAAdapter()
    if normalized in {"vit5", "acos-vit5", "trained", "trained-vit5", "acos-vit5-large-final"}:
        return ViT5ABSAAdapter()
    raise ExternalABSAInferenceNotConfigured(f"Unknown ABSA adapter: {name}")


_ASPECT_PATTERNS = {
    "Food Quality": ["món", "đồ ăn", "thức ăn", "phở", "bún", "cơm", "ngon", "dở", "nguội"],
    "Food Safety": ["ngộ độc", "đau bụng", "vệ sinh an toàn", "ôi thiu", "mốc"],
    "Service": ["nhân viên", "phục vụ", "chờ", "order", "gọi món", "thái độ"],
    "Price": ["giá", "đắt", "rẻ", "hóa đơn", "phí"],
    "Cleanliness": ["bẩn", "dơ", "bàn", "nhà vệ sinh", "mùi"],
    "Ambience": ["không gian", "ồn", "nhạc", "ánh sáng", "nóng", "lạnh"],
    "Location": ["địa điểm", "vị trí", "gửi xe", "đậu xe", "khó tìm"],
    "Menu": ["menu", "thực đơn", "hết món", "ít món"],
}

_NEGATIVE_PATTERNS = [
    "tệ",
    "dở",
    "bẩn",
    "đắt",
    "chậm",
    "lâu",
    "không ngon",
    "khó chịu",
    "ngộ độc",
    "đau bụng",
]
_POSITIVE_PATTERNS = ["ngon", "tốt", "thân thiện", "sạch", "hợp lý", "nhanh"]


def _is_cuda_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "cuda" in message and ("out of memory" in message or "memoryallocation" in message)


def _sentiment(text: str, rating: Any) -> str:
    normalized = text.lower()
    if any(pattern in normalized for pattern in _NEGATIVE_PATTERNS):
        return "negative"
    if any(pattern in normalized for pattern in _POSITIVE_PATTERNS):
        return "positive"
    if isinstance(rating, int | float):
        if rating <= 2:
            return "negative"
        if rating >= 4:
            return "positive"
    return "neutral"


def _parse_generation_items(generation: str) -> list[dict[str, Any]]:
    text = generation.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("annotations", "aspects", "items"):
                if isinstance(payload.get(key), list):
                    return [item for item in payload[key] if isinstance(item, dict)]
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    pseudo_json_items = _parse_flat_acos_key_value_sequence(text)
    if pseudo_json_items:
        return pseudo_json_items

    items = []
    for segment in text.replace("\n", ";").split(";"):
        parts = [part.strip() for part in segment.split("|")]
        if len(parts) >= 2:
            items.append(
                {
                    "aspect": parts[0],
                    "sentiment": parts[1],
                    "aspect_term": parts[2] if len(parts) > 2 else "",
                    "opinion": parts[3] if len(parts) > 3 else "",
                }
            )
    return items


def _parse_flat_acos_key_value_sequence(text: str) -> list[dict[str, Any]]:
    """Parse ACOS generations like:
    ["aspect_expression": "không gian", "aspect_category": "Ambience", "sentiment": "positive", ...]
    The trained notebooks emit this as a flat sequence of repeated key/value triples, not valid JSON.
    """
    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', text)
    if not pairs:
        return []

    items: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    item_boundary_keys = {"aspect_expression", "aspect_term", "aspect"}

    for key, value in pairs:
        normalized_key = key.strip()
        if normalized_key in item_boundary_keys and current:
            items.append(current)
            current = {}
        current[normalized_key] = value

    if current:
        items.append(current)

    return [
        item
        for item in items
        if any(
            item.get(key) is not None
            for key in ("aspect", "aspect_category", "sentiment", "polarity", "aspect_expression")
        )
    ]


def _normalize_aspect(value: str) -> str:
    lookup = {
        "food quality": "Food Quality",
        "food_quality": "Food Quality",
        "quality": "Food Quality",
        "food safety": "Food Safety",
        "food_safety": "Food Safety",
        "safety": "Food Safety",
        "service": "Service",
        "price": "Price",
        "cleanliness": "Cleanliness",
        "ambience": "Ambience",
        "ambiance": "Ambience",
        "location": "Location",
        "menu": "Menu",
        "unknown": "Unknown",
    }
    return lookup.get(value.strip().lower(), value.strip())


def _normalize_sentiment(value: str) -> str:
    lookup = {
        "positive": "positive",
        "pos": "positive",
        "negative": "negative",
        "neg": "negative",
        "neutral": "neutral",
        "neu": "neutral",
    }
    return lookup.get(value.strip().lower(), value.strip().lower())


def _opinion_snippet(text: str) -> str:
    compact = " ".join(text.split())
    return compact[:120]


def _stable_id(text: str) -> str:
    return "raw_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
