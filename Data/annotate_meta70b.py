import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Config
MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_INPUT_PATH  = Path(r"./datasets/processed/processed_reviews.csv")
DEFAULT_OUTPUT_PATH = Path(r"./datasets/annotated/annotations_meta70b_v5.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pydantic schema
VALID_CATEGORIES = {"Food Quality", "Food Safety", "Service", "Price", "Cleanliness"}


class ACOSItem(BaseModel):
    aspect_expression: str = Field(
        description="Exact substring in the review; empty string only if no explicit aspect"
    )
    aspect_category: Literal["Food Quality", "Food Safety", "Service", "Price", "Cleanliness"]
    opinion_expression: str = Field(
        description="Exact substring in the review; empty string only if no explicit opinion"
    )
    sentiment: Literal["positive", "negative", "neutral"]

    @model_validator(mode="after")
    def check_not_both_empty(self) -> "ACOSItem":
        if not self.aspect_expression and not self.opinion_expression:
            raise ValueError("Both aspect_expression and opinion_expression cannot be empty.")
        return self


class AnnotationOutput(BaseModel):
    annotations: List[ACOSItem]


class BatchResultItem(BaseModel):
    review_id: str
    annotations: List[ACOSItem]


class BatchAnnotationOutput(BaseModel):
    results: List[BatchResultItem]


SYSTEM_PROMPT = """
Bạn là chuyên gia gán nhãn dữ liệu ABSA (Aspect-Based Sentiment Analysis) cho các đánh giá nhà hàng bằng tiếng Việt.

Nhiệm vụ: Trích xuất tất cả các bộ tứ (quadruple) từ review và trả về JSON hợp lệ.
Mỗi annotation có CHÍNH XÁC 4 trường: aspect_expression, aspect_category, opinion_expression, sentiment.

════════════════════════════════════════
1. ASPECT_EXPRESSION
════════════════════════════════════════
• PHẢI là chuỗi con xuất hiện NGUYÊN VĂN trong review.
• Không tự ý tạo mới hoặc suy diễn ngoài văn bản.
• Nếu khía cạnh ẩn (không xuất hiện trực tiếp): để "".
• Ưu tiên các đối tượng cụ thể: món ăn, đồ uống, nhân viên, giá cả, dụng cụ, khu vực.
• Hạn chế annotate nhận xét quá chung chung không có đối tượng rõ ràng.

════════════════════════════════════════
2. ASPECT_CATEGORY — CHỈ DÙNG 5 GIÁ TRỊ SAU
════════════════════════════════════════

[Food Quality]
  Chất lượng món ăn / đồ uống: mùi vị, độ tươi, cách chế biến, kết cấu, nhiệt độ món ăn, hình thức trình bày.

[Food Safety]
  CHỈ dùng khi review đề cập RÕ RÀNG đến: mất vệ sinh thực phẩm ảnh hưởng sức khỏe, ngộ độc, thực phẩm hỏng, ôi thiu, dị vật trong thức ăn, điều kiện chế biến không an toàn.
  KHÔNG dùng Food Safety cho: khẩu vị cá nhân, yêu cầu chế biến riêng, sở thích nguyên liệu, hiểu nhầm giao tiếp.
  Phân biệt với Cleanliness:
    Cleanliness → vệ sinh nhìn thấy được ở đồ vật/không gian.
    Food Safety  → trực tiếp nhiễm vào thực phẩm / nguy cơ sức khỏe.

[Service]
  Chất lượng phục vụ: thái độ nhân viên, tốc độ phục vụ, thời gian chờ, giao tiếp, xử lý yêu cầu/khiếu nại.
  Ưu tiên Service khi opinion liên quan đến tốc độ, thời gian, giao tiếp, hỗ trợ khách hàng.

[Price]
  Giá cả: chi phí, hóa đơn, phụ phí, mức độ hợp lý, khuyến mãi.

[Cleanliness]
  Độ sạch sẽ của không gian: bàn ghế, dụng cụ ăn uống (nhìn thấy được), sàn nhà, nhà vệ sinh, khu vực ăn uống.

!! NGUYÊN TẮC BẮT BUỘC VỀ CATEGORY !!
  Nếu nhận xét KHÔNG thuộc rõ ràng một trong 5 category: BỎ QUA HOÀN TOÀN. KHÔNG ép vào category gần nhất. Thà bỏ sót còn hơn gán sai.
  Các nhận xét sau ĐÃ BỊ LOẠI TRỪ khỏi taxonomy: không gian, thiết kế, view, cảnh quan, bầu không khí, nhiệt độ môi trường, âm thanh, ánh sáng, cảm giác chung, trải nghiệm tổng thể không rõ đối tượng -> KHÔNG annotate dù opinion rất rõ ràng.

════════════════════════════════════════
3. OPINION_EXPRESSION
════════════════════════════════════════
• PHẢI xuất hiện NGUYÊN VĂN trong review.
• Lấy span NGẮN NHẤT vẫn giữ đầy đủ ý nghĩa đánh giá.
• Không copy nguyên câu dài.

QUY TẮC SPAN:
  ✓ Bao gồm intensifier.
  ✓ Bao gồm phủ định.
  ✗ Không lấy thêm complement so sánh trừ khi bắt buộc.

XỬ LÝ PHỦ ĐỊNH TIẾNG VIỆT:
  "không + tích cực" → sentiment: negative
  "không + tiêu cực" → sentiment: positive
  "chẳng có gì đặc biệt" → sentiment: negative
  "không đến nỗi nào"    → sentiment: positive hoặc neutral
  QUAN TRỌNG: giữ NGUYÊN CỤM PHỦ ĐỊNH trong opinion_expression.

════════════════════════════════════════
4. SENTIMENT — CHỈ DÙNG 3 GIÁ TRỊ
════════════════════════════════════════
  positive → tích cực, hài lòng, khen ngợi.
  negative → tiêu cực, không hài lòng, phàn nàn.
  neutral  → thuần mô tả, không kèm cảm xúc rõ ràng. KHÔNG dùng cho các trường hợp mang sắc thái đánh giá nhẹ như hơi mắc, khá rẻ, tạm ổn, tàm tạm.

════════════════════════════════════════
5. QUY TẮC TRÍCH XUẤT
════════════════════════════════════════

[Đầy đủ]
  Không bỏ sót khía cạnh phù hợp với taxonomy. Một câu chứa nhiều aspect → tạo nhiều annotation độc lập.

[Không suy diễn]
  Không thêm aspect/opinion không xuất hiện trong review. Không annotate từ ngữ cảnh mơ hồ.
[Chống trùng lặp — BẮT BUỘC]
  Một bộ tứ (aspect_expression, aspect_category, opinion_expression, sentiment) chỉ được xuất hiện MỘT LẦN DUY NHẤT trong output.
  aspect_expression, opinion_expression phải copy y nguyên từng ký tự từ review gốc, không viết tắt, không sửa chính tả, không paraphrase và phải là các từ liên tiếp trong review gốc.
  Nếu cùng cụm từ lặp lại nhiều lần trong review → CHỈ tạo 1 annotation, KHÔNG tạo thêm.
  Trước khi output, kiểm tra: có bộ tứ nào giống nhau không? Nếu có → xóa bản trùng, chỉ giữ lại 1.

[Multi-opinion cùng một aspect]
  Một aspect có thể có nhiều opinions khác nhau → nhiều annotation. Mỗi annotation là một bộ tứ độc lập.

[Điều kiện tạo annotation]
  Chỉ tạo khi đồng thời có: (a) khía cạnh thuộc đúng 1 trong 5 category VÀ (b) opinion đánh giá tương ứng rõ ràng.

[Không có annotation hợp lệ]
  Trả về: {"annotations": []}

════════════════════════════════════════
SELF-CHECK TRƯỚC KHI OUTPUT
════════════════════════════════════════
Trước khi trả về JSON, tự kiểm tra:
  □ Có annotation nào bị gán sai category không? (đặc biệt: view/cảnh quan → phải bị loại bỏ)
  □ Có bộ tứ nào trùng lặp không? (dù aspect xuất hiện nhiều lần → chỉ 1 annotation)
  □ opinion_expression có xuất hiện nguyên văn trong review không?
  □ aspect_expression có thuộc đúng 1 trong 5 category không?

Chỉ output sau khi đã xác nhận 4 điều trên.

════════════════════════════════════════
OUTPUT
════════════════════════════════════════
Chỉ trả về JSON hợp lệ. Không giải thích. Không markdown.
Không thêm bất kỳ văn bản nào ngoài JSON.

{
  "annotations": [
    {
      "aspect_expression": "...",
      "aspect_category": "Food Quality|Food Safety|Service|Price|Cleanliness",
      "opinion_expression": "...",
      "sentiment": "positive|negative|neutral"
    }
  ]
}
"""

# Helpers
def load_env_file(env_file: str | None) -> None:
    if load_dotenv is None:
        return
    path = Path(env_file) if env_file else Path(".env")
    if path.exists():
        load_dotenv(path, override=False)


def build_user_prompt(review: str) -> str:
    return f"Review:\n{review}\n\nReturn JSON only."


def build_batch_user_prompt(items: list[dict]) -> str:
    return (
        "Annotate each review below. Return JSON only.\n\n"
        "Input reviews:\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
        + """

Output format:
{
  "results": [
    {
      "review_id": "...",
      "annotations": [
        {
          "aspect_expression": "...",
          "aspect_category": "Food Quality|Food Safety|Service|Price|Cleanliness",
          "opinion_expression": "...",
          "sentiment": "positive|negative|neutral"
        }
      ]
    }
  ]
}
"""
    )


def chunk_rows(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]


def load_done(output_path: Path) -> set[str]:
    """Return set of review_ids already successfully written to the output file."""
    success_ids: set[str] = set()

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        success_ids.add(str(rec.get("review_id")))
                except Exception:
                    pass

    return success_ids


def load_history(output_path: Path) -> set[str]:
    """Return set of review_ids already successfully written to the output file."""
    success_ids: set[str] = set()

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        success_ids.add(str(rec.get("review_id")))
                except Exception:
                    pass

    return success_ids


def validate_spans(annotations: list[dict], review: str) -> list[dict]:
    """
    Post-processing: drop any quadruple whose aspect_expression or
    opinion_expression is non-empty but does NOT appear verbatim in the review.
    Logs a warning for each dropped item.
    """
    clean = []
    for item in annotations:
        ae = item.get("aspect_expression", "")
        oe = item.get("opinion_expression", "")

        ae_ok = (ae == "") or (ae in review)
        oe_ok = (oe == "") or (oe in review)

        if ae_ok and oe_ok:
            clean.append(item)
        else:
            bad = []
            if not ae_ok:
                bad.append(f"aspect_expression='{ae}'")
            if not oe_ok:
                bad.append(f"opinion_expression='{oe}'")
            logger.warning("Dropped non-verbatim span(s): %s", ", ".join(bad))

    return clean

# API call with adaptive back-off
def call_model(
    client: OpenAI,
    review: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> tuple[str, dict, str | None]:
    """
    Returns (status, result_dict, error_str).
    1 initial request + up to max_retries retries.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(review)},
                ],
                temperature=temperature,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            parsed = AnnotationOutput.model_validate(data)
            clean_annotations = validate_spans(
                [a.model_dump() for a in parsed.annotations], review
            )
            return "success", {"annotations": clean_annotations}, None

        except Exception as exc:
            if attempt == max_retries:
                logger.error("All %d retries failed: %s", max_retries, exc)
                return "failed", {"annotations": []}, str(exc)

            wait = base_delay * (2 ** attempt)
            logger.warning(
                "Attempt %d failed (%s). Retrying in %.1fs…",
                attempt + 1, exc, wait
            )
            time.sleep(wait)

    return "failed", {"annotations": []}, "Unknown error"


def call_model_batch(
    client: OpenAI,
    batch_items: list[dict],
    review_map: dict[str, str],
    temperature: float = 0.0,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> tuple[str, dict, str | None]:
    """
    Returns (status, batch_result_dict, error_str).
    batch_result_dict format:
    {
        "review_id": {"annotations": [...]}
    }
    """
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_batch_user_prompt(batch_items)},
                ],
                temperature=temperature,
                max_tokens=1200 * max(1, len(batch_items)),
                response_format={"type": "json_object"},
            )

            content = resp.choices[0].message.content
            data = json.loads(content)
            parsed = BatchAnnotationOutput.model_validate(data)

            output = {}

            for item in parsed.results:
                review_id = str(item.review_id)
                review = review_map.get(review_id, "")

                clean_annotations = validate_spans(
                    [a.model_dump() for a in item.annotations],
                    review
                )

                output[review_id] = {
                    "annotations": clean_annotations
                }

            for item in batch_items:
                review_id = str(item["review_id"])
                if review_id not in output:
                    output[review_id] = {"annotations": []}

            return "success", output, None

        except Exception as exc:
            if attempt == max_retries:
                logger.error("Batch failed after %d retries: %s", max_retries, exc)
                return "failed", {}, str(exc)

            wait = base_delay * (2 ** attempt)
            logger.warning(
                "Batch attempt %d failed (%s). Retrying in %.1fs…",
                attempt + 1, exc, wait
            )
            time.sleep(wait)

    return "failed", {}, "Unknown error"

# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate restaurant reviews and write JSONL incrementally."
    )
    parser.add_argument("--input",      default=str(DEFAULT_INPUT_PATH),  help="File CSV đầu vào.")
    parser.add_argument("--output",     default=str(DEFAULT_OUTPUT_PATH), help="File JSONL đầu ra.")
    parser.add_argument("--review_col", default="review_text",            help="Tên cột chứa nội dung review.")
    parser.add_argument("--id_col",     default="review_id",              help="Tên cột chứa ID review.")
    parser.add_argument("--limit",      type=int, default=None,           help="Giới hạn số dòng để chạy thử.")
    parser.add_argument("--delay",      type=float, default=1.0,          help="Số giây chờ giữa các request/batch (default: 1.0).")
    parser.add_argument("--batch_size", type=int, default=1,              help="Số review trong mỗi request (default: 1).")
    parser.add_argument("--base_url",
                        default=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                        help="Base URL của API.")
    parser.add_argument("--env_file",
                        default=".env",
                        help="File .env chứa biến môi trường.")
    args = parser.parse_args()

    load_env_file(args.env_file)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY environment variable.")

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit)

    success_ids = load_history(output_path)

    df_ids = df[args.id_col].astype(str)
    pending_mask = ~df_ids.isin(success_ids)
    pending = df[pending_mask].copy()

    logger.info(
        "Total: %d | Success done: %d | Pending: %d",
        len(df), len(success_ids), len(pending),
    )

    counts = {"success": 0, "failed": 0, "skipped": len(success_ids)}

    batch_size = max(1, args.batch_size)

    if batch_size == 1:
        iterator = (
            tqdm(pending.iterrows(), total=len(pending), desc="Annotating", unit="review")
            if HAS_TQDM else pending.iterrows()
        )

        with output_path.open("a", encoding="utf-8") as out:
            for _, row in iterator:
                review_id = str(row[args.id_col])
                review = str(row[args.review_col])

                status, result, error = call_model(client, review)
                counts[status] = counts.get(status, 0) + 1

                if status == "success":
                    record = {
                        "review_id": review_id,
                        "restaurant_id": row.get("restaurant_id", None),
                        "restaurant_name": row.get("restaurant_name", None),
                        "stars": row.get("stars", None),
                        "review_text": review,
                        "model": MODEL,
                        "status": status,
                        "annotations": result["annotations"],
                        "error": error,
                    }

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()

                    n = len(record["annotations"])
                    if not HAS_TQDM:
                        logger.info("✓ %s | %d annotation(s)", review_id, n)
                    else:
                        iterator.set_postfix({"id": review_id, "ann": n, "status": status})  # type: ignore[union-attr]
                else:
                    logger.warning("Skipped after retries: %s", review_id)
                    if not HAS_TQDM:
                        logger.info("✗ %s | skipped", review_id)
                    else:
                        iterator.set_postfix({"id": review_id, "status": "skipped"})  # type: ignore[union-attr]

                time.sleep(args.delay)

    else:
        batches = list(chunk_rows(pending, batch_size))

        iterator = (
            tqdm(batches, total=len(batches), desc="Annotating", unit="batch")
            if HAS_TQDM else batches
        )

        with output_path.open("a", encoding="utf-8") as out:
            for batch_df in iterator:
                batch_items = []
                review_map = {}

                for _, row in batch_df.iterrows():
                    review_id = str(row[args.id_col])
                    review = str(row[args.review_col])

                    batch_items.append({
                        "review_id": review_id,
                        "review": review
                    })

                    review_map[review_id] = review

                status, batch_result, error = call_model_batch(
                    client=client,
                    batch_items=batch_items,
                    review_map=review_map,
                )

                if status == "success":
                    counts["success"] = counts.get("success", 0) + len(batch_df)

                    for _, row in batch_df.iterrows():
                        review_id = str(row[args.id_col])
                        review = str(row[args.review_col])

                        result = batch_result.get(review_id, {"annotations": []})

                        record = {
                            "review_id": review_id,
                            "restaurant_id": row.get("restaurant_id", None),
                            "restaurant_name": row.get("restaurant_name", None),
                            "stars": row.get("stars", None),
                            "review_text": review,
                            "model": MODEL,
                            "status": status,
                            "annotations": result["annotations"],
                            "error": error,
                        }

                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()

                    if not HAS_TQDM:
                        logger.info("✓ batch | %d review(s)", len(batch_df))
                    else:
                        iterator.set_postfix({
                            "batch_size": len(batch_df),
                            "status": status
                        })  # type: ignore[union-attr]

                else:
                    counts["failed"] = counts.get("failed", 0) + len(batch_df)

                    for _, row in batch_df.iterrows():
                        review_id = str(row[args.id_col])
                        logger.warning("Skipped after retries: %s", review_id)

                    if not HAS_TQDM:
                        logger.info("✗ batch | skipped")
                    else:
                        iterator.set_postfix({
                            "batch_size": len(batch_df),
                            "status": "skipped"
                        })  # type: ignore[union-attr]

                time.sleep(args.delay)

    logger.info(
        "Done. success=%d | failed=%d | skipped=%d",
        counts["success"], counts["failed"], counts["skipped"],
    )


if __name__ == "__main__":
    main()