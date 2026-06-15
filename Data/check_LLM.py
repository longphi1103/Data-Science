import json
import csv
import argparse
import unicodedata
from collections import Counter, defaultdict


VALID_CATEGORIES = {
    "FOOD QUALITY",
    "FOOD SAFETY"
    "SERVICE",
    "PRICE",
    "AMBIENCE",
    "CLEANLINESS",
    "LOCATION",
    "MENU",
}

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
}


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa Unicode để tránh lỗi do tiếng Việt bị lệch mã.
    Không xóa dấu, không sửa nội dung.
    """
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).strip()


def text_contains(review_text: str, span: str) -> bool:
    """
    Kiểm tra span có nằm trong review không.
    Dùng so khớp lowercase + chuẩn hóa Unicode.
    """
    review_text = normalize_text(review_text).lower()
    span = normalize_text(span).lower()

    if not span:
        return False

    return span in review_text


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                item["_line_no"] = line_no
                data.append(item)
            except json.JSONDecodeError as e:
                data.append({
                    "_line_no": line_no,
                    "_invalid_json": True,
                    "_error": str(e),
                    "_raw": line,
                })

    return data


def get_labels(item: dict):
    """
    Hỗ trợ nhiều tên field để tránh lỗi do LLM sinh không đồng nhất.
    Chuẩn nhất nên là item["labels"].
    """
    labels = item.get("labels", [])

    if labels is None:
        return []

    if not isinstance(labels, list):
        return []

    return labels


def get_expressions(label: dict):
    """
    Hỗ trợ các tên field phổ biến:
    - aspect_expression
    - aspect_expressions
    - expression
    - expressions
    """
    expr = (
        label.get("aspect_expression")
        or label.get("aspect_expressions")
        or label.get("expression")
        or label.get("expressions")
        or []
    )

    if isinstance(expr, str):
        return [expr]

    if isinstance(expr, list):
        return expr

    return []


def get_evidences(label: dict):
    evidence = label.get("evidence", [])

    if isinstance(evidence, str):
        return [evidence]

    if isinstance(evidence, list):
        return evidence

    return []


def validate_item(item: dict):
    issues = []

    line_no = item.get("_line_no")

    if item.get("_invalid_json"):
        issues.append({
            "line_no": line_no,
            "review_id": "",
            "issue_type": "INVALID_JSON",
            "message": item.get("_error", ""),
            "label_index": "",
            "value": item.get("_raw", ""),
        })
        return issues

    review_id = item.get("review_id", "")
    review_text = normalize_text(item.get("review_text", ""))

    if not review_id:
        issues.append({
            "line_no": line_no,
            "review_id": review_id,
            "issue_type": "MISSING_REVIEW_ID",
            "message": "Thiếu review_id",
            "label_index": "",
            "value": "",
        })

    if not review_text:
        issues.append({
            "line_no": line_no,
            "review_id": review_id,
            "issue_type": "MISSING_REVIEW_TEXT",
            "message": "Thiếu review_text",
            "label_index": "",
            "value": "",
        })

    labels = get_labels(item)

    if not isinstance(labels, list):
        issues.append({
            "line_no": line_no,
            "review_id": review_id,
            "issue_type": "INVALID_LABELS_FORMAT",
            "message": "Field labels phải là list",
            "label_index": "",
            "value": str(labels),
        })
        return issues

    seen_label_keys = set()

    for idx, label in enumerate(labels):
        if not isinstance(label, dict):
            issues.append({
                "line_no": line_no,
                "review_id": review_id,
                "issue_type": "INVALID_LABEL_FORMAT",
                "message": "Mỗi label phải là dict/object",
                "label_index": idx,
                "value": str(label),
            })
            continue

        category = normalize_text(label.get("aspect_category", ""))
        sentiment = normalize_text(label.get("sentiment", "")).lower()
        expressions = get_expressions(label)
        evidences = get_evidences(label)

        if category not in VALID_CATEGORIES:
            issues.append({
                "line_no": line_no,
                "review_id": review_id,
                "issue_type": "UNKNOWN_CATEGORY",
                "message": f"Category không hợp lệ: {category}",
                "label_index": idx,
                "value": category,
            })

        if sentiment not in VALID_SENTIMENTS:
            issues.append({
                "line_no": line_no,
                "review_id": review_id,
                "issue_type": "UNKNOWN_SENTIMENT",
                "message": f"Sentiment không hợp lệ: {sentiment}",
                "label_index": idx,
                "value": sentiment,
            })

        if not evidences:
            issues.append({
                "line_no": line_no,
                "review_id": review_id,
                "issue_type": "EMPTY_EVIDENCE",
                "message": "Label không có evidence",
                "label_index": idx,
                "value": "",
            })

        for ev in evidences:
            ev_norm = normalize_text(ev)
            if not ev_norm:
                issues.append({
                    "line_no": line_no,
                    "review_id": review_id,
                    "issue_type": "EMPTY_EVIDENCE_ITEM",
                    "message": "Một evidence bị rỗng",
                    "label_index": idx,
                    "value": ev,
                })
            elif not text_contains(review_text, ev_norm):
                issues.append({
                    "line_no": line_no,
                    "review_id": review_id,
                    "issue_type": "EVIDENCE_NOT_IN_REVIEW",
                    "message": "Evidence không nằm trong review_text",
                    "label_index": idx,
                    "value": ev_norm,
                })

        for expr in expressions:
            expr_norm = normalize_text(expr)
            if not expr_norm:
                issues.append({
                    "line_no": line_no,
                    "review_id": review_id,
                    "issue_type": "EMPTY_EXPRESSION_ITEM",
                    "message": "Một aspect_expression bị rỗng",
                    "label_index": idx,
                    "value": expr,
                })
            elif not text_contains(review_text, expr_norm):
                issues.append({
                    "line_no": line_no,
                    "review_id": review_id,
                    "issue_type": "EXPRESSION_NOT_IN_REVIEW",
                    "message": "aspect_expression không nằm trong review_text",
                    "label_index": idx,
                    "value": expr_norm,
                })

        label_key = (
            category,
            sentiment,
            tuple(sorted([normalize_text(x).lower() for x in expressions])),
            tuple(sorted([normalize_text(x).lower() for x in evidences])),
        )

        if label_key in seen_label_keys:
            issues.append({
                "line_no": line_no,
                "review_id": review_id,
                "issue_type": "DUPLICATE_LABEL",
                "message": "Label bị trùng hoàn toàn",
                "label_index": idx,
                "value": str(label_key),
            })

        seen_label_keys.add(label_key)

    # Kiểm tra cùng một aspect nhưng nhiều sentiment khác nhau
    aspect_to_sentiments = defaultdict(set)

    for label in labels:
        if not isinstance(label, dict):
            continue

        category = normalize_text(label.get("aspect_category", ""))
        sentiment = normalize_text(label.get("sentiment", "")).lower()

        if category in VALID_CATEGORIES and sentiment in VALID_SENTIMENTS:
            aspect_to_sentiments[category].add(sentiment)

    for category, sentiments in aspect_to_sentiments.items():
        if len(sentiments) > 1:
            issues.append({
                "line_no": line_no,
                "review_id": review_id,
                "issue_type": "CONFLICT_SENTIMENT_SAME_ASPECT",
                "message": f"Cùng aspect {category} có nhiều sentiment: {sorted(sentiments)}",
                "label_index": "",
                "value": ",".join(sorted(sentiments)),
            })

    return issues


def write_issues_csv(issues, output_path: str):
    fieldnames = [
        "line_no",
        "review_id",
        "issue_type",
        "message",
        "label_index",
        "value",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for issue in issues:
            writer.writerow(issue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File JSONL nhãn LLM")
    parser.add_argument("--output", default="annotation_issues.csv", help="File CSV lỗi")
    args = parser.parse_args()

    data = load_jsonl(args.input)

    all_issues = []
    for item in data:
        all_issues.extend(validate_item(item))

    write_issues_csv(all_issues, args.output)

    print(f"Đã kiểm tra {len(data)} dòng.")
    print(f"Tổng số lỗi: {len(all_issues)}")

    issue_counter = Counter(issue["issue_type"] for issue in all_issues)

    print("\nThống kê lỗi:")
    for issue_type, count in issue_counter.most_common():
        print(f"- {issue_type}: {count}")

    print(f"\nĐã lưu lỗi vào: {args.output}")


if __name__ == "__main__":
    main()