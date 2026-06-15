import json
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict


HIGH_RISK_ISSUES = {
    "EMPTY_ANNOTATIONS",
    "MISSING_OR_NULL_ANNOTATIONS",
    "UNKNOWN_ASPECT_CATEGORY",
    "UNKNOWN_SENTIMENT",
    "MULTIPLE_CATEGORIES_IN_ONE_ANNOTATION",
    "ASPECT_EXPRESSION_NOT_IN_REVIEW",
    "OPINION_EXPRESSION_NOT_IN_REVIEW",
    "EMPTY_ASPECT_EXPRESSION",
    "EMPTY_OPINION_EXPRESSION",
    "CONFLICT_SENTIMENT_SAME_ASPECT",
}


RISK_CONNECTORS = [
    "nhưng",
    "mà",
    "tuy nhiên",
    "mặc dù",
    "dù",
    "trong khi",
    "cuối cùng",
    "so với",
]


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if not item.get("review_id"):
                item["review_id"] = f"line_{line_no}"

            item["_line_no"] = line_no
            data.append(item)

    return data


def load_issues_csv(path):
    issues_by_review = defaultdict(set)

    if not path:
        return issues_by_review

    path = Path(path)

    if not path.exists():
        print(f"Không tìm thấy file issues: {path}")
        return issues_by_review

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            review_id = str(row.get("review_id", "")).strip()
            issue_type = str(row.get("issue_type", "")).strip()

            if review_id and issue_type:
                issues_by_review[review_id].add(issue_type)

    return issues_by_review


def get_annotations(item):
    annotations = item.get("annotations", [])
    if isinstance(annotations, list):
        return annotations
    return []


def get_categories(item):
    categories = set()

    for ann in get_annotations(item):
        if not isinstance(ann, dict):
            continue

        cat = ann.get("aspect_category")
        if cat:
            categories.add(str(cat).strip())

    return categories


def get_sentiments(item):
    sentiments = set()

    for ann in get_annotations(item):
        if not isinstance(ann, dict):
            continue

        sent = ann.get("sentiment")
        if sent:
            sentiments.add(str(sent).lower().strip())

    return sentiments


def compute_risk_score(item, issues):
    score = 0

    review_text = str(item.get("review_text", "")).lower()
    annotations = get_annotations(item)
    categories = get_categories(item)
    sentiments = get_sentiments(item)

    for issue in issues:
        if issue in HIGH_RISK_ISSUES:
            score += 4
        else:
            score += 1

    if len(annotations) == 0:
        score += 6

    if len(annotations) >= 4:
        score += 3

    if len(categories) >= 3:
        score += 3

    if "Menu" in categories:
        score += 3
    if "Cleanliness" in categories:
        score += 2
    if "Location" in categories:
        score += 2
    if "Food Safety" in categories:
        score += 2

    if "positive" in sentiments and "negative" in sentiments:
        score += 3

    if "neutral" in sentiments:
        score += 2

    if len(review_text) > 500:
        score += 2

    if any(conn in review_text for conn in RISK_CONNECTORS):
        score += 2

    return score


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_target_categories(text):
    if not text:
        return None

    parts = text.replace(";", ",").split(",")
    return {p.strip() for p in parts if p.strip()}


def add_selected(x, selected, selected_ids):
    selected.append(x)
    selected_ids.add(x["review_id"])


def count_covered(selected, field_name):
    counts = defaultdict(int)

    for x in selected:
        for value in x[field_name]:
            counts[value] += 1

    return counts


def select_for_coverage(
    enriched,
    selected,
    selected_ids,
    target_values,
    field_name,
    min_per_value,
    max_size,
    label_name,
):
    missing = set()

    for value in sorted(target_values):
        while True:
            covered_counts = count_covered(selected, field_name)

            if covered_counts[value] >= min_per_value:
                break

            if len(selected) >= max_size:
                print(f"Cảnh báo: Đã đủ size={max_size}, không thể bao phủ thêm {label_name}: {value}")
                missing.add(value)
                break

            candidates = [
                x for x in enriched
                if x["review_id"] not in selected_ids and value in x[field_name]
            ]

            if not candidates:
                missing.add(value)
                break

            best = max(
                candidates,
                key=lambda x: (
                    x["risk_score"],
                    len(x["issues"]),
                    len(x["categories"]),
                    len(x["sentiments"]),
                )
            )

            add_selected(best, selected, selected_ids)

    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File JSONL gốc, ví dụ last_cleaned.jsonl")
    parser.add_argument("--issues", default=None, help="File annotation_issues.csv từ validate")
    parser.add_argument("--output", default="calibration_set.jsonl")
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_per_issue", type=int, default=1)
    parser.add_argument("--min_per_category", type=int, default=1)
    parser.add_argument("--min_per_sentiment", type=int, default=1)

    parser.add_argument(
        "--target_categories",
        default=None,
        help="Danh sách category cần bao phủ, cách nhau bằng dấu phẩy hoặc chấm phẩy. Nếu bỏ trống thì lấy tất cả category có trong dữ liệu."
    )

    args = parser.parse_args()
    random.seed(args.seed)

    data = load_jsonl(args.input)
    issues_by_review = load_issues_csv(args.issues)

    enriched = []

    all_issue_types = set()
    all_categories = set()
    all_sentiments = set()

    for item in data:
        rid = str(item.get("review_id"))
        issues = issues_by_review.get(rid, set())
        categories = get_categories(item)
        sentiments = get_sentiments(item)
        score = compute_risk_score(item, issues)

        all_issue_types.update(issues)
        all_categories.update(categories)
        all_sentiments.update(sentiments)

        item["_calibration_info"] = {
            "risk_score": score,
            "validation_issues": sorted(issues),
            "instruction": "Reviewer kiểm tra và sửa trực tiếp field annotations theo guideline."
        }

        enriched.append({
            "item": item,
            "review_id": rid,
            "risk_score": score,
            "issues": issues,
            "categories": categories,
            "sentiments": sentiments,
        })

    target_categories = parse_target_categories(args.target_categories)
    if target_categories is None:
        target_categories = all_categories

    target_sentiments = {"positive", "negative", "neutral"} & all_sentiments

    selected = []
    selected_ids = set()

    # Bước 1: chọn để bao phủ tất cả lỗi validate
    missing_issues = select_for_coverage(
        enriched=enriched,
        selected=selected,
        selected_ids=selected_ids,
        target_values=all_issue_types,
        field_name="issues",
        min_per_value=args.min_per_issue,
        max_size=args.size,
        label_name="issue"
    )

    # Bước 2: chọn để bao phủ tất cả aspect category
    missing_categories = select_for_coverage(
        enriched=enriched,
        selected=selected,
        selected_ids=selected_ids,
        target_values=target_categories,
        field_name="categories",
        min_per_value=args.min_per_category,
        max_size=args.size,
        label_name="category"
    )

    # Bước 3: chọn để bao phủ sentiment
    missing_sentiments = select_for_coverage(
        enriched=enriched,
        selected=selected,
        selected_ids=selected_ids,
        target_values=target_sentiments,
        field_name="sentiments",
        min_per_value=args.min_per_sentiment,
        max_size=args.size,
        label_name="sentiment"
    )

    # Bước 4: phần còn lại ưu tiên review có risk_score cao
    high_risk_target = int(args.size * 0.6)

    sorted_by_risk = sorted(enriched, key=lambda x: x["risk_score"], reverse=True)

    for x in sorted_by_risk:
        if len(selected) >= high_risk_target:
            break

        if x["review_id"] not in selected_ids:
            add_selected(x, selected, selected_ids)

    # Bước 5: nếu vẫn thiếu thì lấy random để tăng tính đại diện
    remaining = [x for x in enriched if x["review_id"] not in selected_ids]
    need_more = args.size - len(selected)

    if need_more > 0:
        random_samples = random.sample(remaining, min(need_more, len(remaining)))
        for x in random_samples:
            add_selected(x, selected, selected_ids)

    random.shuffle(selected)

    calibration_items = [x["item"] for x in selected]

    write_jsonl(calibration_items, args.output)

    print("===== DONE =====")
    print(f"Tổng review gốc: {len(data)}")
    print(f"Số review calibration: {len(calibration_items)}")
    print(f"Đã xuất file: {args.output}")

    # Thống kê nhanh
    cat_count = defaultdict(int)
    sent_count = defaultdict(int)
    issue_count = defaultdict(int)
    empty_count = 0

    for item in calibration_items:
        rid = str(item.get("review_id"))
        anns = get_annotations(item)

        if len(anns) == 0:
            empty_count += 1

        for cat in get_categories(item):
            cat_count[cat] += 1

        for sent in get_sentiments(item):
            sent_count[sent] += 1

        for issue in issues_by_review.get(rid, set()):
            issue_count[issue] += 1

    print("\n===== COVERAGE REPORT =====")

    print("\nLỗi validate trong calibration:")
    for issue in sorted(all_issue_types):
        print(f"- {issue}: {issue_count.get(issue, 0)}")

    if missing_issues:
        print("\nCảnh báo: Các lỗi chưa được bao phủ:")
        for issue in sorted(missing_issues):
            print(f"- {issue}")

    print("\nAspect category trong calibration:")
    for cat in sorted(target_categories):
        print(f"- {cat}: {cat_count.get(cat, 0)}")

    if missing_categories:
        print("\nCảnh báo: Các category chưa được bao phủ:")
        for cat in sorted(missing_categories):
            print(f"- {cat}")

    print("\nSentiment trong calibration:")
    for sent in sorted(target_sentiments):
        print(f"- {sent}: {sent_count.get(sent, 0)}")

    if missing_sentiments:
        print("\nCảnh báo: Các sentiment chưa được bao phủ:")
        for sent in sorted(missing_sentiments):
            print(f"- {sent}")

    print(f"\nReview có annotations rỗng: {empty_count}")

    print("\nPhân bố category thực tế:")
    for cat, count in sorted(cat_count.items(), key=lambda x: x[1], reverse=True):
        print(f"- {cat}: {count}")

    print("\nPhân bố sentiment thực tế:")
    for sent, count in sorted(sent_count.items(), key=lambda x: x[1], reverse=True):
        print(f"- {sent}: {count}")


if __name__ == "__main__":
    main()