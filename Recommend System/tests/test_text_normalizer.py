from absa_recommender.review_normalizer import normalize_review
from absa_recommender.text_normalizer import normalize_review_text


def test_normalize_review_text_lowercases_whitespace_punctuation_and_emoji() -> None:
    text = "  Đồ ăn   Ngon 😋👍!!!\nPhục vụ\tCHẬM...  "

    assert normalize_review_text(text) == "đồ ăn ngon! phục vụ chậm."


def test_normalize_review_text_removes_unicode_replacement_character() -> None:
    text = "Giá mỗi người 300-400 n đồ ăn ổn \ufffd"

    assert normalize_review_text(text) == "giá mỗi người 300-400 n đồ ăn ổn"


def test_normalize_review_text_removes_object_replacement_and_format_chars() -> None:
    text = "Giá mỗi người 300-400 n đồ ăn ổn \ufffc\u200b\u202e"

    assert normalize_review_text(text) == "giá mỗi người 300-400 n đồ ăn ổn"


def test_normalize_review_text_removes_private_use_google_maps_icon_chars() -> None:
    text = "Giá mỗi người 300-400 n ₫ đồ ăn: 4 dịch vụ: 5 bầu không khí: 5 0:06 0:18 \ue8dc"

    assert normalize_review_text(text) == "giá mỗi người 300-400 n ₫ đồ ăn: 4 dịch vụ: 5 bầu không khí: 5 0:06 0:18"


def test_normalize_review_text_expands_vietnamese_abbreviations() -> None:
    text = "nvien pv ko tốt, chờ 30p"

    assert normalize_review_text(text) == "nhân viên phục vụ không tốt, chờ 30 phút"


def test_normalize_review_text_applies_phrase_replacements() -> None:
    text = "Món sold out, phục vụ lâu, đợi lâu"

    assert normalize_review_text(text) == "món hết món, phục vụ chậm, chờ lâu"


def test_normalize_review_text_does_not_replace_k_inside_price() -> None:
    text = "Giá 100k nhưng k ngon"

    assert normalize_review_text(text) == "giá 100k nhưng không ngon"


def test_normalize_review_text_keeps_negation_when_stopwords_disabled() -> None:
    text = "Không được ngon lắm"

    assert normalize_review_text(text) == "không ngon"


def test_normalize_review_uses_normalized_text_for_absa_input_and_hash() -> None:
    review = normalize_review(
        {
            "review_id": "rv_1",
            "restaurant_id": "res_1",
            "text": " Ban hoi dơ, nv pv ko tốt 😡!!! ",
            "rating": 2,
            "review_time": "2026-06-10T00:00:00",
        }
    )

    assert review["review_text"] == "bẩn hoi bẩn, nhân viên phục vụ không tốt!"
    assert review["review_text_hash"]
    assert review["review_month"] == "2026-06"