import duckdb

RESTAURANT_ID = "pizza_4p_s_tòa_nhà_hoàng_thành"
DB_PATH = "data/local.duckdb"

terms = [
    "bình luận synthetic",
    "ghi nhận demo",
    "synthetic tháng",
    "demo cho tháng",
    "trải nghiệm tham khảo tháng",
]

con = duckdb.connect(DB_PATH)

marker_condition = " OR ".join(["review_text ILIKE ?" for _ in terms])
marker_params = [f"%{term}%" for term in terms]
marker_hits = con.execute(
    f"SELECT COUNT(*) FROM reviews WHERE {marker_condition}",
    marker_params,
).fetchone()[0]

synthetic_counts = con.execute(
    """
    SELECT review_month, COUNT(*)
    FROM reviews
    WHERE restaurant_id = ?
      AND review_id LIKE 'synthetic_demo_%'
    GROUP BY review_month
    ORDER BY review_month
    """,
    [RESTAURANT_ID],
).fetchall()

monthly_counts = con.execute(
    """
    SELECT review_month, COUNT(*)
    FROM reviews
    WHERE restaurant_id = ?
      AND review_month IN ('2026-04', '2026-05')
    GROUP BY review_month
    ORDER BY review_month
    """,
    [RESTAURANT_ID],
).fetchall()

sample_reviews = con.execute(
    """
    SELECT review_month, review_id, review_text
    FROM reviews
    WHERE restaurant_id = ?
      AND review_id LIKE 'synthetic_demo_%'
    ORDER BY review_month, review_id
    LIMIT 8
    """,
    [RESTAURANT_ID],
).fetchall()

con.close()

print("marker_hits=", marker_hits)
print("synthetic_counts=", synthetic_counts)
print("monthly_counts=", monthly_counts)
print("sample_reviews=")
for row in sample_reviews:
    print(row)