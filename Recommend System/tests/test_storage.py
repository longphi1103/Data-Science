from pathlib import Path

import duckdb

from absa_recommender.normalize_absa import load_absa_jsonl
from absa_recommender.recommender import generate_priority_ranking
from absa_recommender.storage import (
    get_priority_run,
    init_db,
    list_priority_runs,
    save_priority_run,
)


SAMPLE_PATH = Path("data/samples/absa_outputs.jsonl")


def test_init_db_creates_priority_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "local.duckdb"

    init_db(db_path)

    with duckdb.connect(str(db_path)) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}

    assert "restaurants" in tables
    assert "reviews" in tables
    assert "absa_annotations" in tables
    assert "aspect_monthly_stats" in tables
    assert "peer_aspect_monthly_stats" in tables
    assert "priority_runs" in tables
    assert "priority_items" in tables
    assert "subproblem_predictions" not in tables
    assert "feedback" not in tables


def test_save_priority_run_and_items(tmp_path: Path) -> None:
    db_path = tmp_path / "local.duckdb"
    response = generate_priority_ranking(load_absa_jsonl(SAMPLE_PATH), top_n=3)

    run_id = save_priority_run(db_path, response, scoring_config_hash="hash")
    run = get_priority_run(db_path, run_id)

    with duckdb.connect(str(db_path)) as connection:
        item_count = connection.execute(
            "SELECT COUNT(*) FROM priority_items WHERE priority_run_id = ?",
            [run_id],
        ).fetchone()[0]

    assert run_id.startswith("priority_")
    assert run is not None
    assert run["output"]["items"]
    assert item_count == len(response.items)
    assert list_priority_runs(db_path)
