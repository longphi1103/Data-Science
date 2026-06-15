from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from absa_recommender.storage import default_db_path, init_db


TABLES_IN_DELETE_ORDER = [
    "priority_items",
    "priority_runs",
    "peer_aspect_monthly_stats",
    "aspect_monthly_stats",
    "absa_annotations",
    "reviews",
    "crawl_runs",
    "restaurants",
]


def _table_exists(connection: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def _count_rows(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name in TABLES_IN_DELETE_ORDER:
        if _table_exists(connection, table_name):
            counts[table_name] = int(
                connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            )
        else:
            counts[table_name] = 0
    return counts


def clear_records(db_path: str | Path) -> dict[str, int]:
    init_db(db_path)
    with duckdb.connect(str(db_path)) as connection:
        before_counts = _count_rows(connection)
        for table_name in TABLES_IN_DELETE_ORDER:
            if _table_exists(connection, table_name):
                connection.execute(f"DELETE FROM {table_name}")
        connection.execute("CHECKPOINT")
    return before_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete all current records from the ABSA DuckDB database while keeping the schema."
    )
    parser.add_argument(
        "--db-path",
        default=str(default_db_path()),
        help="DuckDB file path. Defaults to ABSA_DB_PATH or data/local.duckdb.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation flag for deleting records.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not args.yes:
        raise SystemExit(
            "Refusing to delete records without confirmation. Re-run with --yes, for example:\n"
            f"  uv run python scripts/clear_duckdb_records.py --db-path {db_path} --yes"
        )

    before_counts = clear_records(db_path)
    print(f"Cleared DuckDB records from: {db_path}")
    for table_name, count in before_counts.items():
        print(f"- {table_name}: deleted {count} rows")


if __name__ == "__main__":
    main()