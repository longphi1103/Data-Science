import json
from pathlib import Path

from typer.testing import CliRunner

from absa_recommender.cli import app


runner = CliRunner()
SAMPLE_PATH = Path("data/samples/absa_outputs.jsonl")
STREAMLIT_SAMPLE_PATH = Path("data/samples/streamlit_priority_200.jsonl")


def test_validate_command_exits_0() -> None:
    result = runner.invoke(app, ["validate", "--input", str(SAMPLE_PATH)])

    assert result.exit_code == 0
    assert "reviews: 3" in result.output
    assert "annotations: 7" in result.output


def test_score_priority_creates_output_json(tmp_path: Path) -> None:
    output = tmp_path / "priority.json"

    result = runner.invoke(
        app,
        [
            "score-priority",
            "--input",
            str(SAMPLE_PATH),
            "--restaurant-id",
            "res_demo",
            "--top-n",
            "5",
            "--output",
            str(output),
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert output.exists()
    assert payload["items"]
    assert "saved:" in result.output


def test_adapter_orchestration_commands_exit_0(tmp_path: Path) -> None:
    assert runner.invoke(app, ["discover-peers", "res_demo"]).exit_code == 0
    crawl_result = runner.invoke(
        app,
        [
            "crawl-month",
            "res_demo",
            "2026-06",
            "--input",
            str(STREAMLIT_SAMPLE_PATH),
            "--db-path",
            str(tmp_path / "cli.duckdb"),
        ],
    )
    infer_result = runner.invoke(
        app,
        [
            "infer-absa",
            "2026-06",
            "--input",
            str(STREAMLIT_SAMPLE_PATH),
            "--adapter",
            "placeholder",
        ],
    )
    backfill_result = runner.invoke(
        app,
        [
            "backfill",
            "res_demo",
            "2026-01",
            "2026-06",
            "--input",
            str(STREAMLIT_SAMPLE_PATH),
            "--db-path",
            str(tmp_path / "backfill.duckdb"),
        ],
    )
    assert crawl_result.exit_code == 0
    assert '"status": "success"' in crawl_result.output
    assert infer_result.exit_code == 0
    assert '"status": "completed"' in infer_result.output
    assert backfill_result.exit_code == 0


def test_run_full_command_uses_source_and_placeholder_absa(tmp_path: Path) -> None:
    db_path = tmp_path / "full.duckdb"
    output = tmp_path / "priority.json"

    result = runner.invoke(
        app,
        [
            "run-full",
            "--input",
            str(STREAMLIT_SAMPLE_PATH),
            "--restaurant-id",
            "res_demo",
            "--month",
            "2026-06",
            "--top-n",
            "5",
            "--db-path",
            str(db_path),
            "--output",
            str(output),
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert "absa_adapter: placeholder-rule-absa-v0" in result.output
    assert output.exists()


def test_run_monthly_and_db_read_commands(tmp_path: Path) -> None:
    db_path = tmp_path / "cli.duckdb"
    output = tmp_path / "priority.json"

    run_result = runner.invoke(
        app,
        [
            "run-monthly",
            "--input",
            str(STREAMLIT_SAMPLE_PATH),
            "--restaurant-id",
            "res_demo",
            "--month",
            "2026-06",
            "--top-n",
            "10",
            "--db-path",
            str(db_path),
            "--output",
            str(output),
            "--force",
        ],
    )
    dashboard_result = runner.invoke(
        app,
        [
            "show-dashboard",
            "--restaurant-id",
            "res_demo",
            "--month",
            "2026-06",
            "--db-path",
            str(db_path),
        ],
    )
    runs_result = runner.invoke(
        app,
        ["list-runs", "--restaurant-id", "res_demo", "--db-path", str(db_path)],
    )
    history_result = runner.invoke(
        app,
        [
            "aspect-history",
            "--restaurant-id",
            "res_demo",
            "--aspect",
            "Service",
            "--db-path",
            str(db_path),
        ],
    )

    assert run_result.exit_code == 0
    assert "status: completed" in run_result.output
    assert output.exists()
    assert dashboard_result.exit_code == 0
    assert '"priority"' in dashboard_result.output
    assert runs_result.exit_code == 0
    assert "priority_" in runs_result.output
    assert history_result.exit_code == 0


def test_show_labels_includes_location_and_menu() -> None:
    result = runner.invoke(app, ["show-labels"])

    assert result.exit_code == 0
    assert "Location" in result.output
    assert "Menu" in result.output
