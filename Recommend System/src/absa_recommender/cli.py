import json
import sys
from pathlib import Path
from typing import Any

import typer

from absa_recommender.absa_inference import build_absa_adapter, infer_absa_with_adapter
from absa_recommender.config import load_label_schema
from absa_recommender.crawler.monthly import crawl_reviews_for_month, persist_crawl_result
from absa_recommender.monthly_pipeline import run_monthly_from_absa_jsonl, run_monthly_from_source
from absa_recommender.normalize_absa import flatten_reviews, load_absa_jsonl
from absa_recommender.recommender import generate_priority_ranking
from absa_recommender.storage import (
    aspect_history as storage_aspect_history,
    dashboard_payload,
    default_db_path,
    list_restaurants,
    list_priority_runs,
)
from absa_recommender.sources.local_jsonl_adapter import LocalJsonlAdapter

app = typer.Typer(help="Local ABSA aspect priority engine.")


@app.command()
def validate(
    input_path: Path = typer.Option(
        Path("data/samples/absa_outputs.jsonl"),
        "--input",
        help="Path to ABSA JSONL input.",
    ),
) -> None:
    """Parse JSONL and validate labels against label_schema.yaml."""
    _configure_stdout()
    schema = load_label_schema("configs/label_schema.yaml")
    reviews = load_absa_jsonl(input_path)
    extractions = flatten_reviews(reviews, schema, strict=True)
    typer.echo(f"reviews: {len(reviews)}")
    typer.echo(f"annotations: {len(extractions)}")


@app.command("score-priority")
def score_priority(
    input_path: Path = typer.Option(
        Path("data/samples/absa_outputs.jsonl"),
        "--input",
        help="Path to ABSA JSONL input.",
    ),
    restaurant_id: str | None = typer.Option(
        None,
        "--restaurant-id",
        help="Override restaurant_id for this batch.",
    ),
    month: str | None = typer.Option(
        None,
        "--month",
        help="Review month to score, for example 2026-06.",
    ),
    top_n: int = typer.Option(5, "--top-n", min=1, help="Number of aspects."),
    output: Path = typer.Option(
        Path("out/priority.json"),
        "--output",
        help="Path to write priority JSON.",
    ),
) -> None:
    """Rank Top-N aspects to improve from ABSA JSONL."""
    _configure_stdout()
    reviews = _load_reviews_with_restaurant_override(input_path, restaurant_id)
    response = generate_priority_ranking(reviews, top_n=top_n, review_month=month)
    _write_json(output, response.model_dump(mode="json"))
    typer.echo(f"saved: {output}")
    for item in response.items:
        typer.echo(
            f"{item.rank}. {item.aspect} score={item.priority_score:.2f} "
            f"confidence={item.priority_confidence:.2f}"
        )


@app.command("run-monthly")
def run_monthly(
    input_path: Path = typer.Option(
        Path("data/samples/absa_outputs.jsonl"),
        "--input",
        help="Path to monthly ABSA JSONL input.",
    ),
    restaurant_id: str | None = typer.Option(None, "--restaurant-id"),
    month: str | None = typer.Option(None, "--month"),
    top_n: int = typer.Option(5, "--top-n", min=1),
    db_path: Path = typer.Option(default_db_path(), "--db-path", help="DuckDB path."),
    force: bool = typer.Option(False, "--force", help="Create a new run even if one exists."),
    output: Path = typer.Option(Path("out/priority.json"), "--output"),
) -> None:
    """Persist a complete local monthly run from existing ABSA annotations."""
    _configure_stdout()
    if restaurant_id is None:
        raise typer.BadParameter("--restaurant-id is required for persisted monthly runs")
    if month is None:
        raise typer.BadParameter("--month is required for persisted monthly runs")
    result = run_monthly_from_absa_jsonl(
        input_path,
        restaurant_id=restaurant_id,
        review_month=month,
        top_n=top_n,
        db_path=db_path,
        force=force,
    )
    _write_json(output, result["output"])
    typer.echo(f"status: {result['status']}")
    typer.echo(f"db: {result['db_path']}")
    typer.echo(f"priority_run_id: {result['priority_run_id']}")
    typer.echo(f"saved: {output}")


@app.command("run-full")
def run_full(
    input_path: Path = typer.Option(
        Path("data/samples/streamlit_priority_200.jsonl"),
        "--input",
        help="Raw review JSONL source used by the local crawler adapter.",
    ),
    restaurant_id: str = typer.Option(..., "--restaurant-id"),
    month: str = typer.Option(..., "--month"),
    top_n: int = typer.Option(5, "--top-n", min=1),
    db_path: Path = typer.Option(default_db_path(), "--db-path", help="DuckDB path."),
    absa_adapter: str = typer.Option(
        "placeholder",
        "--absa-adapter",
        help="ABSA adapter name: placeholder, preannotated, or trained/vit5.",
    ),
    source_adapter: str = typer.Option(
        "local-jsonl",
        "--source-adapter",
        help="Source adapter: local-jsonl or google-maps.",
    ),
    gmaps_live: bool = typer.Option(False, "--gmaps-live", help="Run Google Maps crawler in live Playwright mode."),
    gmaps_discover_from_area: bool = typer.Option(
        False,
        "--gmaps-discover-from-area",
        help="Discover Google Maps places from area before crawling.",
    ),
    gmaps_area_name: str | None = typer.Option(None, "--gmaps-area-name"),
    gmaps_bbox: str | None = typer.Option(None, "--gmaps-bbox"),
    gmaps_target_url: str | None = typer.Option(None, "--gmaps-target-url"),
    force: bool = typer.Option(False, "--force", help="Create a new run even if one exists."),
    output: Path = typer.Option(Path("out/priority.json"), "--output"),
) -> None:
    """Run local source crawl, ABSA inference, persistence, scoring and dashboard snapshot."""
    _configure_stdout()
    result = run_monthly_from_source(
        input_path,
        restaurant_id=restaurant_id,
        review_month=month,
        top_n=top_n,
        db_path=db_path,
        force=force,
        absa_adapter=absa_adapter,
        source_adapter=source_adapter,
        gmaps_live=gmaps_live,
        gmaps_discover_from_area=gmaps_discover_from_area,
        gmaps_area_name=gmaps_area_name,
        gmaps_bbox=gmaps_bbox,
        gmaps_target_url=gmaps_target_url,
    )
    _write_json(output, result["output"])
    typer.echo(f"status: {result['status']}")
    typer.echo(f"db: {result['db_path']}")
    typer.echo(f"absa_adapter: {result['absa_adapter']}")
    typer.echo(f"source_adapter: {result.get('source_adapter', 'local_jsonl')}")
    typer.echo(f"priority_run_id: {result['priority_run_id']}")
    typer.echo(f"saved: {output}")


@app.command("compute-stats")
def compute_stats(
    input_path: Path = typer.Option(Path("data/samples/absa_outputs.jsonl"), "--input"),
    restaurant_id: str | None = typer.Option(None, "--restaurant-id"),
    month: str | None = typer.Option(None, "--month"),
    output: Path = typer.Option(Path("out/aspect_monthly_stats.json"), "--output"),
) -> None:
    """Compute monthly aspect stats by running priority scoring and exporting item stats."""
    _configure_stdout()
    reviews = _load_reviews_with_restaurant_override(input_path, restaurant_id)
    response = generate_priority_ranking(reviews, top_n=100, review_month=month)
    rows = [
        {
            "restaurant_id": response.restaurant_id,
            "review_month": response.review_month,
            "aspect": item.aspect,
            "mention_count": item.mention_count,
            "negative_count": item.negative_count,
            "negative_rate_smoothed": item.negative_rate_smoothed,
            "severity": item.severity,
            "mention_share": item.mention_share,
            "rating_gap": item.rating_gap,
        }
        for item in response.items
    ]
    _write_json(output, rows)
    typer.echo(f"saved: {output}")


@app.command("show-dashboard")
def show_dashboard_payload(
    restaurant_id: str = typer.Option(..., "--restaurant-id"),
    month: str = typer.Option(..., "--month"),
    db_path: Path = typer.Option(default_db_path(), "--db-path"),
) -> None:
    """Print persisted dashboard payload from DuckDB."""
    _configure_stdout()
    payload = dashboard_payload(db_path, restaurant_id, month)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.command("list-runs")
def list_runs(
    restaurant_id: str | None = typer.Option(None, "--restaurant-id"),
    db_path: Path = typer.Option(default_db_path(), "--db-path"),
) -> None:
    """List persisted priority runs in DuckDB."""
    _configure_stdout()
    typer.echo(
        json.dumps(
            list_priority_runs(db_path, restaurant_id=restaurant_id),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


@app.command("aspect-history")
def aspect_history(
    restaurant_id: str = typer.Option(..., "--restaurant-id"),
    aspect: str = typer.Option(..., "--aspect"),
    db_path: Path = typer.Option(default_db_path(), "--db-path"),
) -> None:
    """Print persisted priority history for one aspect."""
    _configure_stdout()
    typer.echo(
        json.dumps(
            storage_aspect_history(db_path, restaurant_id, aspect),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


@app.command("discover-peers")
def discover_peers(
    restaurant_id: str,
    radius_meters: int = 1500,
    db_path: Path = typer.Option(default_db_path(), "--db-path"),
) -> None:
    """List persisted peers for a restaurant from DuckDB."""
    _configure_stdout()
    restaurants = list_restaurants(db_path)
    peers = [
        row
        for row in restaurants
        if row["restaurant_id"] != restaurant_id and bool(row.get("is_peer", False))
    ]
    typer.echo(
        json.dumps(
            {
                "status": "loaded_from_duckdb",
                "restaurant_id": restaurant_id,
                "radius_meters": radius_meters,
                "peers": peers,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


@app.command("crawl-month")
def crawl_month(
    restaurant_id: str,
    month: str,
    db_path: Path = typer.Option(default_db_path(), "--db-path"),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Optional raw review JSONL source. Without this, only a manual crawl_runs record is created.",
    ),
) -> None:
    """Crawl monthly reviews through a local source adapter or create a manual crawl run record."""
    _configure_stdout()
    if input_path is None:
        from absa_recommender.storage import save_crawl_run

        crawl_run_id = save_crawl_run(
            db_path,
            source="manual",
            target_month=month,
            area_id="local",
            status="created",
            num_restaurants=1,
            num_reviews_fetched=0,
            num_reviews_inserted=0,
            num_duplicates=0,
        )
        status = "created"
        reviews_inserted = 0
        duplicates = 0
    else:
        result = crawl_reviews_for_month(
            LocalJsonlAdapter(input_path),
            restaurant_id=restaurant_id,
            month=month,
        )
        crawl_run_id = persist_crawl_result(db_path, result)
        status = "success"
        reviews_inserted = len(result.reviews)
        duplicates = len(result.duplicate_reviews)
    typer.echo(
        json.dumps(
            {
                "status": status,
                "crawl_run_id": crawl_run_id,
                "restaurant_id": restaurant_id,
                "review_month": month,
                "reviews_inserted": reviews_inserted,
                "duplicates": duplicates,
                "db_path": str(db_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("infer-absa")
def infer_absa(
    month: str,
    input_path: Path = typer.Option(
        Path("data/samples/streamlit_priority_200.jsonl"),
        "--input",
        help="Review JSONL input used by the selected ABSA adapter.",
    ),
    adapter_name: str = typer.Option(
        "preannotated",
        "--adapter",
        help="ABSA adapter name: preannotated or placeholder.",
    ),
    output: Path | None = typer.Option(None, "--output", help="Optional JSON output path."),
) -> None:
    """Run ABSA inference through the selected local adapter."""
    _configure_stdout()
    raw_records = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            row_month = row.get("review_month") or str(row.get("review_time", ""))[:7]
            if row_month == month:
                raw_records.append(row)
    adapter = build_absa_adapter(adapter_name)
    reviews = infer_absa_with_adapter(raw_records, adapter=adapter)
    payload = {
        "status": "completed",
        "adapter": adapter.model_version,
        "review_month": month,
        "review_count": len(reviews),
        "annotation_count": sum(len(review.annotations) for review in reviews),
    }
    if output is not None:
        _write_json(output, payload)
    typer.echo(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )


@app.command()
def backfill(
    restaurant_id: str,
    start_month: str,
    end_month: str,
    input_path: Path = typer.Option(
        Path("data/samples/streamlit_priority_200.jsonl"),
        "--input",
        help="ABSA JSONL input used for each month in the range.",
    ),
    db_path: Path = typer.Option(default_db_path(), "--db-path"),
    top_n: int = typer.Option(10, "--top-n", min=1),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run persisted monthly scoring across an inclusive month range."""
    _configure_stdout()
    results = []
    for month in _month_range(start_month, end_month):
        try:
            results.append(
                run_monthly_from_absa_jsonl(
                    input_path,
                    restaurant_id=restaurant_id,
                    review_month=month,
                    top_n=top_n,
                    db_path=db_path,
                    force=force,
                )
            )
        except ValueError as error:
            results.append({"status": "skipped", "review_month": month, "reason": str(error)})
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2, default=str))


@app.command("show-labels")
def show_labels() -> None:
    """Print labels loaded from label_schema.yaml."""
    _configure_stdout()
    schema = load_label_schema("configs/label_schema.yaml")
    typer.echo("Aspects:")
    for label in schema.get("aspects", []):
        typer.echo(f"- {label}")
    typer.echo("Sentiments:")
    for label in schema.get("sentiments", []):
        typer.echo(f"- {label}")


def _load_reviews_with_restaurant_override(input_path: Path, restaurant_id: str | None):
    reviews = load_absa_jsonl(input_path)
    if restaurant_id is None:
        return reviews
    return [review.model_copy(update={"restaurant_id": restaurant_id}) for review in reviews]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start = [int(part) for part in start_month.split("-")]
    end_year, end = [int(part) for part in end_month.split("-")]
    months = []
    year = start_year
    month = start
    while (year, month) <= (end_year, end):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months
