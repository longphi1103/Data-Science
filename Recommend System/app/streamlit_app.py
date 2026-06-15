import hashlib
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# The trained ABSA adapter imports Hugging Face transformers. Streamlit's default
# source watcher introspects every loaded module and can accidentally trigger
# transformers' lazy vision imports, which require optional torchvision even
# though this text-only ABSA pipeline does not use it.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")


def _streamlit_cli_args(argv: list[str]) -> list[str]:
    """Translate convenient direct-run args into Streamlit CLI args.

    This lets commands such as:
        uv run app/streamlit_app.py --port 8051
    launch the app through Streamlit instead of running in bare Python mode.
    """
    streamlit_args: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--port":
            if index + 1 >= len(argv):
                raise SystemExit("--port requires a value")
            streamlit_args.extend(["--server.port", argv[index + 1]])
            index += 2
            continue
        if arg.startswith("--port="):
            streamlit_args.extend(["--server.port", arg.split("=", 1)[1]])
            index += 1
            continue
        streamlit_args.append(arg)
        index += 1
    return streamlit_args


def _bootstrap_streamlit_if_direct_run() -> None:
    if __name__ != "__main__":
        return
    if os.environ.get("ABSA_STREAMLIT_BOOTSTRAPPED") == "1":
        return

    env = os.environ.copy()
    env["ABSA_STREAMLIT_BOOTSTRAPPED"] = "1"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        *_streamlit_cli_args(sys.argv[1:]),
        str(Path(__file__)),
    ]
    try:
        raise SystemExit(subprocess.call(command, env=env))
    except KeyboardInterrupt:
        raise SystemExit(0) from None


_bootstrap_streamlit_if_direct_run()

import altair as alt  # noqa: E402
import duckdb  # noqa: E402
import streamlit as st  # noqa: E402

from absa_recommender.aggregation import aggregate_aspect_stats  # noqa: E402
from absa_recommender.config import load_label_schema  # noqa: E402
from absa_recommender.config import load_yaml  # noqa: E402
from absa_recommender.explore_jobs import (  # noqa: E402
    ExploreJobRequest,
    coerce_progress_percent,
    read_job_log,
    read_latest_job_status,
    read_job_status,
    read_running_job_status,
    start_explore_job,
)
from absa_recommender.normalize_absa import flatten_reviews, load_absa_jsonl  # noqa: E402
from absa_recommender.text_normalizer import normalize_review_text  # noqa: E402
from absa_recommender.recommender import generate_priority_ranking  # noqa: E402
from absa_recommender.schemas import ABSAReview, PriorityResponse  # noqa: E402
from absa_recommender.scoring import (  # noqa: E402
    compute_global_negative_rate_by_aspect,
    smoothed_negative_rate,
)
from absa_recommender.storage import (  # noqa: E402
    dashboard_payload,
    default_db_path,
    find_priority_run,
    list_restaurants,
    list_review_months,
)


SAMPLE_PATH = Path("data/samples/absa_outputs.jsonl")
CRAWLER_CONFIG_PATH = Path("configs/crawler.yaml")
ABSA_MODEL_CONFIG_PATH = Path("configs/absa_model.yaml")
EXPLORE_JOBS_DIR = Path("data/explore_jobs")


st.set_page_config(page_title="ABSA Aspect Priority Engine", layout="wide")
st.title("ABSA Aspect Priority Engine")


def main() -> None:
    label_schema = load_label_schema("configs/label_schema.yaml")
    st.sidebar.header("Input")
    use_duckdb = st.sidebar.checkbox("Load from DuckDB", value=False)
    db_path = st.sidebar.text_input("DuckDB path", value=str(default_db_path()))
    if use_duckdb:
        selected_restaurant, selected_month = _show_duckdb_dashboard_selectors(db_path)
        _show_explore_controls(db_path)
        _show_duckdb_dashboard(db_path, selected_restaurant, selected_month)
        return

    uploaded_file = st.sidebar.file_uploader("ABSA JSONL", type=["jsonl", "json"])
    default_restaurant_id = st.sidebar.text_input("Default restaurant_id", value="unknown")
    review_month = st.sidebar.text_input("Review month", value="")
    top_n = st.sidebar.slider("Top N", min_value=1, max_value=20, value=5)
    generate = st.sidebar.button("Score priority", type="primary")

    reviews = _load_reviews(uploaded_file)
    _show_labels(label_schema)

    if not generate:
        st.info("Upload a monthly ABSA JSONL file or use the bundled sample, then score priority.")
        st.caption(f"Loaded reviews: {len(reviews)}")
        return

    month = review_month.strip() or None
    extractions = flatten_reviews(
        reviews,
        label_schema,
        default_restaurant_id=default_restaurant_id,
        strict=True,
    )
    target_reviews = _target_reviews(reviews, default_restaurant_id)
    peer_benchmarks = _peer_benchmarks(
        extractions,
        label_schema,
        default_restaurant_id,
        month,
    )
    response = generate_priority_ranking(
        target_reviews,
        top_n=top_n,
        default_restaurant_id=default_restaurant_id,
        review_month=month,
        peer_benchmarks=peer_benchmarks,
    )

    tabs = st.tabs(
        [
            "Monthly Overview",
            "Top-N Aspects",
            "Aspect Detail",
            "History",
            "Data Quality",
        ]
    )
    with tabs[0]:
        _show_overview(response, extractions)
    with tabs[1]:
        _show_priority_items(response)
    with tabs[2]:
        _show_aspect_detail(response)
    with tabs[3]:
        _show_history(response)
    with tabs[4]:
        _show_data_quality(response, extractions)


def _load_reviews(uploaded_file) -> list[ABSAReview]:
    if uploaded_file is None:
        return load_absa_jsonl(SAMPLE_PATH)

    text = uploaded_file.getvalue().decode("utf-8")
    reviews = []
    for line in text.splitlines():
        if line.strip():
            reviews.append(ABSAReview.model_validate(json.loads(line)))
    return reviews


def _make_explore_progress_callback(
    progress_bar: Any,
    status_box: Any,
    current_step: Any,
    log_box: Any,
):
    log_lines: list[str] = []

    def callback(message: str, percent: int | None = None) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        log_lines.append(line)
        if len(log_lines) > 120:
            del log_lines[:-120]

        print(f"[streamlit-explore] {line}", flush=True)

        if percent is not None:
            progress_value = max(0, min(100, int(percent)))
            progress_bar.progress(progress_value, text=message)
            status_box.update(label=message)
        else:
            status_box.write(message)

        current_step.info(message)
        rendered_lines = "\n".join(log_lines)
        log_box.code(rendered_lines, language="text")

    return callback


def _show_duckdb_dashboard_selectors(db_path: str) -> tuple[str | None, str | None]:
    restaurants = list_restaurants(db_path)
    if not restaurants:
        return None, None

    st.sidebar.divider()
    restaurant_ids = [row["restaurant_id"] for row in restaurants]
    default_restaurant = st.session_state.get("explore_selected_restaurant")
    restaurant_index = restaurant_ids.index(default_restaurant) if default_restaurant in restaurant_ids else 0
    selected_restaurant = st.sidebar.selectbox("Restaurant", restaurant_ids, index=restaurant_index)

    months = list_review_months(db_path, selected_restaurant)
    if not months:
        return selected_restaurant, None

    default_month = st.session_state.get("explore_selected_month")
    month_index = months.index(default_month) if default_month in months else 0
    selected_month = st.sidebar.selectbox("Month", months, index=month_index)
    return selected_restaurant, selected_month


def _show_explore_controls(db_path: str) -> None:
    crawler_config = load_yaml(CRAWLER_CONFIG_PATH)
    absa_config = load_yaml(ABSA_MODEL_CONFIG_PATH)
    streamlit_config = crawler_config.get("streamlit", {})
    explore_config = streamlit_config.get("explore", {})
    gmaps_config = crawler_config.get("google_maps", {})
    peer_config = gmaps_config.get("peer_discovery", {})
    absa_inference_config = absa_config.get("inference", {})

    crawl_output_path = Path(explore_config.get("output_path", "data/gmaps_streamlit_raw.jsonl"))
    default_month = str(explore_config.get("default_month", datetime.now().strftime("%Y-%m")))
    default_area_name = str(peer_config.get("default_area_name", ""))
    explore_top_n = len(load_label_schema("configs/label_schema.yaml").get("aspects", []))
    source_adapter = str(explore_config.get("source_adapter", crawler_config.get("source_adapter", "google-maps")))
    default_live = bool(gmaps_config.get("live", False))
    default_discover_from_area = bool(peer_config.get("discover_from_area", True))
    adapter_options = [str(item) for item in absa_inference_config.get("adapter_options", ["placeholder", "trained"])]
    default_absa_adapter = str(absa_inference_config.get("adapter", adapter_options[0] if adapter_options else "placeholder"))
    if default_absa_adapter not in adapter_options:
        adapter_options.insert(0, default_absa_adapter)

    current_job_status = _current_explore_job_status()
    job_running = current_job_status.get("status") == "running"

    st.sidebar.divider()
    st.sidebar.header("Google Maps Explore")
    target_url = st.sidebar.text_input(
        "Restaurant / eatery Google Maps URL",
        placeholder=str(gmaps_config.get("target_url_placeholder", "https://www.google.com/maps/place/...")),
    )
    review_month = st.sidebar.text_input("Crawl month", value=default_month, help="Format: YYYY-MM")
    area_name = st.sidebar.text_input(
        "Ward / administrative area for peer discovery",
        value=default_area_name,
        help=(
            "Use a ward/district/city name resolvable by OpenStreetMap Nominatim, "
            "not the full restaurant street address."
        ),
    )
    restaurant_id = st.sidebar.text_input(
        "Restaurant ID",
        value=_restaurant_id_from_url(target_url) if target_url.strip() else "",
        help="Auto-derived from URL; override if you want a stable custom ID.",
    )
    absa_adapter = st.sidebar.selectbox(
        "ABSA adapter",
        adapter_options,
        index=adapter_options.index(default_absa_adapter),
        help=str(
            absa_inference_config.get(
                "adapter_help",
                "Use placeholder in Docker unless the trained model tokenizer is available.",
            )
        ),
    )
    live = st.sidebar.checkbox("Live Google Maps crawl", value=default_live)
    discover_from_area = st.sidebar.checkbox("Discover peer restaurants in ward", value=default_discover_from_area)
    run_explore = st.sidebar.button("Start Explore Job", type="primary", disabled=job_running)

    if not run_explore:
        return

    target_url = target_url.strip()
    restaurant_id = restaurant_id.strip()
    month = review_month.strip()
    if not target_url:
        st.error("Please enter a Google Maps URL before running Explore.")
        return
    if not restaurant_id:
        st.error("Please provide a Restaurant ID before running Explore.")
        return
    if not _is_valid_month(month):
        st.error("Crawl month must use YYYY-MM format.")
        return

    effective_area_name = _normalize_area_name_for_discovery(area_name) if discover_from_area else None
    if discover_from_area and not effective_area_name:
        st.error("Please enter a ward/administrative area for peer discovery.")
        return
    if effective_area_name and effective_area_name != area_name.strip():
        st.info(
            "Using administrative area for peer discovery: "
            f"{effective_area_name}. Full street addresses cannot be resolved as area polygons."
        )

    existing = find_priority_run(db_path, restaurant_id, month)
    if existing is not None:
        st.success(
            "This restaurant/month has already been processed. Skipped pipeline execution."
        )
        st.json(
            {
                "restaurant_id": restaurant_id,
                "review_month": month,
                "priority_run_id": existing.get("priority_run_id"),
                "status": existing.get("status"),
            }
        )
        st.session_state["explore_selected_restaurant"] = restaurant_id
        st.session_state["explore_selected_month"] = month
        return

    request = ExploreJobRequest(
        input_path=str(crawl_output_path),
        restaurant_id=restaurant_id,
        review_month=month,
        top_n=explore_top_n,
        db_path=db_path,
        force=bool(explore_config.get("force", False)),
        area_id=_area_id_from_name(effective_area_name or str(peer_config.get("default_area_id", "gmaps_area"))),
        absa_adapter=absa_adapter,
        source_adapter=source_adapter,
        gmaps_live=live,
        gmaps_discover_from_area=discover_from_area,
        gmaps_area_name=effective_area_name,
        gmaps_bbox=gmaps_config.get("bbox"),
        gmaps_target_url=target_url,
    )
    status = start_explore_job(request, job_dir=EXPLORE_JOBS_DIR)
    st.session_state["explore_job_status_path"] = status.get("status_path")
    if status.get("reused_existing"):
        st.warning("Another Explore job is already running. The monitor will track that job instead of starting a new one.")
    else:
        st.success("Explore job started in the background. Existing dashboard data remains available below.")
    st.json(
        {
            "job_id": status.get("job_id"),
            "status": status.get("status"),
            "restaurant_id": restaurant_id,
            "review_month": month,
            "pid": status.get("pid"),
            "log_path": status.get("log_path"),
        }
    )
    st.session_state["explore_selected_restaurant"] = restaurant_id
    st.session_state["explore_selected_month"] = month


@st.fragment(run_every="2s")
def _show_explore_job_status() -> dict[str, Any]:
    status = _current_explore_job_status()
    status_value = status.get("status", "idle")
    if status_value == "idle":
        st.info("No Explore job is running. Start one from the sidebar to monitor progress here.")
        return status

    state = "running" if status_value == "running" else "complete" if status_value == "success" else "error"
    progress_percent = coerce_progress_percent(status.get("percent"))
    with st.status("Explore Job Status", state=state, expanded=status_value == "running"):
        cols = st.columns(4)
        cols[0].metric("Status", status_value)
        cols[1].metric("Progress", f"{progress_percent}%")
        cols[2].metric("Restaurant", status.get("restaurant_id") or "-")
        cols[3].metric("Month", status.get("review_month") or "-")
        st.caption(status.get("message") or "")
        st.progress(progress_percent, text=status.get("message") or "Explore job progress")
        if st.button("Refresh job status"):
            st.rerun(scope="fragment")
        log_text = read_job_log(status.get("log_path"), max_lines=50)
        _show_fixed_debug_log(log_text)
        if status_value == "success":
            st.session_state["explore_selected_restaurant"] = status.get("restaurant_id")
            st.session_state["explore_selected_month"] = status.get("review_month")
            result = status.get("result")
            if result:
                st.json(result)
    return status


def _show_fixed_debug_log(log_text: str) -> None:
    st.markdown("**Debug log**")
    if not log_text:
        st.caption("Waiting for crawler output...")
        log_text = ""
    st.markdown(
        """
        <style>
        .explore-debug-log {
            height: 360px;
            overflow-y: auto;
            white-space: pre-wrap;
            background: #0e1117;
            color: #f1f5f9;
            border: 1px solid #30363d;
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.85rem;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="explore-debug-log">{html.escape(log_text)}</div>',
        unsafe_allow_html=True,
    )



def _current_explore_job_status() -> dict[str, Any]:
    running = read_running_job_status(EXPLORE_JOBS_DIR)
    if running.get("status") == "running":
        if running.get("status_path"):
            st.session_state["explore_job_status_path"] = running.get("status_path")
        return running

    status_path = st.session_state.get("explore_job_status_path")
    if status_path:
        status = read_job_status(status_path)
        if status.get("status") != "idle":
            return status
    status = read_latest_job_status(EXPLORE_JOBS_DIR)
    if status.get("status") != "idle" and status.get("status_path"):
        st.session_state["explore_job_status_path"] = status.get("status_path")
    return status


def _show_duckdb_dashboard(db_path: str, selected_restaurant: str | None, selected_month: str | None) -> None:
    tab_labels = [
        "Monthly Overview",
        "Top-N Aspects",
        "Aspect Detail",
        "History",
        "Data Quality",
        "Job Monitor",
    ]
    selected_tab = st.radio(
        "Dashboard section",
        tab_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="duckdb_dashboard_tab",
    )

    restaurants = list_restaurants(db_path)
    if not restaurants:
        if selected_tab == "Job Monitor":
            _show_explore_job_status()
        else:
            st.warning("No restaurants found in DuckDB. Run a persisted monthly pipeline first.")
        return

    if selected_restaurant is None:
        if selected_tab == "Job Monitor":
            _show_explore_job_status()
        else:
            st.warning("No restaurant selected.")
        return

    months = list_review_months(db_path, selected_restaurant)
    if not months or selected_month is None:
        if selected_tab == "Job Monitor":
            _show_explore_job_status()
        else:
            st.warning("No review months found for the selected restaurant.")
        return

    payload = dashboard_payload(db_path, selected_restaurant, selected_month)

    if selected_tab == "Monthly Overview":
        _show_persisted_overview(payload)
    elif selected_tab == "Top-N Aspects":
        _show_persisted_priority(payload)
    elif selected_tab == "Aspect Detail":
        _show_persisted_aspect_detail(payload, db_path)
    elif selected_tab == "History":
        _show_persisted_history(payload)
    elif selected_tab == "Data Quality":
        _show_persisted_data_quality(payload)
    elif selected_tab == "Job Monitor":
        _show_explore_job_status()

def _show_persisted_overview(payload: dict[str, Any]) -> None:
    overview = payload.get("overview", {})
    cols = st.columns(4)
    cols[0].metric("Total reviews", overview.get("total_reviews", 0))
    cols[1].metric("Total ABSA annotations", overview.get("total_absa_annotations", 0))
    rating = overview.get("average_rating")
    cols[2].metric("Average rating", "n/a" if rating is None else f"{rating:.2f}")
    cols[3].metric(
        "Negative annotation rate",
        f"{overview.get('negative_annotation_rate', 0.0):.2%}",
    )
    stats = payload.get("aspect_stats", [])
    st.subheader("Aspect mentions")
    st.bar_chart(
        [{"label": row["aspect"], "count": row["mention_count"]} for row in stats],
        x="label",
        y="count",
    )
    sentiment_rows = [
        {"aspect": row["aspect"], "sentiment": "negative", "count": row["negative_count"]}
        for row in stats
    ] + [
        {"aspect": row["aspect"], "sentiment": "positive", "count": row["positive_count"]}
        for row in stats
    ] + [
        {"aspect": row["aspect"], "sentiment": "neutral", "count": row["neutral_count"]}
        for row in stats
    ]
    st.subheader("Sentiment distribution by aspect")
    _show_sentiment_distribution_chart(sentiment_rows)


def _show_persisted_priority(payload: dict[str, Any]) -> None:
    items = payload.get("priority", [])
    if not items:
        st.warning("No persisted priority items found.")
        return
    peer_negative_rate_by_aspect = {
        row["aspect"]: row.get("peer_negative_rate")
        for row in payload.get("peer_benchmark", [])
    }
    peer_restaurant_count_by_aspect = {
        row["aspect"]: row.get("peer_restaurant_count", 0)
        for row in payload.get("peer_benchmark", [])
    }
    rows = [
        {
            "Rank": item["rank"],
            "Aspect": item["aspect"],
            "Priority": item["priority_score"],
            "Confidence": item["priority_confidence"],
            "Negative rate": item["negative_rate_smoothed"],
            "Severity": item["severity"],
            "Trend": item["trend_score"],
            "Peer gap": item["benchmark_gap"],
            "Peer avg": peer_negative_rate_by_aspect.get(item["aspect"], 0.0),
            "Peer restaurants": peer_restaurant_count_by_aspect.get(item["aspect"], 0),
        }
        for item in items
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    _show_priority_bar_chart(rows)
    _show_trend_peer_gap_quadrant_chart(rows)
    _show_peer_benchmark_charts(rows)


def _show_persisted_aspect_detail(payload: dict[str, Any], db_path: str) -> None:
    items = payload.get("priority", [])
    if not items:
        st.warning("No aspect detail available.")
        return
    selected = st.selectbox("Aspect", [item["aspect"] for item in items])
    item = next(row for row in items if row["aspect"] == selected)
    cols = st.columns(4)
    cols[0].metric("Rank", f"#{item['rank']}")
    cols[1].metric("Priority score", f"{item['priority_score']:.2f}")
    cols[2].metric("Confidence", f"{item['priority_confidence']:.2f}")
    cols[3].metric("Severity", f"{item['severity']:.2f}")
    component_rows = [{"metric": key, "value": value} for key, value in item["component_scores"].items()]
    _show_metric_value_chart(component_rows)
    st.dataframe(component_rows, width="stretch", hide_index=True)
    _show_aspect_review_examples(
        db_path,
        str(payload.get("restaurant_id", "")),
        str(payload.get("review_month", "")),
        str(selected),
    )


def _load_aspect_review_examples(
    db_path: str,
    restaurant_id: str,
    review_month: str,
    aspect: str,
    sentiment: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    try:
        with duckdb.connect(str(db_path), read_only=True) as connection:
            rows = connection.execute(
                """
                WITH ranked_annotations AS (
                    SELECT
                        r.review_id,
                        r.review_text,
                        r.rating,
                        r.review_time,
                        a.opinion_text,
                        a.model_confidence,
                        a.severity,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.review_id
                            ORDER BY
                                COALESCE(a.severity, 0) DESC,
                                COALESCE(a.model_confidence, 0) DESC
                        ) AS row_number
                    FROM absa_annotations a
                    JOIN reviews r ON r.review_id = a.review_id
                    WHERE a.restaurant_id = ?
                      AND a.review_month = ?
                      AND a.aspect = ?
                      AND a.sentiment = ?
                      AND COALESCE(r.review_text, '') <> ''
                )
                SELECT review_text, rating, review_time, opinion_text, model_confidence, severity
                FROM ranked_annotations
                WHERE row_number = 1
                ORDER BY
                    COALESCE(severity, 0) DESC,
                    COALESCE(model_confidence, 0) DESC,
                    review_time DESC NULLS LAST
                LIMIT ?
                """,
                [restaurant_id, review_month, aspect, sentiment, limit],
            ).fetchall()
    except Exception as error:
        st.caption(f"Could not load {sentiment} review examples: {error}")
        return []

    columns = [
        "review_text",
        "rating",
        "review_time",
        "opinion_text",
        "model_confidence",
        "severity",
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _render_review_example_cards(rows: list[dict[str, Any]], empty_message: str) -> None:
    if not rows:
        st.caption(empty_message)
        return

    for index, row in enumerate(rows, start=1):
        rating = row.get("rating")
        severity = row.get("severity")
        confidence = row.get("model_confidence")
        review_time = row.get("review_time")
        metadata = [
            f"rating: {'n/a' if rating is None else rating}",
            f"severity: {'n/a' if severity is None else f'{float(severity):.2f}'}",
            f"confidence: {'n/a' if confidence is None else f'{float(confidence):.2f}'}",
        ]
        if review_time is not None:
            metadata.append(f"time: {review_time}")

        st.markdown(f"**#{index}** · " + " · ".join(metadata))
        st.write(normalize_review_text(str(row.get("review_text") or "").strip()))


def _show_aspect_review_examples(
    db_path: str,
    restaurant_id: str,
    review_month: str,
    aspect: str,
) -> None:
    st.subheader("Top review comments for selected aspect")
    limit = st.slider(
        "Number of comments per sentiment",
        min_value=1,
        max_value=10,
        value=5,
        key=f"aspect_review_examples_limit_{aspect}",
    )
    negative_rows = _load_aspect_review_examples(
        db_path,
        restaurant_id,
        review_month,
        aspect,
        "negative",
        limit=limit,
    )
    positive_rows = _load_aspect_review_examples(
        db_path,
        restaurant_id,
        review_month,
        aspect,
        "positive",
        limit=limit,
    )

    negative_column, positive_column = st.columns(2)
    with negative_column:
        st.markdown("#### Top negative comments")
        _render_review_example_cards(
            negative_rows,
            "No negative comments found for this aspect.",
        )
    with positive_column:
        st.markdown("#### Top positive comments")
        _render_review_example_cards(
            positive_rows,
            "No positive comments found for this aspect.",
        )


def _show_persisted_peer_benchmark(payload: dict[str, Any]) -> None:
    rows = payload.get("peer_benchmark", [])
    target_negative_rate_by_aspect = {
        row["aspect"]: row["negative_rate_smoothed"]
        for row in payload.get("aspect_stats", [])
    }
    chart_rows = [
        {
            **row,
            "target_negative_rate": target_negative_rate_by_aspect.get(row.get("aspect"), 0.0),
        }
        for row in rows
    ]
    _show_peer_benchmark_charts(chart_rows)
    st.dataframe(chart_rows, width="stretch", hide_index=True)


def _show_persisted_history(payload: dict[str, Any]) -> None:
    run = payload.get("priority_run")
    if run is None:
        st.warning("No priority run snapshot found.")
        return
    st.json(
        {
            "priority_run_id": run.get("priority_run_id"),
            "restaurant_id": run.get("restaurant_id"),
            "review_month": run.get("review_month"),
            "generated_at": str(run.get("generated_at")),
            "scoring_config_hash": run.get("scoring_config_hash"),
            "absa_model_version": run.get("absa_model_version"),
        }
    )


def _show_persisted_data_quality(payload: dict[str, Any]) -> None:
    overview = payload.get("overview", {})
    quality = payload.get("data_quality", {})
    cols = st.columns(4)
    cols[0].metric("Missing review_time", overview.get("missing_review_time_count", 0))
    cols[1].metric(
        "Low confidence annotations",
        quality.get("low_confidence_annotation_count", 0),
    )
    cols[2].metric(
        "Low confidence rate",
        f"{quality.get('low_confidence_annotation_rate', 0.0):.2%}",
    )
    cols[3].metric("Peer benchmark rows", len(payload.get("peer_benchmark", [])))


def _target_reviews(reviews: list[ABSAReview], target_restaurant_id: str) -> list[ABSAReview]:
    target = [review for review in reviews if (review.restaurant_id or target_restaurant_id) == target_restaurant_id]
    return target or reviews


def _peer_benchmarks(
    extractions,
    label_schema: dict[str, Any],
    target_restaurant_id: str,
    review_month: str | None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    scoring_config = load_yaml("configs/scoring.yaml")
    month_extractions = [
        item
        for item in extractions
        if review_month is None or item.review_month == review_month
    ]
    if not month_extractions:
        return {}

    stats = aggregate_aspect_stats(month_extractions, scoring_config)
    global_rates = compute_global_negative_rate_by_aspect(month_extractions, label_schema)
    scoring = scoring_config.get("scoring", scoring_config)
    alpha = float(scoring.get("smoothing", {}).get("alpha", 10))
    stats = [
        item.model_copy(
            update={
                "negative_rate_smoothed": smoothed_negative_rate(
                    item.negative_count,
                    item.mention_count,
                    global_rates.get(item.aspect, 0.0),
                    alpha,
                )
            }
        )
        for item in stats
    ]
    target_months = {
        item.review_month
        for item in stats
        if item.restaurant_id == target_restaurant_id
    }
    target_aspects = {
        (item.review_month, item.aspect)
        for item in stats
        if item.restaurant_id == target_restaurant_id
    }
    benchmarks: dict[tuple[str, str, str], dict[str, Any]] = {}
    for month, aspect in target_aspects:
        if month not in target_months:
            continue
        peers = [
            item
            for item in stats
            if item.restaurant_id != target_restaurant_id
            and item.review_month == month
            and item.aspect == aspect
        ]
        if not peers:
            continue
        total_mentions = sum(item.mention_count for item in peers)
        total_negative = sum(item.negative_count for item in peers)
        benchmarks[(target_restaurant_id, month, aspect)] = {
            "peer_restaurant_count": len({item.restaurant_id for item in peers}),
            "peer_total_mentions": total_mentions,
            "peer_negative_rate": total_negative / total_mentions if total_mentions else 0.0,
        }
    return benchmarks


def _show_labels(label_schema: dict[str, Any]) -> None:
    with st.expander("Configured labels", expanded=False):
        col_aspects, col_sentiments = st.columns(2)
        with col_aspects:
            st.subheader("Aspects")
            st.write(", ".join(label_schema.get("aspects", [])))
        with col_sentiments:
            st.subheader("Sentiments")
            st.write(", ".join(label_schema.get("sentiments", [])))


def _show_overview(response: PriorityResponse, extractions) -> None:
    month_extractions = [
        item
        for item in extractions
        if response.review_month == "multiple" or item.review_month == response.review_month
        if response.restaurant_id == "multiple" or item.restaurant_id == response.restaurant_id
    ]
    total_reviews = len({item.review_id for item in month_extractions})
    total_annotations = len(month_extractions)
    negative_count = sum(item.sentiment == "negative" for item in month_extractions)
    ratings = [item.rating for item in month_extractions if item.rating is not None]

    cols = st.columns(4)
    cols[0].metric("Total reviews", total_reviews)
    cols[1].metric("Total ABSA annotations", total_annotations)
    cols[2].metric("Average rating", f"{_mean(ratings):.2f}" if ratings else "n/a")
    cols[3].metric(
        "Negative annotation rate",
        f"{negative_count / total_annotations:.2%}" if total_annotations else "0.00%",
    )

    st.subheader("Aspect mentions")
    st.bar_chart(
        _count_rows(month_extractions, "aspect"),
        x="label",
        y="count",
    )
    sentiment_rows = _sentiment_rows(month_extractions)
    st.subheader("Sentiment distribution by aspect")
    _show_sentiment_distribution_chart(sentiment_rows)


def _show_priority_items(response: PriorityResponse) -> None:
    if not response.items:
        st.warning("No priority items generated.")
        return

    rows = [
        {
            "Rank": item.rank,
            "Aspect": item.aspect,
            "Priority": item.priority_score,
            "Confidence": item.priority_confidence,
            "Negative rate": item.negative_rate_smoothed,
            "Severity": item.severity,
            "Trend": item.trend_score,
            "Peer gap": item.benchmark_gap,
            "Peer avg": item.peer_summary.peer_negative_rate,
            "Peer restaurants": item.peer_summary.peer_restaurant_count,
        }
        for item in response.items
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    _show_priority_bar_chart(rows)
    _show_trend_peer_gap_quadrant_chart(rows)
    _show_peer_benchmark_charts(rows)

    for item in response.items:
        with st.expander(f"#{item.rank} {item.aspect}", expanded=item.rank == 1):
            st.write("**Component scores**")
            component_rows = [
                {"component": key, "score": value}
                for key, value in item.component_scores.items()
            ]
            _show_metric_value_chart(
                [
                    {"metric": row["component"], "value": row["score"]}
                    for row in component_rows
                ]
            )
            st.dataframe(component_rows, width="stretch", hide_index=True)
            st.write("**Opinion examples**")
            for example in item.opinion_examples:
                st.write(f"- {example}")
            st.write("**Data quality flags**")
            st.write(", ".join(item.data_quality_flags) if item.data_quality_flags else "none")


def _show_aspect_detail(response: PriorityResponse) -> None:
    if not response.items:
        st.warning("No aspect detail available.")
        return
    selected = st.selectbox("Aspect", [item.aspect for item in response.items])
    item = next(row for row in response.items if row.aspect == selected)
    cols = st.columns(4)
    cols[0].metric("Rank", f"#{item.rank}")
    cols[1].metric("Priority score", f"{item.priority_score:.2f}")
    cols[2].metric("Confidence", f"{item.priority_confidence:.2f}")
    cols[3].metric("Severity", f"{item.severity:.2f}")
    metric_rows = [
        {"metric": "negative_rate", "value": item.negative_rate_smoothed},
        {"metric": "mention_share", "value": item.mention_share},
        {"metric": "rating_gap", "value": item.rating_gap},
        {"metric": "trend_score", "value": item.trend_score},
        {"metric": "benchmark_gap", "value": item.benchmark_gap},
    ]
    _show_metric_value_chart(metric_rows)
    st.dataframe(metric_rows, width="stretch", hide_index=True)


def _show_history(response: PriorityResponse) -> None:
    st.info(
        "History uses persisted DuckDB priority_runs/priority_items snapshots when storage is wired."
    )
    st.json(
        {
            "restaurant_id": response.restaurant_id,
            "review_month": response.review_month,
            "generated_at": response.generated_at.isoformat(),
        }
    )


def _show_data_quality(response: PriorityResponse, extractions) -> None:
    target_extractions = [
        item
        for item in extractions
        if response.restaurant_id == "multiple" or item.restaurant_id == response.restaurant_id
    ]
    missing_time = sum(item.review_time is None for item in target_extractions)
    low_confidence = sum(
        item.model_confidence is not None and item.model_confidence < 0.5
        for item in target_extractions
    )
    cols = st.columns(4)
    cols[0].metric("Missing review_time", missing_time)
    cols[1].metric("Low confidence annotations", low_confidence)
    cols[2].metric(
        "Aspects missing peer benchmark",
        sum("low_peer_support" in item.data_quality_flags for item in response.items),
    )
    cols[3].metric(
        "Aspects missing history",
        sum("insufficient_history" in item.data_quality_flags for item in response.items),
    )
    flag_rows = [
        {"aspect": item.aspect, "flags": ", ".join(item.data_quality_flags)}
        for item in response.items
    ]
    _show_data_quality_flag_charts(flag_rows)
    st.dataframe(flag_rows, width="stretch", hide_index=True)


def _show_sentiment_distribution_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar()
        .encode(
            x=alt.X("aspect:N", sort="-y", title="Aspect"),
            y=alt.Y("count:Q", stack="normalize", title="Sentiment share"),
            color=alt.Color("sentiment:N", title="Sentiment"),
            tooltip=["aspect:N", "sentiment:N", "count:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def _show_priority_bar_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar()
        .encode(
            x=alt.X("Priority:Q", title="Priority score"),
            y=alt.Y("Aspect:N", sort="-x", title="Aspect"),
            color=alt.Color("Priority:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["Rank:Q", "Aspect:N", "Priority:Q", "Confidence:Q", "Negative rate:Q"],
        )
        .properties(height=max(240, 32 * len(rows)))
    )
    st.altair_chart(chart, width="stretch")


def _show_metric_value_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="Value"),
            y=alt.Y("metric:N", sort="-x", title="Metric"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["metric:N", "value:Q"],
        )
        .properties(height=max(220, 34 * len(rows)))
    )
    st.altair_chart(chart, width="stretch")


def _show_trend_peer_gap_quadrant_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    chart_rows = []
    for row in rows:
        trend = float(row.get("Trend", 0.0) or 0.0)
        peer_gap = float(row.get("Peer gap", 0.0) or 0.0)
        # The raw scoring components are naturally small deltas around zero. Keep the
        # original values for tooltips/table, but plot signed deltas on a compact,
        # symmetric domain so bubbles do not collapse into the lower-left corner.
        chart_row = dict(row)
        chart_row["Trend plot"] = trend
        chart_row["Peer gap plot"] = peer_gap
        chart_rows.append(chart_row)

    max_abs_trend = max([abs(row["Trend plot"]) for row in chart_rows] + [0.02])
    max_abs_peer_gap = max([abs(row["Peer gap plot"]) for row in chart_rows] + [0.02])
    trend_padding = max(0.01, max_abs_trend * 0.30)
    peer_gap_padding = max(0.01, max_abs_peer_gap * 0.30)
    trend_domain = [-max_abs_trend - trend_padding, max_abs_trend + trend_padding]
    peer_gap_domain = [-max_abs_peer_gap - peer_gap_padding, max_abs_peer_gap + peer_gap_padding]
    midpoint = 0.0

    chart = (
        alt.Chart(alt.Data(values=chart_rows))
        .mark_circle(opacity=0.82)
        .encode(
            x=alt.X("Peer gap plot:Q", scale=alt.Scale(domain=peer_gap_domain), title="Peer gap / benchmark gap"),
            y=alt.Y("Trend plot:Q", scale=alt.Scale(domain=trend_domain), title="Trend score"),
            size=alt.Size("Priority:Q", title="Priority score", legend=alt.Legend(orient="right")),
            color=alt.Color(
                "Severity:Q",
                scale=alt.Scale(scheme="reds"),
                title="Severity",
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                "Rank:Q",
                "Aspect:N",
                "Priority:Q",
                "Trend:Q",
                "Peer gap:Q",
                "Severity:Q",
                "Negative rate:Q",
                "Peer avg:Q",
                "Peer restaurants:Q",
            ],
        )
        .properties(height=460)
    )
    vertical_rule = (
        alt.Chart(alt.Data(values=[{"x": midpoint}]))
        .mark_rule(color="gray", strokeDash=[6, 4])
        .encode(x="x:Q")
    )
    horizontal_rule = (
        alt.Chart(alt.Data(values=[{"y": midpoint}]))
        .mark_rule(color="gray", strokeDash=[6, 4])
        .encode(y="y:Q")
    )
    st.subheader("Trend vs Peer Gap Quadrant")
    st.caption(
        "Upper-right aspects are both worsening over time and performing worse than peers. "
        "Bubble size represents priority score; color represents severity."
    )
    layered_chart = (chart + vertical_rule + horizontal_rule).configure_legend(
        orient="right",
        direction="vertical",
        titleLimit=180,
        labelLimit=180,
    )
    st.altair_chart(layered_chart, width="stretch")


def _show_peer_benchmark_charts(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    normalized_rows = [_normalize_peer_benchmark_row(row) for row in rows]
    comparison_rows = [
        {"Aspect": row["Aspect"], "Metric": "Target negative rate", "Rate": row["Target negative rate"]}
        for row in normalized_rows
    ] + [
        {"Aspect": row["Aspect"], "Metric": "Peer avg", "Rate": row["Peer avg"]}
        for row in normalized_rows
    ]

    comparison_chart = (
        alt.Chart(alt.Data(values=comparison_rows))
        .mark_bar()
        .encode(
            x=alt.X("Aspect:N", title="Aspect"),
            y=alt.Y("Rate:Q", title="Negative rate"),
            xOffset=alt.XOffset("Metric:N"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Aspect:N", "Metric:N", "Rate:Q"],
        )
        .properties(height=320)
    )
    st.caption("Target vs Peer negative rate")
    st.altair_chart(comparison_chart, width="stretch")



def _show_data_quality_flag_charts(rows: list[dict[str, Any]]) -> None:
    flag_rows: list[dict[str, Any]] = []
    for row in rows:
        aspect = str(row.get("aspect", ""))
        flags = [
            flag.strip()
            for flag in str(row.get("flags", "")).split(",")
            if flag.strip()
        ]
        flag_rows.extend({"aspect": aspect, "flag": flag, "count": 1} for flag in flags)
    if not flag_rows:
        st.caption("No data quality flags to visualize.")
        return

    flag_count_chart = (
        alt.Chart(alt.Data(values=flag_rows))
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Flag count"),
            y=alt.Y("flag:N", sort="-x", title="Flag"),
            color=alt.Color("flag:N", legend=None),
            tooltip=["flag:N", "count():Q"],
        )
        .properties(height=max(220, 34 * len({row["flag"] for row in flag_rows})))
    )
    st.caption("Data quality flag counts")
    st.altair_chart(flag_count_chart, width="stretch")

    aspect_flag_chart = (
        alt.Chart(alt.Data(values=flag_rows))
        .mark_bar()
        .encode(
            x=alt.X("aspect:N", title="Aspect"),
            y=alt.Y("count():Q", title="Flag count"),
            color=alt.Color("flag:N", title="Flag"),
            tooltip=["aspect:N", "flag:N", "count():Q"],
        )
        .properties(height=300)
    )
    st.caption("Flags by aspect")
    st.altair_chart(aspect_flag_chart, width="stretch")


def _normalize_peer_benchmark_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Aspect": row.get("Aspect", row.get("aspect", "")),
        "Target negative rate": float(
            row.get(
                "Target negative rate",
                row.get("target_negative_rate", row.get("Negative rate", 0.0)),
            )
            or 0.0
        ),
        "Peer avg": float(row.get("Peer avg", row.get("peer_negative_rate", 0.0)) or 0.0),
        "Peer restaurants": int(row.get("Peer restaurants", row.get("peer_restaurant_count", 0)) or 0),
        "Peer gap": float(row.get("Peer gap", row.get("benchmark_gap", 0.0)) or 0.0),
        "Flag": row.get("Flag", row.get("peer_support_flag", "")) or "",
    }


def _count_rows(items, field: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in items:
        label = getattr(item, field)
        counts[label] = counts.get(label, 0) + 1
    return [{"label": label, "count": count} for label, count in sorted(counts.items())]


def _sentiment_rows(items) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item.aspect, item.sentiment)
        grouped[key] = grouped.get(key, 0) + 1
    return [
        {"aspect": aspect, "sentiment": sentiment, "count": count}
        for (aspect, sentiment), count in sorted(grouped.items())
    ]


def _restaurant_id_from_url(url: str) -> str:
    parsed = urlparse(url.strip())
    slug = ""
    if parsed.path:
        parts = [part for part in parsed.path.split("/") if part]
        if "place" in parts:
            index = parts.index("place")
            if index + 1 < len(parts):
                slug = unquote(parts[index + 1])
        elif parts:
            slug = unquote(parts[-1])
    base = _slugify(slug or parsed.netloc or "gmaps_restaurant")
    digest = hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:10]
    return f"{base}_{digest}"


def _area_id_from_name(area_name: str) -> str:
    return _slugify(area_name.strip() or "gmaps_area")


def _normalize_area_name_for_discovery(area_name: str) -> str | None:
    """Return a Nominatim-friendly administrative area name.

    Users often paste the restaurant street address into the area field. The
    crawler's --area-name must resolve to an administrative polygon, so for
    address-like inputs we keep the ward/city/country suffix.
    """
    cleaned = area_name.strip()
    if not cleaned:
        return None

    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) < 3:
        return cleaned

    first_part = parts[0].lower()
    address_markers = (
        "ngÃ¡ch",
        "ngÃµ",
        "ngo ",
        "Ä‘Æ°á»ng",
        "duong",
        "sá»‘",
        "so ",
        "háº»m",
        "hem",
        "kiá»‡t",
        "kiet",
    )
    looks_like_street_address = any(character.isdigit() for character in first_part) or any(
        marker in first_part for marker in address_markers
    )
    if not looks_like_street_address:
        return cleaned

    return ", ".join(parts[-3:])


def _slugify(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized[:80] or "unknown"


def _is_valid_month(value: str) -> bool:
    if len(value) != 7 or value[4] != "-":
        return False
    year, month = value.split("-", 1)
    return year.isdigit() and month.isdigit() and 1 <= int(month) <= 12


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    main()



