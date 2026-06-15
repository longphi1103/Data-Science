# ABSA Aspect Priority Engine

Local-first priority scoring for Vietnamese restaurant ABSA outputs.

The system ranks **Top-N aspects to improve** for a target restaurant and review month. It does not detect sub-problems, recommend actions, or use an action catalog.

## Quick Start

```bash
# Validate that the sample ABSA JSONL can be parsed and labels match configs/label_schema.yaml.
uv run absa-priority validate --input data/samples/streamlit_priority_200.jsonl

# Run the full local pipeline: source JSONL crawl, ABSA inference, DuckDB persistence and priority scoring.
uv run absa-priority run-full --input data/samples/streamlit_priority_200.jsonl --restaurant-id res_demo --month 2026-06 --top-n 10 --absa-adapter placeholder --source-adapter local-jsonl --db-path data/local.duckdb --force

# Print the persisted dashboard payload from DuckDB for the target restaurant/month.
uv run absa-priority show-dashboard --restaurant-id res_demo --month 2026-06 --db-path data/local.duckdb

# Start the Streamlit dashboard; enable "Load from DuckDB" and use data/local.duckdb.
# In DuckDB mode, enter a Google Maps restaurant URL, crawl month and ward/area, then click "Run Explore".
uv run streamlit run app/streamlit_app.py

# Start the FastAPI service; persisted endpoints read ABSA_DB_PATH or data/local.duckdb.
uv run uvicorn absa_recommender.api:app --reload
```

The old `absa-rec` script remains as a compatibility alias, but `absa-priority` is the preferred command.

## Current Scope

Implemented:

- ABSA JSONL validation and normalization.
- `aspect_category -> aspect`, `aspect_expression -> aspect_term`, `opinion_expression -> opinion_text`.
- Severity scoring.
- Monthly aggregation by `restaurant_id`, `review_month`, `aspect`.
- Peer benchmark from peer restaurant rows in the same persisted dataset.
- Previous-priority lookup for trend when prior months exist.
- Priority score and priority confidence.
- Top-N aspect ranking.
- DuckDB persistence for reviews, annotations, stats, peer benchmark, priority runs and priority items.
- FastAPI endpoints that read persisted DuckDB snapshots.
- Streamlit upload mode, DuckDB mode, and Google Maps Explore controls for URL + month driven runs.
- Docker Compose services for API, Streamlit, one-shot full monthly run and monthly scheduler.
- Local compliant crawler orchestration for source JSONL ingestion, monthly filtering, caps, normalization, deduplication and crawl audit rows.
- Google Maps crawler source adapter that invokes `.localworkspace/gmaps_url_crawler_single_discovery.py` and reads its JSONL output.
- ViT5/ACOS ABSA adapter for `models/acos_vit5_large_final.zip`, plus pre-annotated and placeholder adapters.

Removed:

- Sub-problem rules, locator and prototype matcher.
- Action catalog and action recommendation output.
- Taxonomy miner/review flow.
- Action feedback endpoint.

Available local adapters:

- `LocalJsonlAdapter`: reads local JSONL files for demo/deployment tests.
- `PreAnnotatedABSAAdapter`: validates pre-annotated ABSA records and treats them as inference output.
- `PlaceholderABSAAdapter`: deterministic low-confidence placeholder that converts raw review text into ABSA-shaped annotations for integration testing.
- `ViT5ABSAAdapter`: loads the trained zipped Hugging Face seq2seq model from `models/acos_vit5_large_final.zip`.
- `GoogleMapsCrawlerAdapter`: runs `.localworkspace/gmaps_url_crawler_single_discovery.py` and consumes the generated monthly JSONL.

Still external by design:

- Production credentials/API policy for live review-source collection.

The repo provides explicit adapter interfaces for those integrations; it does not implement review scraping or anti-detection scraping tactics. Source integrations should use licensed/API-compliant providers with rate limits, retry/backoff and source-policy checks.

## Input ABSA Format

Each JSONL line is one review:

```json
{
  "review_id": "rv_001",
  "review_text": "Ban hoi ban nhung nhan vien than thien.",
  "restaurant_id": "res_001",
  "restaurant_name": "Nha hang A",
  "rating": 3,
  "review_time": "2026-06-14T10:30:00",
  "review_month": "2026-06",
  "annotations": [
    {
      "aspect_expression": "ban",
      "aspect_category": "Cleanliness",
      "opinion_expression": "hoi ban",
      "sentiment": "negative",
      "model_confidence": 0.91
    }
  ]
}
```

`review_month` is optional. If missing and `review_time` exists, the normalizer derives `YYYY-MM`; otherwise it uses `unknown`.

The sample `data/samples/streamlit_priority_200.jsonl` contains:

- target restaurant `res_demo`
- peers `res_peer_01` to `res_peer_06`
- month `2026-06`
- enough peer rows to exercise Streamlit peer benchmark charts

## Streamlit Google Maps Explore

Streamlit DuckDB mode includes a sidebar **Google Maps Explore** form:

1. Enable `Load from DuckDB`.
2. Enter the DuckDB path, usually `data/local.duckdb` or `/app/data/local.duckdb` in Docker.
3. Enter a Google Maps URL for one restaurant/eatery.
4. Choose a crawl month in `YYYY-MM` format.
5. Enter the ward/area name used for peer discovery.
6. Click **Run Explore**.

The app derives a stable default `restaurant_id` from the URL, but it can be overridden before running. The Explore button calls the existing monthly pipeline with the Google Maps source adapter:

```text
GoogleMapsCrawlerAdapter
-> crawl_reviews_for_month
-> selected ABSA adapter
-> run_monthly_from_reviews
-> DuckDB dashboard payload
```

If a priority run already exists for the same `restaurant_id` and month, Streamlit skips processing and displays the existing run id instead of creating duplicate DuckDB records.

`Live Google Maps crawl` is disabled by default. In offline/demo mode, the bundled crawler wrapper only uses available local/demo data. For Docker deployments, use `placeholder` ABSA unless the trained ViT5 tokenizer/model artifacts are available and compatible inside the image.

## Raw Review Input

The full local pipeline can also start from raw review JSONL. Each line should contain at least:

```json
{
  "review_id": "rv_001",
  "review_text": "Nhan vien phuc vu cham, ban hoi ban.",
  "restaurant_id": "res_001",
  "restaurant_name": "Nha hang A",
  "rating": 2,
  "review_time": "2026-06-14T10:30:00",
  "review_month": "2026-06",
  "source": "local_jsonl",
  "source_review_id": "source_rv_001",
  "language": "vi"
}
```

`run-full` uses `LocalJsonlAdapter` and `PlaceholderABSAAdapter` by default. Use `--source-adapter google-maps` to invoke the bundled Google Maps crawler wrapper, and `--absa-adapter trained`/`vit5` to load the zipped ViT5 model.

## Output Shape

`generate_priority_ranking(...)` and persisted priority runs return:

```json
{
  "restaurant_id": "res_demo",
  "restaurant_name": "Nha hang Demo",
  "review_month": "2026-06",
  "generated_at": "2026-06-02T00:00:00Z",
  "top_n": 10,
  "items": [
    {
      "rank": 1,
      "aspect": "Service",
      "priority_score": 39.88,
      "priority_confidence": 0.68,
      "severity": 0.75,
      "mention_count": 103,
      "negative_count": 80,
      "negative_rate_smoothed": 0.74,
      "mention_share": 0.84,
      "rating_gap": 0.52,
      "trend_score": 0.0,
      "benchmark_gap": 1.0,
      "risk_multiplier": 1.0,
      "component_scores": {
        "negative_rate": 0.74,
        "sentiment_severity": 0.75,
        "mention_share": 0.84,
        "rating_gap": 0.52,
        "trend_score": 0.0,
        "benchmark_gap": 1.0
      },
      "peer_summary": {
        "peer_restaurant_count": 6,
        "peer_negative_rate": 0.0,
        "target_vs_peer_gap": 1.0,
        "peer_support_flag": null
      },
      "trend_summary": {
        "previous_month_priority_score": null,
        "priority_delta": null,
        "negative_rate_delta": null,
        "trend_flag": "insufficient_history"
      },
      "opinion_examples": ["..."],
      "data_quality_flags": ["insufficient_history"]
    }
  ]
}
```

There is intentionally no `sub_problem_id`, `recommended_actions`, or `monitoring_kpis`.

## Scoring

Components are normalized to `[0, 1]`:

- `negative_rate`: Bayesian-smoothed negative annotation rate.
- `sentiment_severity`: average severity.
- `mention_share`: `log1p(mention_count) / log1p(total_mentions_for_restaurant)`.
- `rating_gap`: `(5 - avg_rating) / 4`.
- `trend_score`: month-over-month deterioration when history exists.
- `benchmark_gap`: target negative rate above peer average when peer support is sufficient.

Final score:

```text
priority_score = 100 * clamp(risk_multiplier[aspect] * sum(weight_i * component_i), 0, 1)
```

`priority_confidence` blends support, model, peer and history confidence according to `configs/scoring.yaml`.

## CLI

```bash
# Validate ABSA JSONL and label schema compatibility.
uv run absa-priority validate --input data/samples/streamlit_priority_200.jsonl

# Score priority in memory and write JSON output; this does not persist DuckDB tables.
uv run absa-priority score-priority --input data/samples/streamlit_priority_200.jsonl --restaurant-id res_demo --month 2026-06 --top-n 10 --output out/priority.json

# Run the persisted monthly pipeline: reviews, annotations, stats, peer benchmark and priority snapshot are saved to DuckDB.
uv run absa-priority run-monthly --input data/samples/streamlit_priority_200.jsonl --restaurant-id res_demo --month 2026-06 --top-n 10 --db-path data/local.duckdb --output out/priority.json --force

# Run the full local pipeline from source JSONL through placeholder ABSA inference and persisted priority scoring.
uv run absa-priority run-full --input data/samples/streamlit_priority_200.jsonl --restaurant-id res_demo --month 2026-06 --top-n 10 --absa-adapter placeholder --source-adapter local-jsonl --db-path data/local.duckdb --output out/priority.json --force

# Run the integrated Google Maps crawler wrapper plus trained ViT5 ABSA adapter.
uv run absa-priority run-full --input data/gmaps_monthly_raw.jsonl --restaurant-id res_demo --month 2026-06 --top-n 10 --source-adapter google-maps --absa-adapter trained --db-path data/local.duckdb --output out/priority.json --force

# Compute aspect-level stats from an ABSA JSONL file and write a JSON export.
uv run absa-priority compute-stats --input data/samples/streamlit_priority_200.jsonl --restaurant-id res_demo --month 2026-06 --output out/aspect_monthly_stats.json

# Print persisted dashboard payload from DuckDB.
uv run absa-priority show-dashboard --restaurant-id res_demo --month 2026-06 --db-path data/local.duckdb

# List persisted priority runs, optionally filtered by restaurant.
uv run absa-priority list-runs --restaurant-id res_demo --db-path data/local.duckdb

# Print persisted history for a single aspect.
uv run absa-priority aspect-history --restaurant-id res_demo --aspect Service --db-path data/local.duckdb

# Print official labels loaded from configs/label_schema.yaml.
uv run absa-priority show-labels
```

Adapter-oriented commands:

```bash
# List peers already persisted in DuckDB for this target restaurant.
uv run absa-priority discover-peers res_demo --radius-meters 1500 --db-path data/local.duckdb

# Create a manual crawl_runs record for adapter orchestration when no local input is supplied.
uv run absa-priority crawl-month res_demo 2026-06 --db-path data/local.duckdb

# Crawl monthly reviews from local JSONL, normalize/deduplicate them and persist restaurants/reviews/crawl_runs into DuckDB.
uv run absa-priority crawl-month res_demo 2026-06 --input data/samples/streamlit_priority_200.jsonl --db-path data/local.duckdb

# Validate pre-annotated ABSA JSONL through the local ABSA adapter for one month.
uv run absa-priority infer-absa 2026-06 --input data/samples/streamlit_priority_200.jsonl --adapter preannotated

# Run placeholder ABSA inference on raw review rows for integration testing before plugging in a trained model.
uv run absa-priority infer-absa 2026-06 --input data/samples/streamlit_priority_200.jsonl --adapter placeholder

# Run persisted monthly scoring across a month range; months absent from the JSONL are skipped.
uv run absa-priority backfill res_demo 2026-01 2026-06 --input data/samples/streamlit_priority_200.jsonl --db-path data/local.duckdb --top-n 10
```

## Monthly Pipeline

`src/absa_recommender/monthly_pipeline.py` implements two persisted local paths.

`run_monthly_from_absa_jsonl(...)` starts from existing ABSA annotations:

1. Load ABSA reviews.
2. Filter target `review_month`.
3. Normalize and deduplicate reviews.
4. Save `restaurants`.
5. Save `crawl_runs`.
6. Save `reviews`.
7. Flatten annotations and save `absa_annotations`.
8. Compute and save `aspect_monthly_stats`.
9. Compute and save `peer_aspect_monthly_stats`.
10. Load previous priority by aspect when available.
11. Score Top-N aspects and save `priority_runs` plus `priority_items`.

`run_monthly_from_source(...)` starts from source review JSONL:

1. Load monthly review rows through `LocalJsonlAdapter` or `GoogleMapsCrawlerAdapter`.
2. Apply crawler strategy from `configs/crawler.yaml`.
3. Normalize and deduplicate raw reviews.
4. Run selected ABSA adapter, default `placeholder`; `trained`/`vit5` loads `models/acos_vit5_large_final.zip`.
5. Reuse the persisted monthly pipeline for annotations, stats, peer benchmark, trend and priority scoring.

Crawler strategy is intentionally compliant rather than evasive:

- cap reviews per restaurant and restaurants per run;
- use retry/backoff and configurable request pacing for real source adapters;
- require licensed/API-compliant sources;
- do not implement stealth browsers, CAPTCHA bypass or proxy rotation for evasion.

Idempotency key:

```text
restaurant_id + review_month + scoring_config_hash + absa_model_version
```

Use `--force` to create a new run for the same key.

## API

FastAPI title: `ABSA Aspect Priority Engine`.

Routes:

- `GET /health`
- `GET /api/v1/labels`
- `POST /api/v1/priority/run`
- `POST /api/v1/monthly/run`
- `POST /api/v1/monthly/run-raw`
- `POST /api/v1/absa/infer`
- `POST /api/v1/crawl/run`
- `GET /api/v1/restaurants/{restaurant_id}/priority?month=2026-06&top_n=10`
- `GET /api/v1/restaurants/{restaurant_id}/dashboard?month=2026-06`
- `GET /api/v1/restaurants/{restaurant_id}/history`
- `GET /api/v1/restaurants/{restaurant_id}/aspects/{aspect}/history`
- `GET /api/v1/restaurants/{restaurant_id}/peer-benchmark?month=2026-06`

`POST /api/v1/monthly/run` persists pre-annotated ABSA records to `ABSA_DB_PATH` or `data/local.duckdb`.

`POST /api/v1/monthly/run-raw` accepts raw review records, runs the selected ABSA adapter, then persists the monthly priority run.

## Streamlit Dashboard

```bash
# Start dashboard. Use upload mode for ad hoc JSONL, or enable "Load from DuckDB" after run-monthly.
uv run streamlit run app/streamlit_app.py
```

Tabs:

- Monthly Overview
- Top-N Aspects
- Aspect Detail
- Peer Benchmark
- History
- Data Quality

Upload mode computes peer benchmark from a multi-restaurant JSONL file. DuckDB mode reads persisted monthly runs and dashboard payloads.

## DuckDB Storage

`src/absa_recommender/storage.py` initializes and reads/writes:

- `restaurants`
- `crawl_runs`
- `reviews`
- `absa_annotations`
- `aspect_monthly_stats`
- `peer_aspect_monthly_stats`
- `priority_runs`
- `priority_items`

The API and Streamlit DuckDB mode read persisted snapshots from this schema.

## Docker Deployment

Copy `.env.example` to `.env` and adjust values if needed.

```bash
# Build all Docker images.
docker compose build

# Run the one-shot full monthly pipeline job into ./data/local.duckdb.
docker compose --profile job run --rm monthly-run

# Run the monthly scheduler container. configs/scheduler.yaml controls monthly timing.
docker compose --profile scheduler up -d monthly-scheduler

# Start API and Streamlit services.
docker compose up -d api streamlit

# Check API health.
curl http://localhost:8000/health
```

Ports:

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

The compose file mounts:

- `./data:/app/data`
- `./configs:/app/configs`
- `./models:/app/models`
- `./.localworkspace:/app/.localworkspace`
- `missing_review_time_rate`
- `absa_inference_failure_rate`
- `low_confidence_annotation_rate`
- `aspect_coverage`
- `peer_support_rate`
- `dashboard_data_freshness`

Suggested alerts include crawl success below 90%, peer support below 70%, missing review time above 30%, and low-confidence annotations above 25%.

## Configs

Kept:

- `configs/label_schema.yaml`
- `configs/severity_lexicon.yaml`
- `configs/scoring.yaml`
- `configs/text_normalization.yaml`

Added:

- `configs/crawler.yaml`
- `configs/peer_discovery.yaml`
- `configs/scheduler.yaml`
- `configs/absa_model.yaml`
- `configs/dashboard.yaml`
- `configs/source_policy.yaml`

Removed:

- `configs/subproblem_rules.yaml`
- `configs/subproblem_prototypes.yaml`
- `configs/locator.yaml`
- `configs/action_catalog.yaml`
- `configs/taxonomy_miner.yaml`

## Verification

```bash
# Run the full test suite.
uv run pytest

# Run lint checks.
uv run ruff check .
```
