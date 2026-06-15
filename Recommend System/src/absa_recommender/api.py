from typing import Any

from fastapi import FastAPI

from absa_recommender.absa_inference import build_absa_adapter, infer_absa_with_adapter
from absa_recommender.config import load_label_schema
from absa_recommender.monthly_pipeline import run_monthly_from_reviews
from absa_recommender.recommender import generate_priority_ranking
from absa_recommender.schemas import ABSAReview, PriorityResponse
from absa_recommender.storage import (
    aspect_history as storage_aspect_history,
    dashboard_payload,
    default_db_path,
    get_latest_priority_response,
    list_peer_benchmark,
    list_priority_runs,
    save_crawl_run,
)

app = FastAPI(title="ABSA Aspect Priority Engine")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/labels")
def labels() -> dict[str, list[str]]:
    schema = load_label_schema("configs/label_schema.yaml")
    return {
        "aspects": list(schema.get("aspects", [])),
        "sentiments": list(schema.get("sentiments", [])),
    }


@app.post("/api/v1/priority/run", response_model=PriorityResponse)
def priority_from_absa(
    records: list[ABSAReview],
    top_n: int = 5,
    restaurant_id: str = "unknown",
    month: str | None = None,
) -> PriorityResponse:
    return generate_priority_ranking(
        records,
        top_n=top_n,
        default_restaurant_id=restaurant_id,
        review_month=month,
    )


@app.post("/api/v1/monthly/run")
def monthly_run(
    records: list[ABSAReview],
    top_n: int = 5,
    restaurant_id: str = "unknown",
    month: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if month is None:
        raise ValueError("month is required for persisted monthly runs")
    return run_monthly_from_reviews(
        records,
        restaurant_id=restaurant_id,
        review_month=month,
        top_n=top_n,
        db_path=default_db_path(),
        force=force,
    )


@app.post("/api/v1/monthly/run-raw")
def monthly_run_raw(
    records: list[dict[str, Any]],
    top_n: int = 5,
    restaurant_id: str = "unknown",
    month: str | None = None,
    force: bool = False,
    absa_adapter: str = "placeholder",
) -> dict[str, Any]:
    if month is None:
        raise ValueError("month is required for persisted monthly runs")
    adapter = build_absa_adapter(absa_adapter)
    reviews = infer_absa_with_adapter(records, adapter=adapter)
    result = run_monthly_from_reviews(
        reviews,
        restaurant_id=restaurant_id,
        review_month=month,
        top_n=top_n,
        db_path=default_db_path(),
        force=force,
        absa_model_version=adapter.model_version,
    )
    result["absa_adapter"] = adapter.model_version
    return result


@app.post("/api/v1/absa/infer")
def infer_absa(records: list[dict[str, Any]], adapter_name: str = "preannotated") -> dict[str, Any]:
    adapter = build_absa_adapter(adapter_name)
    reviews = infer_absa_with_adapter(records, adapter=adapter)
    return {
        "status": "completed",
        "adapter": adapter.model_version,
        "review_count": len(reviews),
        "annotation_count": sum(len(review.annotations) for review in reviews),
    }


@app.post("/api/v1/crawl/run")
def crawl_run(restaurant_id: str, month: str) -> dict[str, str]:
    crawl_run_id = save_crawl_run(
        default_db_path(),
        source="manual",
        target_month=month,
        area_id="local",
        status="created",
        num_restaurants=1,
        num_reviews_fetched=0,
        num_reviews_inserted=0,
        num_duplicates=0,
    )
    return {
        "status": "created",
        "crawl_run_id": crawl_run_id,
        "restaurant_id": restaurant_id,
        "review_month": month,
    }


@app.get("/api/v1/restaurants/{restaurant_id}/priority")
def restaurant_priority(
    restaurant_id: str,
    month: str,
    top_n: int = 5,
) -> dict[str, Any]:
    run = get_latest_priority_response(default_db_path(), restaurant_id, month)
    if run is None:
        return {"restaurant_id": restaurant_id, "review_month": month, "items": [], "status": "missing"}
    output = run["output"]
    return {
        "restaurant_id": restaurant_id,
        "review_month": month,
        "top_n": top_n,
        "items": output.get("items", [])[:top_n],
        "priority_run_id": run["priority_run_id"],
        "status": run["status"],
    }


@app.get("/api/v1/restaurants/{restaurant_id}/dashboard")
def restaurant_dashboard(restaurant_id: str, month: str) -> dict[str, Any]:
    return dashboard_payload(default_db_path(), restaurant_id, month)


@app.get("/api/v1/restaurants/{restaurant_id}/history")
def restaurant_history(restaurant_id: str) -> dict[str, Any]:
    return {"restaurant_id": restaurant_id, "runs": list_priority_runs(default_db_path(), restaurant_id)}


@app.get("/api/v1/restaurants/{restaurant_id}/aspects/{aspect}/history")
def aspect_history(restaurant_id: str, aspect: str) -> dict[str, Any]:
    return {
        "restaurant_id": restaurant_id,
        "aspect": aspect,
        "history": storage_aspect_history(default_db_path(), restaurant_id, aspect),
    }


@app.get("/api/v1/restaurants/{restaurant_id}/peer-benchmark")
def peer_benchmark(restaurant_id: str, month: str) -> dict[str, Any]:
    return {
        "restaurant_id": restaurant_id,
        "review_month": month,
        "items": list_peer_benchmark(default_db_path(), restaurant_id, month),
    }
