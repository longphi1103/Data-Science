from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ABSAAnnotation(BaseModel):
    aspect_expression: str
    aspect_category: str
    opinion_expression: str
    sentiment: str
    model_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ABSAReview(BaseModel):
    review_id: str
    review_text: str
    restaurant_id: Optional[str] = None
    restaurant_name: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    review_time: Optional[datetime] = None
    review_month: Optional[str] = None
    annotations: list[ABSAAnnotation]


class AspectExtraction(BaseModel):
    extraction_id: str
    review_id: str
    restaurant_id: str
    restaurant_name: Optional[str]
    aspect: str
    aspect_term: str
    opinion_text: str
    sentiment: str
    severity: float = Field(ge=0.0, le=1.0)
    model_confidence: Optional[float]
    review_text: str
    rating: Optional[int]
    review_time: Optional[datetime]
    review_month: str


class AspectStats(BaseModel):
    restaurant_id: str
    review_month: str
    aspect: str
    mention_count: int
    negative_count: int
    positive_count: int
    neutral_count: int
    negative_rate_raw: float = Field(ge=0.0, le=1.0)
    negative_rate_smoothed: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_severity: float = Field(ge=0.0, le=1.0)
    avg_rating: float
    avg_confidence: float = Field(ge=0.0, le=1.0)
    mention_share: float = Field(default=0.0, ge=0.0, le=1.0)
    rating_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    total_mentions_for_restaurant: int
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None


class PeerSummary(BaseModel):
    peer_restaurant_count: int = 0
    peer_negative_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    target_vs_peer_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    peer_support_flag: str | None = None


class TrendSummary(BaseModel):
    previous_month_priority_score: float | None = None
    priority_delta: float | None = None
    negative_rate_delta: float | None = None
    trend_flag: str | None = None


class PriorityItem(BaseModel):
    rank: int
    aspect: str
    priority_score: float = Field(ge=0.0, le=100.0)
    priority_confidence: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    mention_count: int
    negative_count: int
    negative_rate_smoothed: float = Field(ge=0.0, le=1.0)
    mention_share: float = Field(ge=0.0, le=1.0)
    rating_gap: float = Field(ge=0.0, le=1.0)
    trend_score: float = Field(ge=0.0, le=1.0)
    benchmark_gap: float = Field(ge=0.0, le=1.0)
    risk_multiplier: float
    opinion_examples: list[str]
    component_scores: dict[str, float]
    peer_summary: PeerSummary
    trend_summary: TrendSummary
    data_quality_flags: list[str]


class PriorityResponse(BaseModel):
    restaurant_id: str
    restaurant_name: Optional[str] = None
    review_month: str
    generated_at: datetime
    top_n: int
    items: list[PriorityItem]


RecommendationItem = PriorityItem
RecommendationResponse = PriorityResponse
