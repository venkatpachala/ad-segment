from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Platform = Literal["youtube", "instagram", "file", "other"]
Kind = Literal["vod", "short", "live"]
AdType = Literal[
    "preroll",
    "midroll_sponsor_read",
    "product_placement",
    "self_promo",
    "affiliate",
    "platform_inserted",
    "bumper",
    "other",
]


class DetectRequest(BaseModel):
    """Internal pipeline request. Public POST /v1/detect ignores `kind` / `kind_hint`."""

    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    local_path: str | None = None
    file: Any | None = None
    kind_hint: Kind | None = None


class Source(BaseModel):
    url: str
    platform: Platform
    kind: Kind
    duration_s: float
    processed_at: str
    language: str = ""


class Evidence(BaseModel):
    frame_timestamps: list[float] = Field(default_factory=list)
    transcript_span: str = ""
    ocr_text: str = ""
    transcript_lang: str = ""
    signals_used: list[str] = Field(default_factory=list)


class Segment(BaseModel):
    id: str
    start_s: float
    end_s: float
    ad_type: AdType
    confidence: float
    brand: str = ""
    description: str = ""
    evidence: Evidence
    presentation: str | None = None


class Stats(BaseModel):
    wall_clock_s: float
    estimated_cost_usd: float
    frames_sampled: int
    frames_ocrd: int = 0
    ocr_wall_s: float = 0.0
    model_calls: int
    llm_fallback: str = ""


class DetectResponse(BaseModel):
    source: Source
    segments: list[Segment]
    stats: Stats


JobState = Literal[
    "queued",
    "downloading",
    "detecting",
    "done",
    "ingest_failed",
    "failed",
]


class JobError(BaseModel):
    code: str
    detail: str


class JobRecord(BaseModel):
    job_id: str
    status: JobState
    created_at: str
    error: JobError | None = None
    result: DetectResponse | None = None


class DetectApiResponse(BaseModel):
    """Public POST /v1/detect and GET /v1/jobs/{id} body."""

    job_id: str
    status: JobState
    source: Source | None = None
    segments: list[Segment] | None = None
    stats: Stats | None = None
    error: JobError | None = None


def job_to_api(rec: JobRecord) -> dict:
    """Flatten a job record to the public detect contract."""
    body: dict = {"job_id": rec.job_id, "status": rec.status}
    if rec.error is not None:
        body["error"] = {"code": rec.error.code, "detail": rec.error.detail}
    if rec.result is not None:
        dumped = rec.result.model_dump()
        body["source"] = dumped["source"]
        body["segments"] = dumped["segments"]
        body["stats"] = dumped["stats"]
    return body


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")