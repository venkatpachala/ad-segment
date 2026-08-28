"""Stable public error bodies. Never a Python traceback."""
from __future__ import annotations

from fastapi.responses import JSONResponse

from app.pipeline.ingest import IngestError

# Codes the client is documented to handle. Other ingest codes pass through.
_PUBLIC = {
    "youtube_bot_check",
    "no_tesseract",
    "unsupported",
    "unsupported_host",
    "unsupported_media",
    "private_video",
    "private_or_missing",
    "geo_blocked",
    "probe_failed",
    "yt_dlp_missing",
    "not_found",
    "pipeline_failed",
    "ingest_failed",
    "instagram_auth",
    "rate_limited",
}


def public_code(code: str) -> str:
    if code in {"unsupported_host"}:
        return "unsupported"
    if code in {"tesseract_missing", "no_tesseract"}:
        return "no_tesseract"
    return code if code in _PUBLIC or code else "unsupported"


def error_body(code: str, detail: str) -> dict:
    return {"code": public_code(code), "detail": detail}


def error_response(code: str, detail: str, status_code: int = 422) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(code, detail))


def ingest_response(e: IngestError) -> JSONResponse:
    detail = str(e)
    if e.code == "youtube_bot_check":
        detail = (
            f"{e}\n\nYouTube fetch is best-effort. Upload the mp4: "
            "POST /v1/detect (multipart file) or POST /v1/live/sessions (multipart file)."
        )
    if e.code in {"instagram_auth", "rate_limited", "private_or_missing", "unsupported_media"}:
        detail = (
            f"{e}\n\nInstagram fetch is cookie-based and best-effort. "
            "Upload the mp4: POST /v1/detect (multipart file)."
        )
    return error_response(e.code, detail, status_code=422)
