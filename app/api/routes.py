from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.api.errors import error_response, public_code
from app.api.jobs import get_job, submit_job, wait_for_job
from app.api.live_sessions import (
    create_session,
    get_session,
    list_sessions,
    stop_session,
)
from app.config import DETECT_SYNC_S, UPLOADS_DIR
from app.models.schemas import DetectRequest, job_to_api
from app.pipeline.ingest import cookie_status, is_live_url, ytdlp_available
from app.pipeline.ocr import tesseract_available

router = APIRouter()


class LiveSessionRequest(BaseModel):
    url: str | None = None
    file: Any | None = None
    local_path: str | None = None  # internal / tests
    hls_url: str | None = None  # alias of url
    tick_s: float = 2.0
    window_s: float = 16.0
    use_llm: bool = False


def _save_upload(file: UploadFile) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    dest = UPLOADS_DIR / f"{uuid4().hex}{suffix}"
    dest.write_bytes(file.file.read())
    return dest


async def _save_upload_async(upload: UploadFile) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "input.mp4").suffix or ".mp4"
    dest = UPLOADS_DIR / f"{uuid4().hex}{suffix}"
    dest.write_bytes(await upload.read())
    return dest


def _form_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return None


async def _parse_url_or_file(request: Request) -> tuple[str | None, Path | None]:
    """JSON `{url, file: null}` or multipart `url` / `file`."""
    ct = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ct or "application/x-www-form-urlencoded" in ct:
        form = await request.form()
        url = _form_str(form.get("url")) or _form_str(form.get("hls_url"))
        dest: Path | None = None
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            dest = await _save_upload_async(upload)  # type: ignore[arg-type]
        local = _form_str(form.get("local_path"))
        if dest is None and local:
            dest = Path(local)
        return url, dest

    try:
        data = await request.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    url = data.get("url") or data.get("hls_url")
    if isinstance(url, str):
        url = url.strip() or None
    else:
        url = None
    local = data.get("local_path")
    dest = Path(local) if isinstance(local, str) and local.strip() else None
    return url, dest


def _require_tesseract() -> JSONResponse | None:
    if tesseract_available():
        return None
    return error_response(
        "no_tesseract",
        "Tesseract is not installed or not on PATH. OCR is required for detection.",
        status_code=422,
    )


@router.get("/health")
def health():
    return {
        "ok": True,
        "tesseract": tesseract_available(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "yt_dlp": ytdlp_available(),
        "cookies": cookie_status(),
    }


def _detect_job(req: DetectRequest) -> JSONResponse:
    missing = _require_tesseract()
    if missing is not None:
        return missing
    rec = submit_job(req)
    rec = wait_for_job(rec.job_id, DETECT_SYNC_S) or rec
    body = job_to_api(rec)
    if rec.status == "done":
        return JSONResponse(status_code=200, content=body)
    if rec.status == "ingest_failed":
        code = rec.error.code if rec.error else "ingest_failed"
        return JSONResponse(
            status_code=422,
            content={
                "job_id": rec.job_id,
                "status": rec.status,
                "code": public_code(code),
                "detail": rec.error.detail if rec.error else "ingest failed",
            },
        )
    if rec.status == "failed":
        return JSONResponse(
            status_code=500,
            content={
                "job_id": rec.job_id,
                "status": rec.status,
                "code": "pipeline_failed",
                "detail": rec.error.detail if rec.error else "pipeline failed",
            },
        )
    return JSONResponse(status_code=202, content={"job_id": rec.job_id, "status": rec.status})


@router.post("/v1/detect")
async def detect(request: Request) -> JSONResponse:
    """One POST for Short and VOD. Kind is inferred from duration, never from the client.

    JSON: `{"url": "...", "file": null}`
    multipart: `file` = mp4 (production SLA) and/or `url`.
    Sync 200 if work finishes within DETECT_SYNC_S, else 202 + GET /v1/jobs/{id}.
    """
    url, dest = await _parse_url_or_file(request)
    if dest is None and not url:
        return error_response("unsupported", "url or file is required", status_code=400)
    if dest is None and is_live_url(url):
        return error_response(
            "unsupported",
            "Live streams use POST /v1/live/sessions. Upload a recording or pass a VOD/Short URL.",
            status_code=422,
        )
    req = DetectRequest(url=url, local_path=str(dest) if dest else None, kind_hint=None)
    return _detect_job(req)


@router.post("/v1/detect/upload")
def detect_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Alias of POST /v1/detect with multipart file (kept for older clients)."""
    dest = _save_upload(file)
    return _detect_job(DetectRequest(local_path=str(dest), kind_hint=None))


@router.post("/v1/jobs", status_code=202)
def create_job(req: DetectRequest) -> dict:
    if not req.url and not req.local_path:
        raise HTTPException(400, "url or local_path is required")
    rec = submit_job(DetectRequest(url=req.url, local_path=req.local_path, kind_hint=None))
    return {"job_id": rec.job_id, "status": rec.status}


@router.post("/v1/jobs/upload", status_code=202)
def create_job_upload(file: UploadFile = File(...)) -> dict:
    dest = _save_upload(file)
    rec = submit_job(DetectRequest(local_path=str(dest), kind_hint=None))
    return {"job_id": rec.job_id, "status": rec.status}


@router.get("/v1/jobs/{job_id}")
def read_job(job_id: str) -> JSONResponse:
    rec = get_job(job_id)
    if rec is None:
        return error_response("not_found", f"unknown job_id {job_id}", status_code=404)
    return JSONResponse(status_code=200, content=job_to_api(rec))


# ── Live session endpoints ────────────────────────────────────────────────────

@router.post("/v1/live/sessions")
async def create_live_session(request: Request) -> JSONResponse:
    """Start live detection. Returns immediately; poll events until POST .../stop.

    JSON: `{"url": "https://www.youtube.com/live/…", "file": null}`
    multipart: `file` = recording used as fake-live, and/or `url`.
    """
    missing = _require_tesseract()
    if missing is not None:
        return missing
    url, dest = await _parse_url_or_file(request)
    if dest is None and not url:
        return error_response("unsupported", "url or file is required", status_code=400)
    session = create_session(
        url=url,
        local_path=str(dest) if dest else None,
        growing=True if dest and Path(dest).suffix.lower() in {".ts", ".m2ts"} else None,
    )
    return JSONResponse(
        status_code=200,
        content={"session_id": session.session_id, "status": "running"},
    )


@router.get("/v1/live/sessions")
def list_live_sessions() -> list[dict]:
    return list_sessions()


@router.get("/v1/live/sessions/{session_id}")
def get_live_session(session_id: str) -> JSONResponse:
    session = get_session(session_id)
    if session is None:
        return error_response("not_found", f"unknown session_id {session_id}", status_code=404)
    summary = session.to_summary()
    if session.result:
        dumped = session.result.model_dump()
        summary["source"] = dumped["source"]
        summary["segments"] = dumped["segments"]
        summary["stats"] = dumped["stats"]
    if session.error_code:
        summary["error"] = {"code": session.error_code, "detail": session.error or ""}
    return JSONResponse(content=summary)


@router.get("/v1/live/sessions/{session_id}/events")
def get_live_session_events(
    session_id: str,
    after: int = Query(0, description="Return events with seq > after"),
) -> JSONResponse:
    session = get_session(session_id)
    if session is None:
        return error_response("not_found", f"unknown session_id {session_id}", status_code=404)
    events = session.get_events(after_seq=after)
    next_after = after
    if events:
        next_after = int(events[-1].get("seq", after))
    return JSONResponse(
        content={
            "session_id": session_id,
            "status": session.status,
            "events": events,
            "next_after": next_after,
        }
    )


@router.post("/v1/live/sessions/{session_id}/stop")
def stop_live_session(session_id: str) -> JSONResponse:
    """Ctrl+C: flush open candidates and return the final DetectResponse."""
    session = stop_session(session_id)
    if session is None:
        return error_response("not_found", f"unknown session_id {session_id}", status_code=404)
    body: dict = {
        "session_id": session.session_id,
        "status": session.status,
    }
    if session.result:
        dumped = session.result.model_dump()
        body["source"] = dumped["source"]
        body["segments"] = dumped["segments"]
        body["stats"] = dumped["stats"]
    if session.error_code:
        body["error"] = {"code": session.error_code, "detail": session.error or ""}
        if session.status == "ingest_failed":
            return JSONResponse(status_code=422, content=body)
    return JSONResponse(content=body)


@router.get("/v1/live/sessions/{session_id}/stream")
async def stream_live_session(session_id: str) -> StreamingResponse:
    """SSE of the same events as GET .../events."""
    session = get_session(session_id)
    if session is None:
        return error_response("not_found", f"unknown session_id {session_id}", status_code=404)

    async def gen():
        after = 0
        while True:
            sess = get_session(session_id)
            if sess is None:
                break
            events = sess.get_events(after_seq=after)
            for ev in events:
                after = int(ev.get("seq", after))
                yield f"data: {json.dumps(ev)}\n\n"
            if sess.status in {"ended", "failed", "ingest_failed"}:
                yield f"data: {json.dumps({'event': 'session_end', 'status': sess.status})}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")
