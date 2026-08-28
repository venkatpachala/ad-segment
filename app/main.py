from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.errors import error_body, error_response
from app.api.routes import router
from app.models.schemas import DetectRequest
from app.pipeline.ingest import IngestError
from app.pipeline.run import run_detect

app = FastAPI(title="Ad Segment Detector", version="0.1.0")
app.include_router(router)


@app.exception_handler(IngestError)
async def _ingest_error(_request: Request, exc: IngestError) -> JSONResponse:
    from app.api.errors import ingest_response

    return ingest_response(exc)


@app.exception_handler(HTTPException)
async def _http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "unsupported"
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code, str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response("unsupported", str(exc.errors()), status_code=400)

_viewer = Path(__file__).resolve().parent.parent / "viewer"
if _viewer.exists():
    app.mount("/viewer", StaticFiles(directory=_viewer, html=True), name="viewer")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Detect ad segments in a video")
    p.add_argument("--url", default=None)
    p.add_argument("--local", dest="local_path", default=None)
    p.add_argument(
        "--kind",
        dest="kind_hint",
        default=None,
        choices=["vod", "short", "live"],
        help="Optional. Omit for auto (short vs vod from duration). Pass live only for streams.",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not args.url and not args.local_path:
        p.error("provide --url or --local")
    if args.kind_hint == "live":
        print(
            "Live detection is a separate surface: python -m app.live_main "
            "or POST /v1/live/sessions. `app.main --kind live` is not the live path.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        resp = run_detect(
            DetectRequest(url=args.url, local_path=args.local_path, kind_hint=None)
        )
    except IngestError as e:
        print(f"INGEST FAILED [{e.code}]\n{e}", file=sys.stderr)
        sys.exit(2)
    text = resp.model_dump_json(indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    _cli()