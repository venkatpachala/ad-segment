"""Public API contract: two POSTs, orig-lang evidence, live ticks until stop."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import reset_jobs
from app.api.live_sessions import reset_sessions
from app.main import app
from app.models.schemas import DetectResponse, Evidence, Segment, Source, Stats, utc_now
from app.pipeline.live import TICK_S


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.api.routes.tesseract_available", lambda: True)
    reset_jobs()
    reset_sessions()
    with TestClient(app) as c:
        yield c
    reset_sessions()
    reset_jobs()


def _source(kind: str, duration_s: float, language: str = "te") -> Source:
    return Source(
        url="file://fixture",
        platform="file",
        kind=kind,  # type: ignore[arg-type]
        duration_s=duration_s,
        processed_at=utc_now(),
        language=language,
    )


def _overlay(duration_s: float, lang: str = "te") -> Segment:
    return Segment(
        id="seg_01",
        start_s=7.5,
        end_s=duration_s,
        ad_type="other",
        confidence=0.95,
        brand="Avis Vascular Center",
        description="Commercial overlay.",
        evidence=Evidence(
            frame_timestamps=[7.5, 8.0],
            transcript_span="",
            ocr_text="Avis Vascular Center 89297 21641",
            transcript_lang="",
            signals_used=["ocr"],
        ),
        presentation="overlay",
    )


def _preroll(lang: str = "en") -> Segment:
    return Segment(
        id="seg_01",
        start_s=0.0,
        end_s=6.5,
        ad_type="preroll",
        confidence=0.9,
        brand="Career247",
        description="Opening commercial.",
        evidence=Evidence(
            frame_timestamps=[0.5, 3.0],
            transcript_span="sponsored by",
            transcript_lang=lang,
            signals_used=["ocr", "asr"],
        ),
    )


def _self_promo(duration_s: float, lang: str = "en") -> Segment:
    return Segment(
        id="seg_02",
        start_s=max(0.0, duration_s - 18.0),
        end_s=duration_s,
        ad_type="self_promo",
        confidence=0.85,
        brand="",
        description="End-card self promo.",
        evidence=Evidence(
            transcript_span="join the program",
            transcript_lang=lang,
            signals_used=["asr"],
        ),
    )


def _stub_detect(monkeypatch, force_duration: float | None = None):
    """Patch the job worker's run_detect. Kind is inferred from duration, never from client."""

    def fake_run_detect(req, run_id=None, on_status=None):
        assert req.kind_hint is None, "client must not set kind"
        if on_status:
            on_status("detecting")
        dur = 8.0
        if force_duration is not None:
            dur = force_duration
        elif req.local_path:
            try:
                from app.pipeline.ingest import probe_duration

                dur = probe_duration(Path(req.local_path))
            except Exception:
                dur = 8.0
        kind = "short" if dur <= 180 else "vod"
        lang = "te" if kind == "short" else "en"
        segs = [_overlay(dur, lang)] if kind == "short" else [_preroll(lang), _self_promo(dur, lang)]
        return DetectResponse(
            source=_source(kind, dur, lang),
            segments=segs,
            stats=Stats(wall_clock_s=1.2, estimated_cost_usd=0.0, frames_sampled=10, frames_ocrd=4, model_calls=1),
        )

    monkeypatch.setattr("app.api.jobs.run_detect", fake_run_detect)
    return fake_run_detect


def _tiny_mp4(path: Path, duration_s: float) -> Path:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not on PATH")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=160x120:d={duration_s}:r=5",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return path


def _poll_job(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    body: dict = {}
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body.get("status") in {"done", "failed", "ingest_failed"}:
            return body
        time.sleep(0.1)
    return body


def test_detect_requires_url_or_file(client: TestClient):
    r = client.post("/v1/detect", json={"file": None})
    assert r.status_code == 400
    assert r.json()["code"] == "unsupported"
    assert "traceback" not in r.text.lower()


def test_detect_unsupported_host_no_traceback(client: TestClient, monkeypatch):
    _stub_detect(monkeypatch)

    def boom(req, run_id=None, on_status=None):
        from app.pipeline.ingest import IngestError

        raise IngestError("unsupported_host", "Only YouTube URLs or local files are supported.")

    monkeypatch.setattr("app.api.jobs.run_detect", boom)
    r = client.post("/v1/detect", json={"url": "https://example.com/watch?v=abc"})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "unsupported"
    assert "Traceback" not in r.text


def test_detect_youtube_bot_check_code(client: TestClient, monkeypatch):
    from app.pipeline.ingest import IngestError

    def boom(req, run_id=None, on_status=None):
        raise IngestError("youtube_bot_check", "bot check")

    monkeypatch.setattr("app.api.jobs.run_detect", boom)
    r = client.post("/v1/detect", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert r.status_code == 422
    assert r.json()["code"] == "youtube_bot_check"
    assert "traceback" not in r.text.lower()


def test_detect_rejects_live_url(client: TestClient):
    r = client.post(
        "/v1/detect",
        json={"url": "https://www.youtube.com/live/s0LLVQeMmtU", "file": None},
    )
    assert r.status_code == 422
    assert r.json()["code"] == "unsupported"
    assert "live/sessions" in r.json()["detail"].lower()


def test_detect_ignores_client_kind(client: TestClient, monkeypatch, tmp_path):
    _stub_detect(monkeypatch)
    clip = _tiny_mp4(tmp_path / "shortclip.mp4", 6)
    with clip.open("rb") as fh:
        r = client.post(
            "/v1/detect",
            files={"file": ("shortclip.mp4", fh, "video/mp4")},
            data={"kind": "vod", "kind_hint": "vod"},
        )
    assert r.status_code in {200, 202}
    if r.status_code == 202:
        body = _poll_job(client, r.json()["job_id"])
    else:
        body = r.json()
    assert body["status"] == "done"
    assert body["source"]["kind"] == "short"


# ---------------------------------------------------------------------------
# The three ship tests
# ---------------------------------------------------------------------------

def test_api_short_file_overlay(client: TestClient, monkeypatch, tmp_path):
    """1. short file → overlay times + orig-lang evidence."""
    _stub_detect(monkeypatch)
    clip = _tiny_mp4(tmp_path / "avis_short.mp4", 12)
    with clip.open("rb") as fh:
        r = client.post("/v1/detect", files={"file": ("avis_short.mp4", fh, "video/mp4")})
    assert r.status_code in {200, 202}
    body = r.json() if r.status_code == 200 else _poll_job(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert "job_id" in body
    assert body["source"]["kind"] == "short"
    assert body["source"]["language"]
    segs = body["segments"]
    assert segs
    overlay = next(s for s in segs if s.get("presentation") == "overlay")
    assert overlay["start_s"] >= 0
    assert overlay["end_s"] > overlay["start_s"]
    assert overlay["ad_type"] == "other"
    ev = overlay["evidence"]
    assert ev["signals_used"] == ["ocr"]
    assert ev.get("transcript_span", "") == ""
    assert ev.get("ocr_text")


def test_api_vod_file_preroll_or_self_promo(client: TestClient, monkeypatch, tmp_path):
    """2. vod file → preroll/self_promo (kind inferred from duration, not client)."""
    _stub_detect(monkeypatch, force_duration=240.0)
    clip = _tiny_mp4(tmp_path / "long_vod.mp4", 6)  # tiny bytes; stub forces 240s kind=vod
    with clip.open("rb") as fh:
        r = client.post("/v1/detect", files={"file": ("long_vod.mp4", fh, "video/mp4")})
    assert r.status_code in {200, 202}
    body = r.json() if r.status_code == 200 else _poll_job(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert body["source"]["kind"] == "vod"
    types = {s["ad_type"] for s in body["segments"]}
    assert types & {"preroll", "self_promo"}
    for s in body["segments"]:
        assert "transcript_lang" in s["evidence"]
        assert s["evidence"]["transcript_lang"] != "hi" or body["source"]["language"] == "hi"


def test_api_live_ticks_on_local_file_no_future(client: TestClient, monkeypatch, tmp_path):
    """3. live ticks on local ts/mp4 → event at t=10 has no t=400 ads; ticks while running."""

    def fake_live(
        local_path=None,
        url=None,
        tick_s=TICK_S,
        window_s=16.0,
        out_jsonl=None,
        out_json=None,
        use_llm=False,
        max_duration_s=None,
        on_event=None,
        stop_event=None,
        growing=None,
    ):
        saw_stop = False
        for i, now in enumerate([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], start=1):
            if stop_event is not None and stop_event.is_set():
                saw_stop = True
                break
            if on_event:
                on_event(
                    {
                        "seq": i,
                        "now_s": now,
                        "open": [{"start_s": 2.0, "end_s": now, "roi": "tr", "text": "G-Mart"}],
                        "emitted": [],
                    }
                )
            time.sleep(0.04)
        end = 14.0 if not saw_stop else 10.0
        if on_event:
            on_event(
                {
                    "seq": 99,
                    "now_s": end,
                    "open": [],
                    "emitted": [
                        {
                            "start_s": 2.0,
                            "end_s": end,
                            "ad_type": "other",
                            "presentation": "overlay",
                        }
                    ],
                    "event": "session_end",
                }
            )
        return DetectResponse(
            source=_source("live", end, "ml"),
            segments=[
                Segment(
                    id="seg_01",
                    start_s=2.0,
                    end_s=end,
                    ad_type="other",
                    confidence=0.9,
                    brand="G-Mart",
                    description="overlay",
                    evidence=Evidence(
                        transcript_lang="ml",
                        signals_used=["ocr"],
                    ),
                    presentation="overlay",
                )
            ],
            stats=Stats(wall_clock_s=0.5, estimated_cost_usd=0.0, frames_sampled=8, frames_ocrd=2, model_calls=0),
        )

    monkeypatch.setattr("app.live_main.run_live_stream", fake_live)
    clip = _tiny_mp4(tmp_path / "live_clip.mp4", 8)
    with clip.open("rb") as fh:
        r = client.post("/v1/live/sessions", files={"file": ("live_clip.mp4", fh, "video/mp4")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"
    sid = body["session_id"]

    after = 0
    saw_running = False
    ev10 = None
    deadline = time.time() + 8.0
    while time.time() < deadline:
        ev = client.get(f"/v1/live/sessions/{sid}/events", params={"after": after})
        assert ev.status_code == 200
        payload = ev.json()
        if payload["events"] and payload["status"] == "running":
            saw_running = True
        for e in payload["events"]:
            if e.get("now_s") == 10.0:
                ev10 = e
            for item in list(e.get("open") or []) + list(e.get("emitted") or []):
                if e.get("now_s", 0) <= 10.0:
                    assert float(item.get("end_s") or 0) <= 10.0 + 0.05
                    assert float(item.get("start_s") or 0) < 50
                    assert float(item.get("end_s") or 0) < 400
                    assert float(item.get("start_s") or 0) < 400
        after = payload["next_after"]
        if payload["status"] in {"ended", "failed", "ingest_failed"}:
            break
        time.sleep(0.02)

    assert saw_running or ev10 is not None
    if ev10 is None:
        # Fetch everything in case we missed the running window.
        all_ev = client.get(f"/v1/live/sessions/{sid}/events", params={"after": 0}).json()["events"]
        ev10 = next((e for e in all_ev if e.get("now_s") == 10.0), None)
    assert ev10 is not None, "expected a tick at now_s=10"
    for item in list(ev10.get("open") or []) + list(ev10.get("emitted") or []):
        assert float(item.get("end_s") or 0) <= 10.05
        assert float(item.get("start_s") or 0) < 400

    stopped = client.post(f"/v1/live/sessions/{sid}/stop")
    assert stopped.status_code == 200
    final = stopped.json()
    assert final["status"] in {"ended", "running"}
    assert "segments" in final or final["status"] == "ended"


def test_live_json_url_or_file_shape(client: TestClient, monkeypatch):
    def fake_live(**kwargs):
        time.sleep(0.2)
        return DetectResponse(
            source=_source("live", 2.0, ""),
            segments=[],
            stats=Stats(wall_clock_s=0.2, estimated_cost_usd=0.0, frames_sampled=1, frames_ocrd=0, model_calls=0),
        )

    monkeypatch.setattr("app.live_main.run_live_stream", fake_live)
    r = client.post("/v1/live/sessions", json={"url": "https://www.youtube.com/live/fakeid12ab", "file": None})
    assert r.status_code == 200
    assert set(r.json()) >= {"session_id", "status"}
    assert r.json()["status"] == "running"
    client.post(f"/v1/live/sessions/{r.json()['session_id']}/stop")
