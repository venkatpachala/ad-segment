"""In-memory live session store.

A live session is a background tick-loop thread consuming either:
  - a local mp4 (fake-live: ticks through the file, events as they happen)
  - a growing .ts / live URL (real-live: now_s = duration of media actually buffered)

Events are pushed on every tick via callback — never after the stream ends.
POST /v1/live/sessions/{id}/stop is Ctrl+C.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.models.schemas import DetectResponse, utc_now
from app.pipeline.ingest import IngestError
from app.pipeline.live import TICK_S, WINDOW_S

_lock = threading.Lock()
_sessions: dict[str, "LiveSession"] = {}


LiveSessionState = str  # starting | running | stalled | ended | failed | ingest_failed


@dataclass
class LiveSession:
    session_id: str
    url: str | None
    local_path: str | None
    status: LiveSessionState
    created_at: str
    tick_s: float
    window_s: float
    use_llm: bool
    events: list[dict] = field(default_factory=list)
    events_lock: threading.Lock = field(default_factory=threading.Lock)
    result: DetectResponse | None = None
    error: str | None = None
    error_code: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _next_seq: int = 0

    def to_summary(self) -> dict:
        with self.events_lock:
            last_event = self.events[-1] if self.events else {}
        return {
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at,
            "url": self.url,
            "last_now_s": last_event.get("now_s", 0.0),
            "open_count": len(last_event.get("open", [])),
            "committed_count": len(self.result.segments) if self.result else 0,
            "error": self.error,
            "error_code": self.error_code,
        }

    def push_event(self, ev: dict) -> None:
        with self.events_lock:
            if "seq" not in ev:
                self._next_seq += 1
                ev = {"seq": self._next_seq, **ev}
            else:
                self._next_seq = max(self._next_seq, int(ev["seq"]))
            self.events.append(ev)
            if len(self.events) > 2000:
                self.events = self.events[-2000:]
        if ev.get("stalled") and self.status == "running":
            self.status = "stalled"
        elif self.status == "stalled" and not ev.get("stalled"):
            self.status = "running"

    def get_events(self, after_seq: int = 0) -> list[dict]:
        with self.events_lock:
            return [e for e in self.events if int(e.get("seq", 0)) > after_seq]


def reset_sessions() -> None:
    with _lock:
        for s in _sessions.values():
            s.stop_event.set()
        _sessions.clear()


def create_session(
    url: str | None = None,
    local_path: str | None = None,
    tick_s: float = TICK_S,
    window_s: float = WINDOW_S,
    use_llm: bool = False,
    hls_url: str | None = None,
    growing: bool | None = None,
) -> LiveSession:
    """Create and start a live detection session. Returns immediately (status=running)."""
    session_id = uuid4().hex[:12]
    url = url or hls_url
    session = LiveSession(
        session_id=session_id,
        url=url,
        local_path=local_path,
        status="running",
        created_at=utc_now(),
        tick_s=tick_s,
        window_s=window_s,
        use_llm=use_llm,
    )
    session._growing = growing  # type: ignore[attr-defined]
    with _lock:
        _sessions[session_id] = session

    t = threading.Thread(
        target=_run_session,
        args=(session_id,),
        name=f"live-{session_id}",
        daemon=True,
    )
    session._thread = t
    t.start()
    return session


def get_session(session_id: str) -> LiveSession | None:
    with _lock:
        return _sessions.get(session_id)


def list_sessions() -> list[dict]:
    with _lock:
        return [s.to_summary() for s in _sessions.values()]


def stop_session(session_id: str, join_s: float = 45.0) -> LiveSession | None:
    """Ctrl+C: flush open candidates and emit the final DetectResponse."""
    session = get_session(session_id)
    if session is None:
        return None
    session.stop_event.set()
    if session._thread is not None and session._thread.is_alive():
        session._thread.join(timeout=join_s)
    return session


def _run_session(session_id: str) -> None:
    """Background worker: runs the live tick loop, pushes events as they happen."""
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        return

    session.status = "running"
    try:
        from app.live_main import run_live_stream

        growing = getattr(session, "_growing", None)

        def on_event(ev: dict) -> None:
            session.push_event(ev)

        result = run_live_stream(
            local_path=session.local_path,
            url=session.url,
            tick_s=session.tick_s,
            window_s=session.window_s,
            use_llm=session.use_llm,
            on_event=on_event,
            stop_event=session.stop_event,
            growing=growing,
        )
        session.result = result
        session.status = "ended"

    except IngestError as exc:
        session.error = str(exc)
        session.error_code = exc.code
        session.status = "ingest_failed"
        session.push_event(
            {
                "now_s": 0.0,
                "open": [],
                "emitted": [],
                "event": "ingest_failed",
                "code": exc.code,
            }
        )
    except Exception as exc:
        session.error = str(exc)
        session.error_code = "pipeline_failed"
        session.status = "failed"
        session.push_event(
            {
                "now_s": 0.0,
                "open": [],
                "emitted": [],
                "event": "failed",
                "code": "pipeline_failed",
            }
        )
