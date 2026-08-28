"""Live stream ingest worker for real-time YouTube Live / HLS / DASH streams.

Spawns a background process (yt-dlp / ffmpeg) that continuously writes
incoming video segments to an appendable MPEG-TS (.ts) ring buffer.
The live tick loop reads from this buffer in real time — never cache-then-VOD.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from app.config import HLS_STALL_S
from app.pipeline.ingest import (
    IngestError,
    _bot_check_message,
    _cookie_args,
    _ytdlp_exe,
    classify_ytdlp_error,
    probe_duration,
)


class LiveStreamIngest:
    """Manages background streaming of a live URL into a growing .ts buffer."""

    def __init__(self, url: str, buffer_path: Path) -> None:
        self.url = url
        self.buffer_path = buffer_path
        self.process: subprocess.Popen | None = None
        self._stop_event = threading.Event()
        self._last_duration_s = 0.0
        self._last_probe_time = 0.0
        self._last_size = 0
        self._last_growth_wall = 0.0
        self._started_at = 0.0
        self._stderr: list[str] = []
        self._drain_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background live stream download process."""
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        if self.buffer_path.exists():
            try:
                self.buffer_path.unlink()
            except OSError:
                pass

        cmd = [
            _ytdlp_exe(),
            "--no-playlist",
            "-f",
            "best[height<=720]/bestvideo[height<=720]+bestaudio/best",
            "--no-part",
            "--retries",
            "infinite",
            "--fragment-retries",
            "infinite",
            "-o",
            str(self.buffer_path),
        ]
        cookie_args = _cookie_args()
        if cookie_args:
            cmd.extend(cookie_args)
        cmd.append(self.url)

        print(f"INFO: starting live stream ingest worker for {self.url} -> {self.buffer_path.name}")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        self._started_at = time.time()
        self._last_growth_wall = self._started_at
        self._drain_thread = threading.Thread(
            target=self._drain_stderr, name="live-ingest-stderr", daemon=True
        )
        self._drain_thread.start()

    def _drain_stderr(self) -> None:
        proc = self.process
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr.append(line)
                if len(self._stderr) > 400:
                    self._stderr = self._stderr[-200:]
        except Exception:
            pass

    def get_available_duration_s(self) -> float:
        """Probe the current duration of the growing stream buffer. Never future."""
        now = time.time()
        if now - self._last_probe_time < 0.8 and self._last_duration_s > 0:
            self._note_growth()
            return self._last_duration_s

        if not self.buffer_path.exists() or self.buffer_path.stat().st_size < 50_000:
            return 0.0

        self._note_growth()
        try:
            dur = probe_duration(self.buffer_path)
            self._last_duration_s = dur
            self._last_probe_time = now
            return dur
        except Exception:
            return self._last_duration_s

    def _note_growth(self) -> None:
        try:
            size = self.buffer_path.stat().st_size if self.buffer_path.exists() else 0
        except OSError:
            size = 0
        if size > self._last_size:
            self._last_size = size
            self._last_growth_wall = time.time()

    def is_active(self) -> bool:
        """True if the ingest process is running and not terminated."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stalled(self, stall_s: float = HLS_STALL_S) -> bool:
        """True when the buffer has not grown for stall_s seconds."""
        if self._started_at <= 0:
            return False
        self._note_growth()
        return (time.time() - self._last_growth_wall) >= stall_s

    def ingest_error(self) -> IngestError | None:
        """If the worker died before producing media, map stderr to IngestError."""
        if self.is_active():
            return None
        if self.get_available_duration_s() > 1.0:
            return None
        output = "".join(self._stderr[-200:])
        if not output and self.process is not None and self.process.returncode == 0:
            return None
        code = classify_ytdlp_error(output) if output else "youtube_bot_check"
        if code == "youtube_bot_check":
            return IngestError(code, _bot_check_message())
        return IngestError(
            code,
            output.strip() or "Live ingest exited before any media arrived. Upload a recording.",
        )

    def stop(self) -> None:
        """Terminate the background ingest process cleanly."""
        self._stop_event.set()
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None
