"""Live news ad detection CLI — zero-lookahead tick loop.

Usage
-----
    uv run python -m app.live_main \\
        --local data/cache/s0LLVQeMmtU.mp4 \\
        --out   data/live.jsonl

Per-tick budget (target ≤ 600 ms on 720p):
  4 seeks + extract  50–80 ms
  ROI crop (8 crops) 10 ms
  MAE skip check      5 ms
  OCR ≤4 crops       200–400 ms  (0 when L-bar static: MAE skip fires)
  scene cut check     5–200 ms   (Tesseract only on cut)
  captions delta      1 ms
  LiveState.tick()   20 ms
  jsonl write         2 ms
  ─────────────────────────────
  Total static:     ~280 ms
  Total with new:   ~500 ms
  Total worst-case: ~800 ms

Tick overrun policy: if wall > 1.8 s in a tick, skip optional Whisper and
scene-cut full-frame OCR for that tick.

HLS note: replace `--local` with `--hls-url` to attach to a real stream.  The
tick loop is identical; only the buffer source changes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from app.config import RUNS_DIR
from app.models.schemas import DetectResponse, Source, Stats, utc_now
from app.pipeline.asr import CaptionLine, load_transcript
from app.pipeline.ingest import IngestError, ingest, probe_duration
from app.pipeline.live import (
    LIVE_PERSIST_S,
    LiveState,
    MIN_S,
    SILENCE_S,
    TICK_S,
    WINDOW_S,
)
from app.pipeline.live_proposers import (
    PersistHistory,
    SceneCutState,
    intent_proposer_live,
    persist_proposer,
    scene_cut_proposer,
)
from app.pipeline.live_streamer import LiveStreamIngest
from app.pipeline.roi import (
    LIVE_ROIS,
    ROI_BR,
    ROI_TR,
    RoiTickState,
    sense_all_rois,
    ocr_crop,
)
from app.pipeline.types import Candidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tick timestamps: sample this many frames before now (seconds back)
_FRAME_OFFSETS = [1.5, 1.0, 0.5, 0.0]

# Tick wall overrun threshold; above this, drop optional expensive work
_OVERRUN_S = 1.8

# Optional Whisper every N seconds (only if OPENAI_API_KEY set)
_WHISPER_INTERVAL_S = 12.0


# ---------------------------------------------------------------------------
# Frame seek helper (reuses a single cv2.VideoCapture per session)
# ---------------------------------------------------------------------------

class VideoSeeker:
    """Thin wrapper around cv2.VideoCapture that seeks without walking frames."""

    def __init__(self, path: str, is_live: bool = False) -> None:
        self.path = path
        self.is_live = is_live
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened() and not is_live:
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_f = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_s: float = total_f / self.fps if self.fps > 0 and total_f > 0 else 0.0

    def refresh(self, new_duration_s: float | None = None) -> None:
        """Refresh capture handle when reading a growing live stream buffer."""
        if new_duration_s is not None and new_duration_s > self.duration_s:
            self.duration_s = new_duration_s
        if not self._cap.isOpened() and Path(self.path).exists():
            self._cap = cv2.VideoCapture(self.path)
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    def read_at(self, ts_s: float) -> np.ndarray | None:
        """Seek to ts_s and return a BGR frame, or None on failure."""
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.path)
        self._cap.set(cv2.CAP_PROP_POS_MSEC, ts_s * 1000.0)
        ok, frame = self._cap.read()
        if not ok:
            # Fallback: frame index
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts_s * self.fps))
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()


# ---------------------------------------------------------------------------
# Per-tick sensor: seek 4 frames, sense ROIs, propose candidates
# ---------------------------------------------------------------------------

def _seek_tick_frames(
    seeker: VideoSeeker,
    now_s: float,
    offsets: list[float] = _FRAME_OFFSETS,
) -> list[tuple[float, np.ndarray]]:
    """Seek and read frames at [now - offset for offset in offsets].

    Returns a list of (timestamp_s, bgr_frame) for successfully read frames.
    """
    result: list[tuple[float, np.ndarray]] = []
    for off in offsets:
        ts = max(0.0, now_s - off)
        frame = seeker.read_at(ts)
        if frame is not None:
            result.append((ts, frame))
    return result


def _run_tick(
    seeker: VideoSeeker,
    now_s: float,
    roi_states: dict[str, RoiTickState],
    persist_hist: PersistHistory,
    scene_state: SceneCutState,
    cap_lines: list[CaptionLine],
    live_state: LiveState,
    executor: ThreadPoolExecutor,
    last_whisper_s: float,
    overrun_last_tick: bool,
) -> tuple[list[Candidate], list[Candidate], list[Candidate], int, float, bool]:
    """Execute one tick.  Returns (persist_cands, break_cands, intent_cands, ocr_calls, new_last_whisper_s, cut_this_tick)."""

    tick_t0 = time.perf_counter()

    # 1. Seek 4 frames (50–80 ms on cached mp4)
    frames_ts = _seek_tick_frames(seeker, now_s)
    frames_bgr = [f for _, f in frames_ts]
    ocr_calls = 0
    cut_this_tick = False

    # 2. Sense ROIs (MAE skip + ≤2 Tesseract calls if new content)
    roi_results = sense_all_rois(frames_bgr, roi_states)
    ocr_calls += sum(1 for r in roi_results if r.ocr_called)

    # 3. P1 — persist proposer (no Tesseract, just dict lookup)
    persist_cands = persist_proposer(roi_results, now_s, persist_hist)

    # 4. P2 — scene-cut break proposer (Tesseract only on cut)
    break_cand: Candidate | None = None
    if frames_bgr and not overrun_last_tick:
        latest_frame = frames_bgr[-1]
        break_cand, cut_this_tick = scene_cut_proposer(latest_frame, scene_state, now_s)
        if break_cand is not None:
            ocr_calls += 1  # one full-frame OCR fired
    break_cands = [break_cand] if break_cand else []

    # 5. P3 — intent only if new speech cues exist in [now-WINDOW, now]
    intent_cands: list[Candidate] = []
    if cap_lines and not overrun_last_tick:
        window_caps = [
            ln for ln in cap_lines if ln.start_s <= now_s and ln.end_s >= now_s - WINDOW_S
        ]
        if window_caps:
            intent_cands = intent_proposer_live(cap_lines, now_s, WINDOW_S)

    # 6. Whisper default OFF on news (LIVE_WHISPER=1 to enable, still not every tick)
    new_last_whisper_s = last_whisper_s
    if (
        os.getenv("LIVE_WHISPER") == "1"
        and os.getenv("OPENAI_API_KEY")
        and not overrun_last_tick
        and now_s - last_whisper_s >= _WHISPER_INTERVAL_S
        and frames_bgr
    ):
        new_last_whisper_s = now_s

    tick_wall = time.perf_counter() - tick_t0
    overrun = tick_wall > _OVERRUN_S

    return persist_cands, break_cands, intent_cands, ocr_calls, new_last_whisper_s, cut_this_tick


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def _open_summary(live_state: LiveState) -> list[dict]:
    """Public `open` rows: start_s, roi, text (+ end_s so the overlay can grow)."""
    rows: list[dict] = []
    for oc in live_state.open_cands:
        text = max(oc.texts, key=len) if oc.texts else ""
        rows.append(
            {
                "start_s": round(oc.start_s, 2),
                "end_s": round(min(oc.last_evidence_s, live_state.now_s), 2),
                "roi": oc.roi,
                "text": text[:80],
            }
        )
    if rows:
        return rows
    # Fall back to provisional overlays (same shape, growing end_s).
    return [
        {
            "start_s": p.get("start_s", 0.0),
            "end_s": p.get("end_s", 0.0),
            "roi": p.get("roi", ""),
            "text": p.get("text", ""),
        }
        for p in live_state.provisional_overlays()
    ]


def _emitted_summary(segments) -> list[dict]:
    out: list[dict] = []
    for s in segments:
        out.append(
            {
                "start_s": s.start_s,
                "end_s": s.end_s,
                "ad_type": s.ad_type,
                "presentation": s.presentation,
                "brand": s.brand,
                "id": s.id,
            }
        )
    return out


def run_live_stream(
    local_path: str | None = None,
    url: str | None = None,
    tick_s: float = TICK_S,
    window_s: float = WINDOW_S,
    out_jsonl: str | None = None,
    out_json: str | None = None,
    use_llm: bool = False,
    max_duration_s: float | None = None,
    on_event=None,
    stop_event: threading.Event | None = None,
    growing: bool | None = None,
) -> DetectResponse:
    """Run live stream ad detection on a live URL or simulated local file.

    Horizon = 0: ``now_s`` is duration of media actually in the buffer.
    Events are emitted on every tick via *on_event* (do not wait for EOF).
    """
    if not local_path and not url:
        raise ValueError("Either local_path or url must be provided")

    t_wall_0 = time.perf_counter()
    run_id = f"live_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    is_live = bool(url)
    source_label = url if is_live else local_path

    print(f"\n{'=' * 60}")
    print(f"  LIVE STREAM AD DETECTOR (horizon = 0 / on-the-spot)")
    print(f"  Mode   : {'TRUE LIVE STREAM (Real-Time URL)' if is_live else 'LOCAL SIMULATION'}")
    print(f"  Source : {source_label}")
    print(f"  Tick   : {tick_s}s  |  Window : {window_s}s  |  LLM: {use_llm}")
    if is_live:
        print(f"  Controls: Press Ctrl+C at any time to finish & summarize")
    print(f"{'=' * 60}\n")

    ingest_worker: LiveStreamIngest | None = None
    stream_path: Path
    transcript_lang = ""
    cap_lines: list[CaptionLine] = []

    local_src = Path(local_path) if local_path else None
    if growing is None:
        growing = bool(url) or (
            local_src is not None and local_src.suffix.lower() in {".ts", ".m2ts", ".m3u8"}
        )

    if url:
        stream_path = run_dir / "live_buf.ts"
        ingest_worker = LiveStreamIngest(url, stream_path)
        ingest_worker.start()
        growing = True

        print("⏳ Waiting for initial live stream buffer to arrive...")
        wait_start = time.time()
        while ingest_worker.get_available_duration_s() < tick_s:
            if stop_event is not None and stop_event.is_set():
                break
            err = ingest_worker.ingest_error()
            if err is not None:
                ingest_worker.stop()
                raise err
            if not ingest_worker.is_active() and ingest_worker.get_available_duration_s() < tick_s:
                err = ingest_worker.ingest_error() or IngestError(
                    "youtube_bot_check",
                    "Live ingest exited before media arrived. Upload a recording to POST /v1/live/sessions.",
                )
                ingest_worker.stop()
                raise err
            if time.time() - wait_start > 30.0:
                err = ingest_worker.ingest_error()
                if err is not None:
                    ingest_worker.stop()
                    raise err
                print("WARN: buffer wait timeout (30s); proceeding with available stream")
                break
            time.sleep(0.4)

        total_dur = max_duration_s or 86400.0
        video_path = stream_path
    else:
        assert local_path is not None
        src = Path(local_path).expanduser().resolve()
        if not src.exists() and not growing:
            raise FileNotFoundError(src)
        if growing:
            # Tick on the file as it grows — do not copy/cache then run VOD.
            video_path = src
            total_dur = max_duration_s or 86400.0
            print(f"INFO: growing buffer {src}")
        else:
            # Fake-live: tick through the file in 2s media steps. Do not copy 100MB+
            # and do not Whisper the whole programme at session start.
            video_path = src
            try:
                total_dur = probe_duration(src)
            except Exception:
                total_dur = 0.0
            if max_duration_s:
                total_dur = min(total_dur, max_duration_s) if total_dur else max_duration_s
            if os.getenv("LIVE_CAPTIONS") == "1":
                bundle = load_transcript(None, src, run_dir)
                cap_lines = bundle.lines
                transcript_lang = bundle.language
            print(f"INFO: fake-live duration={total_dur:.1f}s  path={src}")
            print(f"INFO: caption_cues={len(cap_lines)} lang={transcript_lang or '-'} whisper=off")

    is_live = bool(growing)
    seeker = VideoSeeker(str(video_path), is_live=is_live)
    if not growing and seeker.duration_s > 0 and abs(seeker.duration_s - total_dur) < 1.0:
        total_dur = seeker.duration_s

    # Per-session state
    live_state = LiveState(transcript_lang=transcript_lang)
    roi_states: dict[str, RoiTickState] = {}
    persist_hist = PersistHistory()
    scene_state = SceneCutState()
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live_ocr")

    jsonl_lines: list[str] = []
    jsonl_path = Path(out_jsonl) if out_jsonl else None
    jsonl_fh = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None

    tick_wall_ms_list: list[float] = []
    ocr_calls_per_tick: list[int] = []
    last_whisper_s = 0.0
    overrun_last_tick = False

    seq = 0
    last_ticked_now = 0.0
    stalled_emitted = False
    stopped = False

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    def _emit(event: dict) -> None:
        nonlocal seq
        seq += 1
        event = {"seq": seq, **event}
        # Horizon check: nothing in this event may live in the future.
        now = float(event.get("now_s") or 0.0)
        for item in list(event.get("open") or []) + list(event.get("emitted") or []):
            if isinstance(item, dict):
                if float(item.get("end_s") or 0.0) > now + 0.05:
                    item["end_s"] = round(now, 2)
        line = json.dumps(event)
        jsonl_lines.append(line)
        if jsonl_fh:
            jsonl_fh.write(line + "\n")
            jsonl_fh.flush()
        if on_event is not None:
            try:
                on_event(event)
            except Exception as exc:
                print(f"WARN: on_event failed: {exc}")

    def _available_s() -> float:
        if ingest_worker is not None:
            return ingest_worker.get_available_duration_s()
        if growing:
            try:
                return probe_duration(Path(video_path))
            except Exception:
                return last_ticked_now
        return total_dur

    try:
        while True:
            if _stopped():
                stopped = True
                print(f"\n🛑 [LIVE] stop requested at now={last_ticked_now:.1f}s")
                break

            tick_t0 = time.perf_counter()

            if growing:
                avail = _available_s()
                seeker.refresh(avail)
                target = last_ticked_now + tick_s if last_ticked_now > 0 else tick_s
                if max_duration_s is not None:
                    target = min(target, max_duration_s)
                if ingest_worker is not None:
                    err = ingest_worker.ingest_error()
                    if err is not None and avail < tick_s:
                        raise err
                    if avail < target:
                        if ingest_worker.stalled():
                            if not stalled_emitted:
                                _emit(
                                    {
                                        "now_s": round(last_ticked_now, 2),
                                        "open": _open_summary(live_state),
                                        "emitted": [],
                                        "stalled": True,
                                    }
                                )
                                stalled_emitted = True
                                print(f"⚠️  [STALLED] buffer not growing at now={last_ticked_now:.1f}s")
                            if not ingest_worker.is_active():
                                print(f"\nINFO: Live stream ended at {avail:.1f}s")
                                break
                        elif not ingest_worker.is_active() and avail <= last_ticked_now:
                            print(f"\nINFO: Live stream ended at {avail:.1f}s")
                            break
                        time.sleep(0.3)
                        continue
                    stalled_emitted = False
                    curr_now = target
                else:
                    if avail < target:
                        # Growing local .ts: wait for more media; never invent ticks.
                        if last_ticked_now > 0 and avail > 0 and avail <= last_ticked_now + 0.5:
                            # File stopped growing near last tick → EOF.
                            size_now = Path(video_path).stat().st_size if Path(video_path).exists() else 0
                            time.sleep(0.4)
                            size_later = Path(video_path).stat().st_size if Path(video_path).exists() else 0
                            if size_later <= size_now:
                                break
                        time.sleep(0.2)
                        continue
                    curr_now = target
            else:
                curr_now = last_ticked_now + tick_s if last_ticked_now > 0 else tick_s
                if curr_now > total_dur + 1e-6:
                    break
                curr_now = min(curr_now, total_dur)

            if curr_now < tick_s * 0.5:
                time.sleep(0.2)
                continue

            persist_cands, break_cands, intent_cands, ocr_calls, last_whisper_s, cut = _run_tick(
                seeker=seeker,
                now_s=curr_now,
                roi_states=roi_states,
                persist_hist=persist_hist,
                scene_state=scene_state,
                cap_lines=cap_lines,
                live_state=live_state,
                executor=executor,
                last_whisper_s=last_whisper_s,
                overrun_last_tick=overrun_last_tick,
            )

            all_cands = persist_cands + break_cands + intent_cands
            emitted = live_state.tick(
                now_s=curr_now,
                window_cands=all_cands,
                cap_lines=cap_lines,
                duration_s=curr_now,
                use_llm=use_llm,
            )
            live_state.ocr_calls_total += ocr_calls
            provisional = live_state.provisional_overlays()

            tick_wall_ms = (time.perf_counter() - tick_t0) * 1000.0
            tick_wall_ms_list.append(tick_wall_ms)
            ocr_calls_per_tick.append(ocr_calls)
            overrun_last_tick = tick_wall_ms > _OVERRUN_S * 1000
            last_ticked_now = curr_now

            growing_emit = [
                {
                    "start_s": p.get("start_s"),
                    "end_s": p.get("end_s"),
                    "ad_type": "other",
                    "presentation": "overlay",
                    "status": "provisional",
                    "roi": p.get("roi"),
                    "text": p.get("text"),
                }
                for p in provisional
            ]
            event = {
                "now_s": round(curr_now, 2),
                "tick_ms": round(tick_wall_ms, 1),
                "ocr_calls": ocr_calls,
                "open": _open_summary(live_state),
                "emitted": growing_emit + _emitted_summary(emitted),
                "provisional": provisional,
                "_cut": cut,
            }
            _emit(event)

            if emitted:
                for seg in emitted:
                    print(
                        f"🔴 [LIVE EMIT @ now={curr_now:06.1f}s] {seg.id}  "
                        f"[{seg.start_s:.1f}s – {seg.end_s:.1f}s]  "
                        f"type={seg.ad_type}  brand='{seg.brand}'"
                    )
            if provisional:
                for p in provisional:
                    print(
                        f"🟡 [PROVISIONAL @ now={curr_now:06.1f}s] "
                        f"{p.get('roi','').upper()} bug: {p.get('text','')} "
                        f"[{p.get('start_s')}s - {p.get('end_s')}s]"
                    )
            if cut:
                print(f"   ↗ cut detected at now={curr_now:.1f}s | tick={tick_wall_ms:.0f}ms | ocr={ocr_calls}")
            elif ocr_calls > 0:
                print(f"   · now={curr_now:06.1f}s | tick={tick_wall_ms:.0f}ms | ocr={ocr_calls}")

            if not growing and curr_now >= total_dur:
                break
            if max_duration_s is not None and curr_now >= max_duration_s:
                break

            if growing:
                elapsed_s = time.perf_counter() - tick_t0
                sleep_s = max(0.0, tick_s - elapsed_s)
                if sleep_s > 0 and not _stopped():
                    time.sleep(sleep_s)

    except KeyboardInterrupt:
        stopped = True
        print(f"\n\n🛑 [LIVE] Interrupted by user (Ctrl+C). Flushing live state at now={last_ticked_now:.1f}s...")
    finally:
        if ingest_worker:
            ingest_worker.stop()
        if jsonl_fh:
            try:
                jsonl_fh.flush()
            except Exception:
                pass

    actual_end_s = last_ticked_now if last_ticked_now > 0 else (total_dur if not growing else 0.0)

    final_emitted = live_state.end(actual_end_s, cap_lines=cap_lines, use_llm=use_llm)
    for seg in final_emitted:
        print(
            f"🏁 [FLUSH @ now={actual_end_s:06.1f}s] {seg.id}  "
            f"[{seg.start_s:.1f}s – {seg.end_s:.1f}s]  "
            f"type={seg.ad_type}  brand='{seg.brand}'"
        )
    flush_event = {
        "now_s": round(actual_end_s, 2),
        "open": [],
        "emitted": _emitted_summary(final_emitted),
        "event": "session_end",
        "final_emitted": [s.model_dump() for s in final_emitted],
    }
    _emit(flush_event)
    if jsonl_fh:
        jsonl_fh.close()

    seeker.close()
    executor.shutdown(wait=False)
    total_wall = time.perf_counter() - t_wall_0

    tick_p95_ms = (
        statistics.quantiles(tick_wall_ms_list, n=20)[18]
        if len(tick_wall_ms_list) >= 20
        else (max(tick_wall_ms_list) if tick_wall_ms_list else 0.0)
    )
    ocr_avg = statistics.mean(ocr_calls_per_tick) if ocr_calls_per_tick else 0.0

    print(f"\n{'─' * 60}")
    print(f"  Session Mode   : {'LIVE STREAM' if growing else 'LOCAL SIMULATION'}")
    print(f"  Ticks          : {live_state.ticks_count}")
    print(f"  Segments       : {len(live_state.committed_segments)}")
    print(f"  tick_p95_ms    : {tick_p95_ms:.0f}ms")
    print(f"  ocr_avg/tick   : {ocr_avg:.2f}")
    print(f"  OCR total      : {live_state.ocr_calls_total}")
    print(f"  Wall time      : {total_wall:.1f}s")
    print(f"  Lookahead      : False")
    print(f"{'─' * 60}\n")

    resp = DetectResponse(
        source=Source(
            url=url or local_path or "",
            platform="youtube" if (url and "youtube" in url) else "file",
            kind="live",
            duration_s=round(actual_end_s, 2),
            processed_at=utc_now(),
            language=transcript_lang,
        ),
        segments=live_state.committed_segments,
        stats=Stats(
            wall_clock_s=round(total_wall, 2),
            estimated_cost_usd=0.0,
            frames_sampled=live_state.ticks_count * len(_FRAME_OFFSETS),
            frames_ocrd=live_state.ocr_calls_total,
            ocr_wall_s=round(total_wall * (ocr_avg / max(1, len(_FRAME_OFFSETS) * 2)), 2),
            model_calls=live_state.commits_count,
        ),
    )

    if out_json:
        Path(out_json).write_text(json.dumps(resp.model_dump(), indent=2), encoding="utf-8")
        print(f"Saved final DetectResponse → {out_json}")

    stats_line = {
        "event": "session_stats",
        "ticks": live_state.ticks_count,
        "ocr_calls_per_tick_avg": round(ocr_avg, 3),
        "tick_p95_ms": round(tick_p95_ms, 1),
        "lookahead": False,
        "wall_s": round(total_wall, 2),
        "stopped": stopped,
    }
    if jsonl_path:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats_line) + "\n")

    return resp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live news ad detection — real-time tick loop on live URLs or local files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--url", default=None,
        help="YouTube Live URL or HLS .m3u8 stream URL (e.g. https://www.youtube.com/live/VIDEO_ID)",
    )
    p.add_argument(
        "--local", default=None,
        help="Local video path to simulate live stream (e.g. data/cache/s0LLVQeMmtU.mp4)",
    )
    p.add_argument(
        "--out", default="data/live.jsonl",
        help="Path to write per-tick JSONL event log (default: data/live.jsonl)",
    )
    p.add_argument(
        "--out-json", default=None,
        help="Optional path to write final DetectResponse JSON",
    )
    p.add_argument(
        "--tick", type=float, default=TICK_S,
        help=f"Tick interval in seconds (default: {TICK_S})",
    )
    p.add_argument(
        "--window", type=float, default=WINDOW_S,
        help=f"Lookback window in seconds (default: {WINDOW_S})",
    )
    p.add_argument(
        "--max-duration", type=float, default=None,
        help="Optional maximum seconds to process before closing session",
    )
    p.add_argument(
        "--llm", action="store_true",
        help="Enable LLM judge on commit (adds ~1.5s latency per commit; off by default)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if not args.local and not args.url:
        print("ERROR: Please provide either --url <live_stream_url> or --local <file_path>")
        exit(1)

    run_live_stream(
        local_path=args.local,
        url=args.url,
        tick_s=args.tick,
        window_s=args.window,
        out_jsonl=args.out,
        out_json=args.out_json,
        use_llm=args.llm,
        max_duration_s=args.max_duration,
    )
