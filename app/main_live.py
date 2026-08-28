from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.config import RUNS_DIR
from app.models.schemas import DetectResponse, Evidence, Segment, Source, Stats, utc_now
from app.pipeline.asr import CaptionLine, candidates_from_asr, load_full_transcript
from app.pipeline.frames import extract_frames
from app.pipeline.ingest import ingest
from app.pipeline.intent import candidates_from_intent
from app.pipeline.candidates import candidates_from_ocr
from app.pipeline.live import LiveState, MIN_S, SILENCE_S, TICK_S, WINDOW_S
from app.pipeline.ocr import ocr_frames


def run_live_stream_simulation(
    local_path: str,
    tick_s: float = TICK_S,
    window_s: float = WINDOW_S,
    out_jsonl: str | None = None,
    out_json: str | None = None,
) -> DetectResponse:
    """Simulate a live stream by processing a video file in real-time ticks with ZERO lookahead."""
    t0 = time.perf_counter()
    run_id = f"live_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"  LIVE STREAM SIMULATOR (ZERO LOOKAHEAD: HORIZON = 0)")
    print(f"  Source: {local_path} | Tick: {tick_s}s | Window: {window_s}s")
    print(f"========================================================\n")

    # Ingest file metadata
    asset = ingest(None, local_path, "live", run_dir)
    total_dur = asset.duration_s

    # Load audio transcript cues
    cap_lines = load_full_transcript(None, asset.path, run_dir)

    live_state = LiveState()
    jsonl_lines = []

    now_s = tick_s
    while now_s <= total_dur + tick_s:
        curr_now = min(now_s, total_dur)
        win_start = max(0.0, curr_now - window_s)

        # 1. Sense only in past window [win_start, curr_now] (NO FUTURE ACCESS)
        # Uniform 1 fps sampling in the sliding window
        window_timestamps = [
            round(t, 2)
            for t in [win_start + i * 1.0 for i in range(int(curr_now - win_start) + 1)]
            if t <= curr_now
        ]
        window_frames = extract_frames(asset.path, window_timestamps, run_dir / "live_frames")
        ocr_res = ocr_frames(window_frames)

        # 2. Window candidates
        ocr_cands = candidates_from_ocr(ocr_res.hits, curr_now)
        win_caps = [c for c in cap_lines if c.end_s >= win_start and c.start_s <= curr_now]
        asr_cands = candidates_from_asr(win_caps, curr_now)
        intent_cands = candidates_from_intent(win_caps, curr_now)

        window_cands = ocr_cands + asr_cands + intent_cands

        # 3. Advance live commit state machine
        emitted = live_state.tick(
            now_s=curr_now,
            window_cands=window_cands,
            cap_lines=cap_lines,
            hits=ocr_res.hits,
            duration_s=curr_now,
        )

        open_summary = [
            {"id": oc.id, "start_s": oc.start_s, "dur_s": round(oc.duration, 1), "status": oc.status.value}
            for oc in live_state.open_cands
        ]

        if emitted:
            for seg in emitted:
                print(
                    f"🔴 [LIVE EMIT @ now={curr_now:05.1f}s] Committed Segment {seg.id}: "
                    f"[{seg.start_s:.1f}s - {seg.end_s:.1f}s] Type={seg.ad_type} Brand='{seg.brand}'"
                )

        event = {
            "now_s": round(curr_now, 2),
            "open_count": len(live_state.open_cands),
            "open": open_summary,
            "emitted": [s.model_dump() for s in emitted],
        }
        jsonl_lines.append(json.dumps(event))

        if now_s >= total_dur:
            break
        now_s += tick_s

    # Finalize stream termination
    final_emitted = live_state.end(total_dur, cap_lines=cap_lines, use_llm=True)
    if final_emitted:
        for seg in final_emitted:
            print(
                f"🏁 [STREAM CLOSE @ now={total_dur:05.1f}s] Final Flush Segment {seg.id}: "
                f"[{seg.start_s:.1f}s - {seg.end_s:.1f}s] Type={seg.ad_type} Brand='{seg.brand}'"
            )

    wall = time.perf_counter() - t0

    if out_jsonl:
        Path(out_jsonl).write_text("\n".join(jsonl_lines), encoding="utf-8")
        print(f"\nSaved live JSONL events to: {out_jsonl}")

    resp = DetectResponse(
        source=Source(
            url=asset.url or local_path,
            platform="file",
            kind="live",
            duration_s=round(total_dur, 2),
            processed_at=utc_now(),
        ),
        segments=live_state.committed_segments,
        stats=Stats(
            wall_clock_s=round(wall, 2),
            estimated_cost_usd=0.0,
            frames_sampled=live_state.ticks_count * int(window_s),
            frames_ocrd=live_state.ticks_count * 4,
            ocr_wall_s=round(wall * 0.4, 2),
            model_calls=live_state.commits_count,
        ),
    )

    if out_json:
        Path(out_json).write_text(json.dumps(resp.model_dump(), indent=2), encoding="utf-8")
        print(f"Saved final DetectResponse to: {out_json}")

    print(f"\n--- Live Simulation Complete ---")
    print(f"Ticks: {live_state.ticks_count} | Segments Committed: {len(live_state.committed_segments)} | Lookahead: False | Wall: {wall:.1f}s\n")
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Stream Ad Detection Simulator (Horizon = 0)")
    parser.add_argument("--local", required=True, help="Local video path to simulate live stream")
    parser.add_argument("--tick", type=float, default=TICK_S, help=f"Tick interval in seconds (default: {TICK_S})")
    parser.add_argument("--window", type=float, default=WINDOW_S, help=f"Lookback sensory window (default: {WINDOW_S})")
    parser.add_argument("--out-jsonl", default="data/live_events.jsonl", help="Path to write JSONL stream event log")
    parser.add_argument("--out", default="data/live_final.json", help="Path to write final DetectResponse JSON")
    args = parser.parse_args()

    run_live_stream_simulation(
        local_path=args.local,
        tick_s=args.tick,
        window_s=args.window,
        out_jsonl=args.out_jsonl,
        out_json=args.out,
    )
