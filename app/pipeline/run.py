from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from app.config import RUNS_DIR, SCENE_SKIP_UNDER_S
from app.models.schemas import DetectRequest, DetectResponse, Source, Stats, utc_now
from app.pipeline.asr import candidates_from_asr, fetch_transcript, load_transcript, refine_asr_start
from app.pipeline.intent import candidates_from_intent
from app.pipeline.candidates import candidates_from_ocr, drop_incidental_ocr, union_candidates
from app.pipeline.frames import extract_frames, sampling_plan
from app.pipeline.ingest import ingest
from app.pipeline.llm_judge import judge_candidates_batch, llm_available
from app.pipeline.ocr import ocr_frames
from app.pipeline.scene_detection import scene_cuts
from app.pipeline.temporal import snap_segments


def _is_youtube(url: str | None) -> bool:
    return bool(url and re.search(r"youtube\.com|youtu\.be", url, re.I))


def run_detect(
    req: DetectRequest,
    run_id: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> DetectResponse:
    t0 = time.perf_counter()
    run_dir = RUNS_DIR / (run_id or time.strftime("%Y%m%dT%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    def status(s: str) -> None:
        if on_status:
            on_status(s)

    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="adseg")

    # ── Step 1: Caption fetch starts IMMEDIATELY (only needs URL, not video) ──
    # This runs concurrently with the video download below, typically saving 5-15s.
    cap_url = req.url if _is_youtube(req.url) else None
    cap_fut: Future = executor.submit(fetch_transcript, cap_url, run_dir)

    # ── Step 2: Ingest / download video (blocking on main thread) ──
    # kind_hint is CLI-only; public POST /v1/detect never sets it (kind from duration).
    status("downloading")
    t1 = time.perf_counter()
    asset = ingest(req.url, req.local_path, req.kind_hint, run_dir)
    print(f"PERF: ingest={time.perf_counter()-t1:.1f}s  duration={asset.duration_s:.1f}s  kind={asset.kind}")
    status("detecting")

    # ── Step 3: Scene cuts in background WHILE we extract frames ──
    # Skip entirely for clips under SCENE_SKIP_UNDER_S (e.g. < 60s) — not worth it.
    t2 = time.perf_counter()
    if asset.duration_s >= SCENE_SKIP_UNDER_S:
        cuts_fut: Future | None = executor.submit(scene_cuts, asset.path, 0.35, 40)
    else:
        cuts_fut = None

    timestamps = sampling_plan(asset.kind, asset.duration_s)
    frames = extract_frames(asset.path, timestamps, run_dir / "frames")

    cuts: list[float] = cuts_fut.result() if cuts_fut else []
    print(f"PERF: frames+scene={time.perf_counter()-t2:.1f}s  frames={len(frames)}  cuts={len(cuts)}")

    # Add extra frames at scene-cut boundaries not already sampled
    extra = [c for c in cuts if all(abs(c - t) > 0.2 for t in timestamps)]
    if extra:
        more = extract_frames(asset.path, extra[:20], run_dir / "frames_cuts")
        frames.extend(more)
        frames.sort(key=lambda f: f.timestamp_s)

    # ── Step 4: OCR — skip near-duplicates, then a small Tesseract thread pool ──
    ocr_result = ocr_frames(frames)
    hits = ocr_result.hits
    print(
        f"PERF: ocr={ocr_result.ocr_wall_s:.1f}s  sampled={ocr_result.frames_sampled}  "
        f"ocrd={ocr_result.frames_ocrd}  hits={len(hits)}"
    )
    (run_dir / "ocr.jsonl").write_text(
        "\n".join(
            f"{h.timestamp_s:.2f}\t{h.score}\t{h.triggers}\t{h.text}" for h in hits
        ),
        encoding="utf-8",
    )

    # ── Step 5: Captions (likely already done since download+frames+OCR took time) ──
    t4 = time.perf_counter()
    pre_caps = cap_fut.result()      # blocks only if caption fetch is still running
    cap_wait = time.perf_counter() - t4
    if cap_wait > 0.5:
        print(f"PERF: waited {cap_wait:.1f}s for caption fetch to finish")

    transcript = load_transcript(
        asset.url if asset.platform == "youtube" else None,
        asset.path,
        run_dir,
        prefetched=pre_caps,  # avoids a second yt-dlp call
    )
    cap_lines = transcript.lines
    transcript_lang = transcript.language
    print(f"INFO: full_transcript_cues={len(cap_lines)} lang={transcript_lang or '-'}")

    # Shut down the thread pool — all background tasks are done
    executor.shutdown(wait=False)

    asr_cands = candidates_from_asr(cap_lines, asset.duration_s)
    asr_cands = [refine_asr_start(c, cap_lines) for c in asr_cands]
    intent_cands = candidates_from_intent(cap_lines, asset.duration_s)
    print(f"INFO: lexicon_cands={len(asr_cands)} intent_cands={len(intent_cands)}")
    (run_dir / "captions_used.txt").write_text(
        "\n".join(f"{ln.start_s:.2f}-{ln.end_s:.2f} {ln.text}" for ln in cap_lines[:400]),
        encoding="utf-8",
    )

    ocr_cands = candidates_from_ocr(hits, asset.duration_s)
    cands = drop_incidental_ocr(ocr_cands + asr_cands + intent_cands, asset.kind, asset.duration_s)
    cands = union_candidates([cands])
    provider = llm_available()
    print(f"INFO: candidates={len(cands)} llm={provider or 'off (rule judge)'}")

    # ── Step 6: LLM judge (optional; OCR-only preroll/overlay uses rules) ──
    t5 = time.perf_counter()
    segments: list = []
    model_calls = 0
    llm_fallback = ""
    if cands:
        segments, model_calls, llm_fallback = judge_candidates_batch(
            cands, asset.kind, asset.duration_s, cap_lines, hits, transcript_lang=transcript_lang
        )
    if model_calls:
        print(f"PERF: llm_judge={time.perf_counter()-t5:.1f}s  calls={model_calls}")
    elif llm_fallback:
        print(f"INFO: llm_fallback={llm_fallback} model_calls=0")

    segments = snap_segments(segments, cuts, asset.duration_s)

    wall = time.perf_counter() - t0
    print(f"PERF: total_wall={wall:.1f}s")
    return DetectResponse(
        source=Source(
            url=asset.url,
            platform=asset.platform,  # type: ignore[arg-type]
            kind=asset.kind,  # type: ignore[arg-type]
            duration_s=round(asset.duration_s, 3),
            processed_at=utc_now(),
            language=transcript_lang,
        ),
        segments=segments,
        stats=Stats(
            wall_clock_s=round(wall, 2),
            estimated_cost_usd=round(0.002 * model_calls, 4),
            frames_sampled=ocr_result.frames_sampled or len(frames),
            frames_ocrd=ocr_result.frames_ocrd,
            ocr_wall_s=round(ocr_result.ocr_wall_s, 2),
            model_calls=model_calls,
            llm_fallback=llm_fallback,
        ),
    )