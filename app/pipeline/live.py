from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generator

from app.models.schemas import Evidence, Segment
from app.pipeline.asr import CaptionLine, candidates_from_asr
from app.pipeline.intent import candidates_from_intent
from app.pipeline.candidates import candidates_from_ocr
from app.pipeline.judge import judge_candidate
from app.pipeline.llm_judge import judge_with_llm, llm_available
from app.pipeline.live_proposers import persist_signature
from app.pipeline.ocr import OcrHit, normalize_ocr_text, ocr_frames
from app.pipeline.temporal import snap_segments
from app.pipeline.types import Candidate, SampledFrame

# Live stream timing constants
TICK_S: float = 2.0         # Tick frequency
WINDOW_S: float = 16.0      # Sliding sensory lookback window (16s = ~3 HLS segments of context)
SILENCE_S: float = 6.0      # Inactivity gap to trigger commit
MIN_S: float = 2.0          # Two live ticks (2s clock) confirm a short TV spot
MAX_OPEN_S: float = 180.0   # Maximum open span before forced commit (handles stuck watermarks)
LIVE_PERSIST_S: float = 8.0 # Same text in same ROI this long → provisional overlay emit


class LiveStatus(str, Enum):
    PROVISIONAL = "provisional"  # Duration < MIN_S; awaiting further evidence
    EXTENDING = "extending"      # Duration >= MIN_S; active confirmed commercial span
    COMMITTED = "committed"      # Closed and emitted to output stream


@dataclass
class OpenCand:
    id: str
    start_s: float
    last_evidence_s: float
    source: str
    raw_triggers: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)
    score_hint: int = 0
    status: LiveStatus = LiveStatus.PROVISIONAL
    roi: str = ""                    # "tr", "br", or "" for non-ROI sources
    text_sig: str = ""               # normalised OCR signature for persist matching
    already_published: bool = False  # True once provisional emit has fired

    @property
    def duration(self) -> float:
        return max(0.0, self.last_evidence_s - self.start_s)

    def to_candidate(self) -> Candidate:
        return Candidate(
            start_s=self.start_s,
            end_s=self.last_evidence_s,
            source=self.source,
            raw_triggers=list(self.raw_triggers),
            texts=list(self.texts),
            frame_timestamps=list(self.frame_timestamps),
            score_hint=self.score_hint,
        )


@dataclass
class LiveState:
    """Zero-lookahead state machine for live streaming ad detection.

    Constraints:
    - Horizon = 0: All boundaries emitted must satisfy end_s <= now_s.
    - Sensors tick periodically on [now_s - WINDOW_S, now_s].
    - LLM / Rule judgment runs ONLY on commit (when a candidate closes).
    - Snap only to scene cuts already seen <= now_s.
    """

    now_s: float = 0.0
    open_cands: list[OpenCand] = field(default_factory=list)
    committed_segments: list[Segment] = field(default_factory=list)
    seen_scene_cuts: list[float] = field(default_factory=list)
    ticks_count: int = 0
    commits_count: int = 0
    ocr_calls_total: int = 0       # total Tesseract calls across all ticks
    tick_wall_ms_list: list[float] = field(default_factory=list)  # per-tick wall ms
    transcript_lang: str = ""
    _next_id: int = 1

    def provisional_overlays(self) -> list[dict]:
        """Return growing-end_s entries for active open commercial tracks immediately."""
        result = []
        for oc in self.open_cands:
            if oc.status == LiveStatus.COMMITTED:
                continue
            if oc.score_hint >= 1:
                oc.already_published = True
                best_text = max(oc.texts, key=len) if oc.texts else ""
                from app.pipeline.ocr import guess_ocr_brand
                brand = guess_ocr_brand(best_text)
                result.append({
                    "start_s": round(oc.start_s, 2),
                    "end_s": round(min(oc.last_evidence_s, self.now_s), 2),
                    "roi": oc.roi,
                    "text": best_text[:80],
                    "brand": brand,
                    "source": oc.source,
                    "status": "active",
                })
        return result

    def tick(
        self,
        now_s: float,
        window_cands: list[Candidate],
        seen_cuts_in_window: list[float] | None = None,
        cap_lines: list[CaptionLine] | None = None,
        hits: list[OcrHit] | None = None,
        duration_s: float | None = None,
        use_llm: bool = True,
    ) -> list[Segment]:
        """Advance stream clock to now_s, update open hypotheses, commit closed spans."""
        self.now_s = now_s
        self.ticks_count += 1
        if seen_cuts_in_window:
            for c in seen_cuts_in_window:
                if c <= now_s and c not in self.seen_scene_cuts:
                    self.seen_scene_cuts.append(c)
            self.seen_scene_cuts.sort()

        emitted_now: list[Segment] = []

        # 1. Clip incoming window candidates to horizon (end_s <= now_s)
        clipped_cands: list[Candidate] = []
        for c in window_cands:
            if c.start_s >= now_s:
                continue
            clipped = Candidate(
                start_s=max(0.0, c.start_s),
                end_s=min(now_s, c.end_s),
                source=c.source,
                raw_triggers=c.raw_triggers,
                texts=c.texts,
                frame_timestamps=[t for t in c.frame_timestamps if t <= now_s],
                score_hint=c.score_hint,
            )
            clipped_cands.append(clipped)

        # 2. Match clipped candidates to active open tracks (prefer roi+text_sig).
        matched_ids = set()
        for c in clipped_cands:
            matched = False
            roi_hint = (
                "left" if "left" in c.source
                else ("bottom" if "bottom" in c.source
                else ("tr" if "tr" in c.source
                else ("br" if "br" in c.source else "")))
            )
            blob = max(c.texts, key=len) if c.texts else ""
            sig = persist_signature(blob) or normalize_ocr_text(blob)[:40]
            persist = "persist" in c.source
            for oc in self.open_cands:
                if oc.status == LiveStatus.COMMITTED:
                    continue
                oc_persist = "persist" in oc.source
                same_sig = bool(
                    sig and oc.text_sig and (
                        sig == oc.text_sig or sig in oc.text_sig or oc.text_sig in sig
                    )
                )
                # Same brand (G-Mart on L-bar vs full-frame) glues even across sources.
                # Otherwise never glue persist overlay with an unrelated break.
                if persist != oc_persist and not same_sig:
                    continue
                same_roi_sig = bool(
                    roi_hint and oc.roi == roi_hint and same_sig
                )
                overlap = (
                    c.start_s <= (oc.last_evidence_s + SILENCE_S)
                    and c.end_s >= (oc.start_s - SILENCE_S)
                )
                if same_roi_sig or same_sig or (overlap and not persist and not oc_persist):
                    oc.start_s = min(oc.start_s, c.start_s)
                    oc.last_evidence_s = max(oc.last_evidence_s, min(c.end_s, now_s))
                    oc.frame_timestamps.extend(c.frame_timestamps)
                    oc.texts.extend(c.texts)
                    if sig:
                        oc.text_sig = oc.text_sig or sig
                    for tr in c.raw_triggers:
                        if tr not in oc.raw_triggers:
                            oc.raw_triggers.append(tr)
                    for src in ("ocr", "asr", "intent"):
                        if src in c.source and src not in oc.source:
                            oc.source = f"{oc.source}+{src}"
                    oc.score_hint = max(oc.score_hint, c.score_hint)
                    if oc.duration >= MIN_S:
                        oc.status = LiveStatus.EXTENDING
                    matched_ids.add(oc.id)
                    matched = True
                    break

            if not matched:
                status = LiveStatus.EXTENDING if (c.end_s - c.start_s) >= MIN_S else LiveStatus.PROVISIONAL
                new_oc = OpenCand(
                    id=f"live_open_{self._next_id:03d}",
                    start_s=c.start_s,
                    last_evidence_s=min(c.end_s, now_s),
                    source=c.source,
                    raw_triggers=list(c.raw_triggers),
                    texts=list(c.texts),
                    frame_timestamps=list(c.frame_timestamps),
                    score_hint=c.score_hint,
                    status=status,
                    roi=roi_hint,
                    text_sig=sig,
                )
                self._next_id += 1
                self.open_cands.append(new_oc)
                matched_ids.add(new_oc.id)

        # 3. Check for candidates that reached commit condition (silence >= SILENCE_S or MAX_OPEN_S)
        remaining_open: list[OpenCand] = []
        for oc in self.open_cands:
            gap = now_s - oc.last_evidence_s
            should_close = gap >= SILENCE_S or oc.duration >= MAX_OPEN_S

            if should_close:
                if oc.duration >= MIN_S and oc.score_hint >= 1:
                    oc.status = LiveStatus.COMMITTED
                    seg = self._judge_and_commit(
                        oc,
                        cap_lines=cap_lines or [],
                        hits=hits or [],
                        duration_s=duration_s or now_s,
                        use_llm=use_llm,
                    )
                    if seg:
                        self.committed_segments.append(seg)
                        emitted_now.append(seg)
                        self.commits_count += 1
                # If duration < MIN_S, it was a 1-frame glitch and is dropped silently
            else:
                remaining_open.append(oc)

        self.open_cands = remaining_open
        return emitted_now

    def _judge_and_commit(
        self,
        oc: OpenCand,
        cap_lines: list[CaptionLine],
        hits: list[OcrHit],
        duration_s: float,
        use_llm: bool = True,
    ) -> Segment | None:
        """Run judge policy ONLY when a candidate closes."""
        cand = oc.to_candidate()
        idx = len(self.committed_segments) + 1
        seg: Segment | None = None

        ocr_only = "ocr" in oc.source and "asr" not in oc.source and "intent" not in oc.source
        # Overlay commit → rules only. LLM only on speech commit.
        if use_llm and (not ocr_only) and llm_available():
            try:
                # 2-second fast timeout for live stream latency
                seg = judge_with_llm(cand, "live", duration_s, idx, cap_lines, hits)
            except Exception as e:
                print(f"INFO: live LLM fallback on commit ({e})")
                seg = judge_candidate(
                    cand, "live", duration_s, idx, cap_lines, transcript_lang=self.transcript_lang
                )
        else:
            seg = judge_candidate(
                cand, "live", duration_s, idx, cap_lines, transcript_lang=self.transcript_lang
            )
        if seg is not None and self.transcript_lang and not seg.evidence.transcript_lang:
            seg.evidence.transcript_lang = self.transcript_lang

        if seg is None:
            return None

        # Snap to past cuts only (cuts <= now_s)
        past_cuts = [c for c in self.seen_scene_cuts if c <= self.now_s]
        snapped_list = snap_segments([seg], past_cuts, duration_s)
        return snapped_list[0] if snapped_list else seg

    def end(
        self,
        stream_end_s: float,
        cap_lines: list[CaptionLine] | None = None,
        hits: list[OcrHit] | None = None,
        use_llm: bool = True,
    ) -> list[Segment]:
        """Flush remaining open candidates when the live session terminates."""
        self.now_s = stream_end_s
        emitted_now: list[Segment] = []
        for oc in self.open_cands:
            if oc.duration >= MIN_S and oc.score_hint >= 1:
                oc.status = LiveStatus.COMMITTED
                seg = self._judge_and_commit(
                    oc,
                    cap_lines=cap_lines or [],
                    hits=hits or [],
                    duration_s=stream_end_s,
                    use_llm=use_llm,
                )
                if seg:
                    self.committed_segments.append(seg)
                    emitted_now.append(seg)
                    self.commits_count += 1
        self.open_cands.clear()
        return emitted_now
