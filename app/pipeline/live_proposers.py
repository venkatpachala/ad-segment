"""Live-news-specific candidate proposers.

Three proposers designed for live news with L-bar bugs:

P1  persist_proposer  — repeated text in a fixed ROI corner for ≥ LIVE_PERSIST_S
                        is a commercial overlay (G-Mart / KCL / BHIM pattern).
P2  scene_cut_proposer — full-frame OCR ONLY when a hard scene cut fires in
                         the tick window.  Detects commercial break slates.
P3  intent_proposer   — thin wrapper around candidates_from_intent, clipped to
                         captions with start_s ≤ now_s (zero-lookahead).

Design notes
------------
- P1 never calls Tesseract; it receives already-computed TickRoiResult objects.
- P2 calls Tesseract once on the full frame only when Bhattacharyya histogram
  distance crosses the cut threshold.  Average = 0 OCR calls/tick on news.
- Neither P1 nor P2 look beyond now_s (horizon = 0 preserved).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.pipeline.roi import (
    TickRoiResult,
    is_brand_candidate,
    is_channel_chrome,
    normalize_ocr_text,
    ocr_crop,
    strip_channel_chrome,
)
from app.pipeline.types import Candidate

if TYPE_CHECKING:
    from app.pipeline.asr import CaptionLine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIVE_PERSIST_S: float = 8.0     # same brand in same ROI this long → overlay
SCENE_CUT_THRESHOLD: float = 0.35  # Bhattacharyya: treat as a hard cut
SCENE_OCR_THRESHOLD: float = 0.28  # milder change: still full-frame OCR (Samsung/Daikin)
HIST_SIZE = [8, 8, 8]
HIST_RANGES = [0, 256, 0, 256, 0, 256]

# OCR score threshold for break detection (mirrors ocr.py OCR_SCORE_THRESHOLD)
BREAK_OCR_SCORE_MIN: int = 2

_SKIP_SIG_TOKENS: frozenset[str] = frozenset(
    {
        "asianet", "news", "live", "friday", "saturday", "sunday", "monday",
        "tuesday", "wednesday", "thursday", "overs", "runs", "wickets",
        "kerala", "cricket", "league", "cable", "august", "breaking",
        "update", "reporter", "channel", "stream", "watch", "subscribe",
        "the", "and", "for", "get", "now", "most", "from", "any", "app",
        "years", "combo", "free", "buy", "with", "hero", "world", "order",
        "digital", "expert", "powered", "available",
    }
)
_STRONG_BRAND = re.compile(
    r"nandilath|\bg[.\s-]?mart\b|\bdaikin\b|\bsamsung\b|\bgalaxy\b|"
    r"chicking|\bjewellers?\b|\bvida\b",
    re.I,
)


# ---------------------------------------------------------------------------
# P1 — Persistent bug proposer
# ---------------------------------------------------------------------------

@dataclass
class _BugTrack:
    """Track for one (roi, text_sig) pair."""
    roi_name: str
    text_sig: str       # first 40 chars of normalize_ocr_text(raw)
    first_seen_s: float
    last_seen_s: float
    sample_texts: list[str] = field(default_factory=list)  # raw OCR samples


class PersistHistory:
    """Cross-tick memory for the persist proposer.

    Key: (roi_name, text_sig)
    Value: _BugTrack
    """

    def __init__(self) -> None:
        self._tracks: dict[tuple[str, str], _BugTrack] = {}
        # roi_name → text_sig that was active last tick (for expiry)
        self._last_active: dict[str, str | None] = {}

    def _make_sig(self, raw_or_norm: str) -> str:
        return persist_signature(raw_or_norm)

    def update(self, roi_result: TickRoiResult, now_s: float) -> None:
        """Record one ROI observation for this tick."""
        roi = roi_result.roi_name
        if roi_result.is_chrome or not roi_result.is_brand:
            # Nothing commercial in this ROI this tick → let existing track age
            self._last_active[roi] = None
            return

        sig = self._make_sig(roi_result.raw_text or roi_result.norm_text)
        if not sig:
            self._last_active[roi] = None
            return

        key = (roi, sig)
        if key in self._tracks:
            self._tracks[key].last_seen_s = now_s
            if roi_result.raw_text:
                self._tracks[key].sample_texts.append(roi_result.raw_text)
        else:
            self._tracks[key] = _BugTrack(
                roi_name=roi,
                text_sig=sig,
                first_seen_s=now_s,
                last_seen_s=now_s,
                sample_texts=[roi_result.raw_text] if roi_result.raw_text else [],
            )
        self._last_active[roi] = sig

    def expire_stale(self, now_s: float, silence_s: float = 6.0) -> None:
        """Remove tracks not seen for silence_s seconds."""
        stale = [
            k for k, t in self._tracks.items()
            if now_s - t.last_seen_s >= silence_s
        ]
        for k in stale:
            del self._tracks[k]

    def active_candidates(self, now_s: float) -> list[Candidate]:
        """Return overlay candidates that have persisted ≥ LIVE_PERSIST_S."""
        cands: list[Candidate] = []
        for (roi, sig), track in self._tracks.items():
            if now_s - track.first_seen_s < LIVE_PERSIST_S:
                continue
            best_text = _best_raw(track.sample_texts) or sig
            cands.append(
                Candidate(
                    start_s=track.first_seen_s,
                    end_s=min(track.last_seen_s, now_s),  # never exceed horizon
                    source=f"ocr_persist_{roi}",
                    raw_triggers=["persist_roi", "live_overlay"],
                    texts=[best_text],
                    frame_timestamps=[track.first_seen_s, min(track.last_seen_s, now_s)],
                    score_hint=3,  # crosses judge_candidate score_hint >= 2 gate
                )
            )
        return cands


def _best_raw(texts: list[str]) -> str:
    """Pick the longest (most complete) raw OCR sample."""
    if not texts:
        return ""
    return max(texts, key=len)


def persist_signature(text: str) -> str:
    """Stable brand key from noisy live OCR (ticker junk around G-Mart / Daikin)."""
    if not text:
        return ""
    stripped = strip_channel_chrome(text)
    if _STRONG_BRAND.search(stripped):
        if re.search(r"nandilath|\bg[.\s-]?mart\b", stripped, re.I):
            return "gmart"
        if re.search(r"\bdaikin\b", stripped, re.I):
            return "daikin"
        if re.search(r"\bsamsung\b|\bgalaxy\b", stripped, re.I):
            return "samsung"
        if re.search(r"chicking", stripped, re.I):
            return "chicking"
        if re.search(r"\bvida\b", stripped, re.I):
            return "vida"
        if re.search(r"jeweller", stripped, re.I):
            return "jewellers"
    m = re.search(r"\b([a-z0-9][a-z0-9\-]{2,}\.(?:com|in))\b", stripped, re.I)
    if m:
        host = m.group(1).lower()
        if "asianet" not in host and "news" not in host:
            return host
    # Do not key persist on a random 5-letter OCR token (ticker junk like "moswll").
    return ""


def persist_proposer(
    roi_results: list[TickRoiResult],
    now_s: float,
    history: PersistHistory,
) -> list[Candidate]:
    """Update persist history and return overlay candidates.

    Call once per tick with the ROI sense results.  Returns candidates that
    have been observed continuously for ≥ LIVE_PERSIST_S.

    No Tesseract calls — only uses already-computed TickRoiResult.text.
    """
    # Update tracks for each ROI result
    for r in roi_results:
        history.update(r, now_s)

    # Age out old tracks
    history.expire_stale(now_s)

    return history.active_candidates(now_s)


# ---------------------------------------------------------------------------
# P2 — Scene-cut break proposer
# ---------------------------------------------------------------------------

def _frame_hist(frame_bgr: np.ndarray) -> np.ndarray:
    """Compute 8×8×8 colour histogram (normalised) on a 64×36 thumbnail."""
    small = cv2.resize(frame_bgr, (64, 36))
    hist = cv2.calcHist([small], [0, 1, 2], None, HIST_SIZE, HIST_RANGES)
    return cv2.normalize(hist, hist).flatten()


def _score_text(text: str) -> tuple[int, list[str]]:
    """Score OCR text for commercial signal.  Reuses ocr._score internals."""
    # Import here to avoid circular imports (ocr doesn't import live_proposers)
    import re as _re
    from app.pipeline.ocr import (  # type: ignore[attr-defined]
        _PRICE, _PHONE, _OFFER, _CTA, _SPONSOR, _BRANDISH, _EDITORIAL,
    )
    triggers: list[str] = []
    score = 0
    if _EDITORIAL.search(text) and not _CTA.search(text) and not _PRICE.search(text):
        return 0, []
    if _PRICE.search(text):
        score += 2; triggers.append("price")
    if _PHONE.search(text):
        score += 2; triggers.append("phone")
    if _OFFER.search(text):
        score += 2; triggers.append("offer")
    if _CTA.search(text):
        score += 2; triggers.append("cta")
    if _SPONSOR.search(text):
        score += 2; triggers.append("sponsor")
    if _BRANDISH.search(text):
        score += 2; triggers.append("brand_word")
    if _STRONG_BRAND.search(text) and score < 2:
        score = 2
        triggers.append("brand_word")
    return score, triggers


@dataclass
class SceneCutState:
    """Carries the previous frame's histogram across ticks."""
    prev_hist: np.ndarray | None = field(default=None, repr=False)


def scene_cut_proposer(
    frame_bgr: np.ndarray,
    state: SceneCutState,
    now_s: float,
) -> tuple[Candidate | None, bool]:
    """Detect a scene cut; if found, OCR the full frame once for break detection.

    Returns (candidate_or_None, cut_detected).

    Full-frame Tesseract is called ONLY when a cut is detected.
    On typical live news, a cut fires ≈ 0–1 times per tick → 0 full-frame
    OCR calls on a stable news show.
    """
    hist = _frame_hist(frame_bgr)
    cut_detected = False

    if state.prev_hist is not None:
        dist = float(cv2.compareHist(state.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        cut_detected = dist >= SCENE_CUT_THRESHOLD
        # OCR on a hard cut OR a milder visual change (static-looking TV spots
        # sit just under 0.35 and were previously skipped entirely).
        if dist >= SCENE_OCR_THRESHOLD:
            text = ocr_crop(frame_bgr)
            score, triggers = _score_text(text)
            from app.pipeline.ocr import guess_ocr_brand
            brand = guess_ocr_brand(text)
            if brand and score < 2 and persist_signature(text):
                score = 2
                triggers.append(f"brand:{brand}")
            if score >= BREAK_OCR_SCORE_MIN:
                cand = Candidate(
                    start_s=now_s,
                    end_s=now_s,
                    source="ocr_break",
                    raw_triggers=triggers,
                    texts=[text],
                    frame_timestamps=[now_s],
                    score_hint=score,
                )
                state.prev_hist = hist
                return cand, cut_detected

    state.prev_hist = hist
    return None, cut_detected


def roi_spot_proposer(roi_results: list[TickRoiResult], now_s: float) -> list[Candidate]:
    """Short commercial spots visible in a live ROI this tick (Daikin / Samsung).

    Does not wait for LIVE_PERSIST_S — a 4s TV spot is still an ad.
    """
    cands: list[Candidate] = []
    for r in roi_results:
        if r.is_chrome or not r.raw_text:
            continue
        score, triggers = _score_text(r.raw_text)
        sig = persist_signature(r.raw_text)
        if score < BREAK_OCR_SCORE_MIN:
            if not _STRONG_BRAND.search(r.raw_text or ""):
                continue
            score = 2
            triggers.append("brand_word")
        cands.append(
            Candidate(
                start_s=now_s,
                end_s=now_s,
                source=f"ocr_break_{r.roi_name}",
                raw_triggers=triggers or ["brand_word"],
                texts=[r.raw_text],
                frame_timestamps=[now_s],
                score_hint=score,
            )
        )
    return cands


# ---------------------------------------------------------------------------
# P3 — Intent proposer wrapper (zero-lookahead)
# ---------------------------------------------------------------------------

def intent_proposer_live(
    cap_lines: list[CaptionLine],
    now_s: float,
    window_s: float,
) -> list[Candidate]:
    """Run intent proposer on captions clipped to [now-window_s, now].

    Horizon = 0: only captions with start_s ≤ now_s are considered.
    """
    from app.pipeline.intent import candidates_from_intent

    win_start = max(0.0, now_s - window_s)
    window_caps = [
        c for c in cap_lines
        if c.end_s >= win_start and c.start_s <= now_s
    ]
    if not window_caps:
        return []
    # candidates_from_intent clips end_s to duration_s=now_s
    cands = candidates_from_intent(window_caps, now_s)
    # Extra clip: drop anything that extends past now_s
    return [
        Candidate(
            start_s=c.start_s,
            end_s=min(c.end_s, now_s),
            source=c.source,
            raw_triggers=c.raw_triggers,
            texts=c.texts,
            frame_timestamps=[t for t in c.frame_timestamps if t <= now_s],
            score_hint=c.score_hint,
        )
        for c in cands
        if c.start_s < now_s
    ]
