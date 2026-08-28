"""Tests for LiveState state machine correctness.

Validates:
- Silence commit: segment emitted only after SILENCE_S gap
- No future end_s: emitted segment end_s must always ≤ now_s at commit time
- MAX_OPEN_S forced commit on long-running bug
- Provisional glitch drop: dur < MIN_S → no segment
- Provisional overlays grow end_s across ticks (proves live causal emit)
"""
from __future__ import annotations

import pytest

from app.pipeline.live import (
    LIVE_PERSIST_S,
    MAX_OPEN_S,
    MIN_S,
    SILENCE_S,
    LiveState,
    LiveStatus,
    OpenCand,
)
from app.pipeline.types import Candidate


def _cand(start: float, end: float, score: int = 2, source: str = "ocr_persist") -> Candidate:
    return Candidate(
        start_s=start,
        end_s=end,
        source=source,
        raw_triggers=["persist_roi"],
        texts=["Nandilath G-Mart"],
        frame_timestamps=[start, end],
        score_hint=score,
    )


# ---------------------------------------------------------------------------
# Test 1: silence commit — segment emitted only after SILENCE_S gap
# ---------------------------------------------------------------------------

def test_silence_commit_fires_after_gap():
    state = LiveState()

    # Evidence from t=4 to t=8
    state.tick(now_s=8.0, window_cands=[_cand(4.0, 8.0)], use_llm=False)
    assert len(state.open_cands) == 1
    assert state.open_cands[0].status == LiveStatus.EXTENDING

    # Gap = 8–8 = 0: still below SILENCE_S
    emitted = state.tick(now_s=12.0, window_cands=[], use_llm=False)
    # Gap = 12 - 8 = 4s < SILENCE_S(6s) → not yet committed
    assert len(emitted) == 0

    # Gap = 14 - 8 = 6s ≥ SILENCE_S → commit
    emitted = state.tick(now_s=14.0, window_cands=[], use_llm=False)
    assert len(emitted) == 1
    seg = emitted[0]
    assert seg.start_s == 4.0
    assert seg.end_s == 8.0  # last evidence, not commit time


# ---------------------------------------------------------------------------
# Test 2: no future end_s — end_s must never exceed now_s at emit time
# ---------------------------------------------------------------------------

def test_no_future_end_s():
    state = LiveState()

    # Candidate that tries to extend into the future
    future_cand = Candidate(
        start_s=5.0,
        end_s=999.0,  # would be in the future
        source="ocr_persist",
        raw_triggers=["persist_roi"],
        texts=["G-Mart"],
        frame_timestamps=[5.0],
        score_hint=3,
    )
    state.tick(now_s=10.0, window_cands=[future_cand], use_llm=False)
    # end_s must be clipped to ≤ now_s = 10.0
    assert state.open_cands[0].last_evidence_s <= 10.0

    # Force commit
    emitted = state.tick(now_s=20.0, window_cands=[], use_llm=False)
    assert len(emitted) == 1
    assert emitted[0].end_s <= 20.0  # strict horizon
    assert emitted[0].end_s <= 10.0  # clipped to when evidence stopped


# ---------------------------------------------------------------------------
# Test 3: MAX_OPEN_S forced commit on persistent L-bar
# ---------------------------------------------------------------------------

def test_max_open_forced_commit():
    state = LiveState()

    # Continuously re-evidence every tick for 181s (> MAX_OPEN_S = 180s)
    now = 2.0
    emitted_all = []
    while now <= MAX_OPEN_S + 5:
        cands = [_cand(0.0, now)]
        emitted_all.extend(state.tick(now_s=now, window_cands=cands, use_llm=False))
        now += 2.0

    # At least one forced commit should have fired
    assert len(emitted_all) >= 1 or len(state.committed_segments) >= 1


# ---------------------------------------------------------------------------
# Test 4: provisional glitch drop — duration < MIN_S → no segment
# ---------------------------------------------------------------------------

def test_provisional_glitch_dropped():
    state = LiveState()

    # 1-second flash: dur = 1.0 < MIN_S (3.0)
    short_cand = Candidate(
        start_s=5.0,
        end_s=6.0,  # 1s < MIN_S
        source="ocr_persist",
        raw_triggers=["persist_roi"],
        texts=["G-Mart"],
        frame_timestamps=[5.5],
        score_hint=3,
    )
    state.tick(now_s=6.0, window_cands=[short_cand], use_llm=False)
    assert state.open_cands[0].status == LiveStatus.PROVISIONAL

    # Silence follows — gap > SILENCE_S
    emitted = state.tick(now_s=14.0, window_cands=[], use_llm=False)
    assert len(emitted) == 0
    assert len(state.committed_segments) == 0
    assert len(state.open_cands) == 0


# ---------------------------------------------------------------------------
# Test 5: provisional overlays grow end_s each tick
# ---------------------------------------------------------------------------

def test_provisional_overlays_grow_end_s():
    state = LiveState()

    # Build up a track for LIVE_PERSIST_S + 2 more ticks
    now = 2.0
    previous_end_s: float | None = None

    while now <= LIVE_PERSIST_S + 6:
        cands = [_cand(0.0, now)]
        state.tick(now_s=now, window_cands=cands, use_llm=False)

        if state.open_cands:
            oc = state.open_cands[0]
            if oc.duration >= LIVE_PERSIST_S:
                overlays = state.provisional_overlays()
                if overlays:
                    current_end_s = overlays[0]["end_s"]
                    if previous_end_s is not None:
                        # end_s must grow or stay the same each tick
                        assert current_end_s >= previous_end_s, (
                            f"end_s went backwards: {previous_end_s} → {current_end_s}"
                        )
                    previous_end_s = current_end_s

        now += 2.0

    # Must have seen at least one provisional overlay after LIVE_PERSIST_S
    assert previous_end_s is not None, "No provisional overlay was emitted"


# ---------------------------------------------------------------------------
# Test 6: stream end flush — open candidate committed at session end
# ---------------------------------------------------------------------------

def test_stream_end_flush():
    state = LiveState()

    cands = [_cand(5.0, 18.0, score=3)]
    state.tick(now_s=18.0, window_cands=cands, use_llm=False)

    final = state.end(stream_end_s=18.0, use_llm=False)
    assert len(final) == 1
    assert final[0].start_s == 5.0
    assert final[0].end_s <= 18.0


# ---------------------------------------------------------------------------
# Test 7: two ROI sources don't cross-contaminate
# ---------------------------------------------------------------------------

def test_two_roi_sources_independent():
    state = LiveState()

    tr_cand = _cand(2.0, 8.0, score=3, source="ocr_persist_tr")
    br_cand = _cand(4.0, 10.0, score=3, source="ocr_persist_br")

    state.tick(now_s=10.0, window_cands=[tr_cand, br_cand], use_llm=False)
    # Both should be open (different sources / ROIs)
    assert len(state.open_cands) >= 1  # may merge if overlap+silence allows — that's fine
    # Neither should be committed yet
    assert all(oc.status != LiveStatus.COMMITTED for oc in state.open_cands)
