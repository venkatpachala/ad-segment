from __future__ import annotations

import pytest
from app.pipeline.live import LiveState, LiveStatus, OpenCand, MIN_S, SILENCE_S
from app.pipeline.types import Candidate


def test_live_tick_provisional_extending_committed():
    state = LiveState()

    # Tick 1 at now=4.0s: Ad appears at 1.0s and ends at 4.0s (dur=3.0s -> reaches MIN_S)
    cands_t1 = [
        Candidate(start_s=1.0, end_s=4.0, source="ocr", raw_triggers=["brand_word"], texts=["Avis Hospital"], frame_timestamps=[1.0, 2.0, 3.0], score_hint=2)
    ]
    emits_1 = state.tick(now_s=4.0, window_cands=cands_t1, use_llm=False)
    assert len(emits_1) == 0  # Still open / active
    assert len(state.open_cands) == 1
    assert state.open_cands[0].status == LiveStatus.EXTENDING
    assert state.open_cands[0].start_s == 1.0
    assert state.open_cands[0].last_evidence_s == 4.0

    # Tick 2 at now=8.0s: Ad continues from 4.0s to 8.0s
    cands_t2 = [
        Candidate(start_s=4.0, end_s=8.0, source="ocr", raw_triggers=["brand_word"], texts=["Avis Hospital"], frame_timestamps=[5.0, 6.0, 7.0], score_hint=2)
    ]
    emits_2 = state.tick(now_s=8.0, window_cands=cands_t2, use_llm=False)
    assert len(emits_2) == 0  # Still open
    assert state.open_cands[0].last_evidence_s == 8.0

    # Tick 3 at now=12.0s: Silence (ad ended at 8.0s). gap = 12.0 - 8.0 = 4.0s < SILENCE_S (6.0s) -> not yet committed
    emits_3 = state.tick(now_s=12.0, window_cands=[], use_llm=False)
    assert len(emits_3) == 0

    # Tick 4 at now=15.0s: Silence continues. gap = 15.0 - 8.0 = 7.0s >= SILENCE_S (6.0s) -> COMMITTED!
    emits_4 = state.tick(now_s=15.0, window_cands=[], use_llm=False)
    assert len(emits_4) == 1
    seg = emits_4[0]
    assert seg.start_s == 1.0
    assert seg.end_s == 8.0
    assert "ocr" in seg.evidence.signals_used
    assert seg.end_s <= 15.0  # Zero lookahead satisfied


def test_live_transient_glitch_dropped():
    state = LiveState()

    # 1-second transient screen flash (below MIN_S=3.0s)
    cands = [
        Candidate(start_s=2.0, end_s=3.0, source="ocr", raw_triggers=["brand_word"], texts=["Random Text"], frame_timestamps=[2.5], score_hint=2)
    ]
    state.tick(now_s=3.0, window_cands=cands, use_llm=False)
    assert state.open_cands[0].status == LiveStatus.PROVISIONAL

    # Silence follows up to now=10.0s (gap = 7.0s >= SILENCE_S) -> dropped silently
    emits = state.tick(now_s=10.0, window_cands=[], use_llm=False)
    assert len(emits) == 0
    assert len(state.committed_segments) == 0
    assert len(state.open_cands) == 0


def test_live_horizon_zero_clipping():
    state = LiveState()

    # If an erroneous future candidate with end_s=20.0 arrives at now=10.0s
    cands = [
        Candidate(start_s=5.0, end_s=20.0, source="ocr", raw_triggers=["brand_word"], texts=["Avis"], frame_timestamps=[6.0, 15.0], score_hint=2)
    ]
    state.tick(now_s=10.0, window_cands=cands, use_llm=False)
    # The candidate is clipped to horizon <= 10.0s
    assert state.open_cands[0].last_evidence_s <= 10.0


def test_live_stream_end_flush():
    state = LiveState()

    # An ad running continuously until the stream terminates
    cands = [
        Candidate(start_s=5.0, end_s=18.0, source="ocr", raw_triggers=["brand_word"], texts=["Avis Hospital"], frame_timestamps=[6.0, 12.0, 18.0], score_hint=2)
    ]
    state.tick(now_s=18.0, window_cands=cands, use_llm=False)

    # Stream suddenly closes at 18.0s
    final_emits = state.end(stream_end_s=18.0, use_llm=False)
    assert len(final_emits) == 1
    assert final_emits[0].start_s == 5.0
    assert final_emits[0].end_s == 18.0


def test_live_main_parser():
    from app.live_main import _build_parser
    parser = _build_parser()
    
    # Test --url
    args = parser.parse_args(["--url", "https://youtube.com/live/test1234", "--tick", "2.0"])
    assert args.url == "https://youtube.com/live/test1234"
    assert args.tick == 2.0
    
    # Test --local
    args2 = parser.parse_args(["--local", "data/test.mp4", "--window", "16.0"])
    assert args2.local == "data/test.mp4"
    assert args2.window == 16.0


def test_live_stream_ingest_init(tmp_path):
    from app.pipeline.live_streamer import LiveStreamIngest
    ts_path = tmp_path / "stream.ts"
    ingest = LiveStreamIngest("https://youtube.com/live/fake", ts_path)
    assert ingest.url == "https://youtube.com/live/fake"
    assert ingest.buffer_path == ts_path
    assert not ingest.is_active()
    assert ingest.get_available_duration_s() == 0.0


def test_event_at_t10_has_no_t400_ads():
    """now=10 must not contain a bug that appears at t=400 (horizon 0)."""
    state = LiveState()
    events = []
    for now in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        window = [
            Candidate(
                start_s=2.0,
                end_s=now,
                source="ocr_persist_tr",
                raw_triggers=["persist_roi"],
                texts=["G-Mart"],
                frame_timestamps=[now],
                score_hint=3,
            ),
            Candidate(
                start_s=400.0,
                end_s=430.0,
                source="ocr",
                raw_triggers=["brand_word"],
                texts=["FUTURE AD"],
                frame_timestamps=[400.0],
                score_hint=5,
            ),
        ]
        emitted = state.tick(now_s=now, window_cands=window, use_llm=False)
        events.append(
            {
                "now_s": now,
                "open": state.provisional_overlays()
                or [
                    {
                        "start_s": oc.start_s,
                        "end_s": oc.last_evidence_s,
                        "text": (oc.texts or [""])[0],
                    }
                    for oc in state.open_cands
                ],
                "emitted": emitted,
            }
        )

    ev10 = next(e for e in events if e["now_s"] == 10.0)
    for item in ev10["open"]:
        assert item["end_s"] <= 10.0 + 1e-6
        assert item["start_s"] < 50
        assert "FUTURE" not in str(item.get("text", "")).upper()
    for seg in ev10["emitted"]:
        assert seg.end_s <= 10.0 + 1e-6
        assert seg.start_s < 50
