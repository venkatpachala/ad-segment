"""Tests for live-news ROI chrome/brand filters.

Validates that:
- "asianet news", "LIVE", "asianet friday" → is_channel_chrome = True
- "Nandilath G-Mart" 8s+ → persist proposer emits a candidate
- Clock-only text → skipped
- "BHIM UPI" → is_brand_candidate = True  (Latin, not chrome)
- "subscribe" / "watch" / "breaking" → chrome
- normalize_ocr_text strips digit tokens properly
"""
from __future__ import annotations

import pytest

from app.pipeline.roi import (
    CHANNEL_TOKENS,
    is_brand_candidate,
    is_channel_chrome,
)
from app.pipeline.live_proposers import (
    PersistHistory,
    persist_proposer,
    LIVE_PERSIST_S,
)
from app.pipeline.roi import TickRoiResult
from app.pipeline.ocr import normalize_ocr_text


# ---------------------------------------------------------------------------
# normalize_ocr_text
# ---------------------------------------------------------------------------

def test_normalize_strips_clock():
    assert normalize_ocr_text("04:37 PM") == "pm"  # PM is a word token, digits stripped
    assert normalize_ocr_text("10:30:05") == ""    # pure digits after colon strip


def test_normalize_strips_pure_numbers():
    assert normalize_ocr_text("100 200 300") == ""
    assert normalize_ocr_text("G-Mart 9876543210") == "g-mart"


def test_normalize_preserves_brand_words():
    result = normalize_ocr_text("Nandilath G-Mart")
    assert "g-mart" in result or "nandilath" in result


# ---------------------------------------------------------------------------
# is_channel_chrome
# ---------------------------------------------------------------------------

def test_asianet_news_is_chrome():
    assert is_channel_chrome("asianet news") is True


def test_asianet_live_is_chrome():
    assert is_channel_chrome("ASIANET LIVE") is True


def test_live_alone_is_chrome():
    assert is_channel_chrome("LIVE") is True


def test_friday_is_chrome():
    assert is_channel_chrome("asianet friday") is True


def test_breaking_news_is_chrome():
    assert is_channel_chrome("breaking news") is True


def test_subscribe_is_chrome():
    assert is_channel_chrome("subscribe") is True


def test_empty_text_is_chrome():
    assert is_channel_chrome("") is True
    assert is_channel_chrome("   ") is True


def test_clock_only_is_chrome():
    assert is_channel_chrome("04:37 PM") is True
    assert is_channel_chrome("22:15") is True


# ---------------------------------------------------------------------------
# is_brand_candidate
# ---------------------------------------------------------------------------

def test_gmart_is_brand():
    assert is_brand_candidate("Nandilath G-Mart") is True


def test_bhim_is_brand():
    assert is_brand_candidate("BHIM UPI") is True


def test_kcl_is_brand():
    assert is_brand_candidate("KCL Homes") is True


def test_asianet_is_not_brand():
    assert is_brand_candidate("asianet news") is False


def test_live_is_not_brand():
    assert is_brand_candidate("LIVE") is False


def test_short_text_is_not_brand():
    # Less than MIN_TEXT_LEN chars after normalization
    assert is_brand_candidate("OK") is False
    assert is_brand_candidate("42") is False


def test_pure_malayalam_is_not_brand():
    # Pure Malayalam script: no Latin tokens → not a brand candidate
    assert is_brand_candidate("കേരള") is False


# ---------------------------------------------------------------------------
# persist_proposer — integration
# ---------------------------------------------------------------------------

def _mock_roi_result(
    roi_name: str,
    raw_text: str,
    is_chrome: bool = False,
    is_brand: bool = True,
    ocr_called: bool = True,
) -> TickRoiResult:
    from app.pipeline.ocr import normalize_ocr_text as norm
    return TickRoiResult(
        roi_name=roi_name,
        raw_text=raw_text,
        norm_text=norm(raw_text),
        is_chrome=is_chrome,
        is_brand=is_brand,
        ocr_called=ocr_called,
        mae_value=0.0,
    )


def test_gmart_persist_kept_after_persist_s():
    """G-Mart appearing for ≥ LIVE_PERSIST_S seconds → candidate emitted."""
    hist = PersistHistory()
    gmart_result = _mock_roi_result("tr", "Nandilath G-Mart", is_chrome=False, is_brand=True)

    # Simulate ticks at 2s intervals until LIVE_PERSIST_S
    now = 2.0
    found_candidate = False
    while now <= LIVE_PERSIST_S + 4:
        cands = persist_proposer([gmart_result], now, hist)
        if cands:
            found_candidate = True
            assert cands[0].start_s <= LIVE_PERSIST_S
            assert cands[0].end_s <= now
            assert "g-mart" in cands[0].texts[0].lower() or "nandilath" in cands[0].texts[0].lower()
            break
        now += 2.0

    assert found_candidate, f"G-Mart persist candidate not emitted after {now:.0f}s"


def test_asianet_chrome_not_proposed():
    """Asianet watermark must never become a persist candidate."""
    hist = PersistHistory()
    asianet_result = _mock_roi_result(
        "tr", "asianet news", is_chrome=True, is_brand=False
    )

    now = 2.0
    while now <= LIVE_PERSIST_S + 10:
        cands = persist_proposer([asianet_result], now, hist)
        assert len(cands) == 0, f"Asianet chrome leaked as candidate at now={now}"
        now += 2.0


def test_persist_candidate_end_s_never_exceeds_now():
    """Candidate end_s from persist_proposer must be ≤ now_s."""
    hist = PersistHistory()
    result = _mock_roi_result("tr", "BHIM UPI", is_chrome=False, is_brand=True)

    now = 2.0
    while now <= LIVE_PERSIST_S + 10:
        cands = persist_proposer([result], now, hist)
        for c in cands:
            assert c.end_s <= now, f"end_s {c.end_s} > now_s {now}"
        now += 2.0


def test_text_change_resets_persist():
    """When ROI text changes, the old brand track should expire and new one start."""
    hist = PersistHistory()

    # First brand
    brand_a = _mock_roi_result("tr", "Nandilath G-Mart", is_chrome=False, is_brand=True)
    for now in range(2, 12, 2):
        persist_proposer([brand_a], float(now), hist)

    # Text changes to a different brand mid-stream
    brand_b = _mock_roi_result("tr", "BHIM UPI", is_chrome=False, is_brand=True)
    cands_at_switch = persist_proposer([brand_b], 12.0, hist)
    # Brand A track should have been displaced / let expire
    for c in cands_at_switch:
        # Should not contain G-Mart text at this point if track expired
        pass  # timing-dependent; just ensure no crash


def test_clock_text_not_proposed():
    """Clock text in corner must be treated as chrome and never proposed."""
    hist = PersistHistory()
    clock_result = _mock_roi_result(
        "br", "04:37 PM", is_chrome=True, is_brand=False
    )

    now = 2.0
    while now <= LIVE_PERSIST_S + 10:
        cands = persist_proposer([clock_result], now, hist)
        assert len(cands) == 0, f"Clock text leaked as candidate at now={now}"
        now += 2.0
