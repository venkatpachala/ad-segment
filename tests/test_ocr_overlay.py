"""OCR-only overlay evidence: Vista Short. No video, no Whisper."""
from __future__ import annotations

from app.pipeline.asr import CaptionLine
from app.pipeline.judge import judge_candidate
from app.pipeline.ocr import (
    _score,
    clean_overlay_ocr,
    extract_phones,
    guess_ocr_brand,
    overlay_native_span,
)
from app.pipeline.types import Candidate


def test_vista_ocr_brand():
    t = "Single Heart beat 640Slice CT Scan With AI Powered 74169 02233"
    score, trig = _score(t)
    assert score >= 2
    assert "phone" in trig or "service" in trig
    assert guess_ocr_brand(t) == "Vista Imaging"


def test_truncated_phone_still_scores_with_scan():
    t = "640Slice CT Scan With AI Powered 74169 0223"
    score, _ = _score(t)
    assert score >= 2


def test_overlay_evidence_strips_asr():
    t = "Single Heart beat 640Slice CT Scan With AI Powered 74169 02233"
    cand = Candidate(
        start_s=7.5,
        end_s=74.9,
        source="ocr",
        raw_triggers=["phone", "service"],
        texts=[t],
        frame_timestamps=[7.5, 8.0],
        score_hint=4,
    )
    seg = judge_candidate(
        cand,
        kind="short",
        duration_s=75,
        idx=1,
        lines=[CaptionLine(8, 12, "ఇక్కీలు పొట్టా పెద్దికా")],
    )
    assert seg is not None
    assert seg.evidence.transcript_span == ""
    assert "640Slice" in seg.evidence.ocr_text
    assert seg.brand == "Vista Imaging"
    assert seg.presentation == "overlay"
    assert seg.evidence.signals_used == ["ocr"]
    assert "ఇక్కీలు" not in (seg.evidence.transcript_span or "")
    assert "ఇక్కీలు" not in (seg.evidence.ocr_text or "")


def test_72000_is_not_price():
    score, trig = _score("ASe%) Dotos) 100% 72,000!")
    assert "price" not in trig
    assert score < 2
    score2, trig2 = _score("OS $085 640Slice CT Scan With AI Powered 74169 02233")
    assert "price" not in trig2
    assert score2 >= 2  # phone/service still fire


def test_clean_avis_banner_not_soup():
    raw = ". = 2 <i el Show aa 8o HD Hd Some? $e Soden SenwSb06! - ‘ ax Avis Vascular Centre | CEs) @ i 489297 21641"
    cleaned = clean_overlay_ocr(raw)
    assert "Avis Vascular" in cleaned
    assert "Show" not in cleaned
    assert "Soden" not in cleaned
    assert "Hd Some" not in cleaned
    phones = extract_phones(raw)
    assert phones
    assert phones[0].replace(" ", "") == "8929721641"


def test_overlay_speech_empty_banner_in_ocr():
    """signals_used == [ocr] ⇒ empty transcript_span. Hiccup captions never appear."""
    blob = (
        "కడుపు ఉబ్బరం పదే పదే వస్తుందా? ఈ కారణాలు తెలుసుకోండి! "
        "Avis Vascular Centre 4500/- 89297 21641"
    )
    cand = Candidate(
        start_s=5.5,
        end_s=65.9,
        source="ocr",
        raw_triggers=["brand_word", "phone"],
        texts=[blob],
        frame_timestamps=[5.5, 6.0],
        score_hint=4,
    )
    seg = judge_candidate(
        cand,
        kind="short",
        duration_s=66,
        idx=1,
        lines=[CaptionLine(8, 12, "ఫ్లాట్ లెన్స్ అని అంటాము")],
    )
    assert seg is not None
    assert seg.evidence.signals_used == ["ocr"]
    assert seg.evidence.transcript_span == ""
    assert seg.evidence.transcript_lang == ""
    assert "ఫ్లాట్ లెన్స్" not in (seg.evidence.ocr_text or "")
    assert "Avis Vascular" in seg.evidence.ocr_text
    assert "89297" in seg.evidence.ocr_text
    # On-screen Telugu may live in ocr_text (pixels), never as speech evidence.
    assert overlay_native_span(blob).startswith("కడుపు")


def test_ocr_junk_caps_are_not_the_brand():
    t = "os 2% dots) Hae0n SBeCYATO 100% With AI Powered 74169 02233 640Slice CT Scan"
    assert guess_ocr_brand(t) == "Vista Imaging"
    t2 = "8% Dot HAoca Seowe? 640Slice CT Scan With AI Powered 74169 02233"
    assert guess_ocr_brand(t2) == "Vista Imaging"
