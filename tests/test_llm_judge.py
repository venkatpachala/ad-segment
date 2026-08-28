"""LLM is optional: skip OCR-only preroll/overlay; 429 is not a failed run."""
from __future__ import annotations

from app.pipeline.llm_judge import _is_rate_limit, llm_needed
from app.pipeline.types import Candidate


def _ocr(start: float, end: float, score: int = 3) -> Candidate:
    return Candidate(
        start_s=start,
        end_s=end,
        source="ocr",
        raw_triggers=["brand_word", "phone"],
        texts=["Vista Imaging 74169 02233"],
        frame_timestamps=[start],
        score_hint=score,
    )


def test_llm_needed_skips_ocr_preroll():
    assert llm_needed([], "vod") is False
    assert llm_needed([_ocr(0.0, 7.0)], "vod") is False
    assert llm_needed([_ocr(5.5, 63.0)], "short") is False


def test_llm_needed_true_for_asr():
    c = Candidate(20, 80, "asr", ["strong:course"], ["join the program"], [], 4)
    assert llm_needed([c], "vod") is True


def test_llm_needed_true_for_mid_vod_ocr():
    assert llm_needed([_ocr(120.0, 128.0)], "vod") is True


def test_rate_limit_detects_429():
    assert _is_rate_limit(RuntimeError("HTTP Error 429: Too Many Requests"))
    assert _is_rate_limit(RuntimeError("503 Service Unavailable"))
    assert not _is_rate_limit(RuntimeError("HTTP Error 401: Unauthorized"))
