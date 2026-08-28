from app.models.schemas import Evidence, Segment
from app.pipeline.candidates import candidates_from_ocr, union_candidates
from app.pipeline.types import Candidate, OcrHit
from app.pipeline.temporal import merge_adjacent_segments, segment_iou


def test_iou_perfect():
    assert segment_iou(5, 63, 5, 63) == 1.0


def test_iou_partial():
    iou = segment_iou(5, 63, 0, 63)
    assert 0.9 < iou <= 1.0


def test_ocr_fade_loop_merges_to_one_segment():
    hits = [
        OcrHit(5.0, "AVIS VASCULAR CENTER 4500", 5, ["price", "brand_word"]),
        OcrHit(12.0, "AVIS VASCULAR CENTER 4500", 5, ["price", "brand_word"]),
        OcrHit(21.0, "AVIS VASCULAR 4500", 4, ["price", "brand_word"]),
        OcrHit(48.0, "AVIS VASCULAR CENTER", 3, ["brand_word"]),
        OcrHit(60.0, "AVIS 4500", 3, ["price", "brand_word"]),
    ]
    cands = candidates_from_ocr(hits, video_end_s=63.0)
    assert len(cands) == 1
    assert cands[0].start_s == 5.0
    assert cands[0].end_s == 63.0


def test_union_merges_close_candidates():
    a = Candidate(5, 12, "ocr")
    b = Candidate(14, 20, "ocr")
    merged = union_candidates([[a, b]])
    assert len(merged) == 1
    assert merged[0].start_s == 5
    assert merged[0].end_s == 20


def _seg(start: float, end: float, ad_type: str = "self_promo", brand: str = "Loreal") -> Segment:
    return Segment(
        id=f"seg_{start:.0f}",
        start_s=start,
        end_s=end,
        ad_type=ad_type,  # type: ignore[arg-type]
        confidence=0.8,
        brand=brand,
        description="",
        evidence=Evidence(frame_timestamps=[start], signals_used=["asr"]),
    )


def test_merge_adjacent_self_promo():
    a = _seg(36.0, 109.0)
    b = _seg(115.0, 163.0)
    merged = merge_adjacent_segments([a, b], gap_s=15.0)
    assert len(merged) == 1
    assert merged[0].start_s == 36.0
    assert merged[0].end_s == 163.0


def test_merge_does_not_join_different_types():
    a = _seg(0.0, 7.0, ad_type="preroll", brand="")
    b = _seg(10.0, 40.0, ad_type="self_promo", brand="X")
    merged = merge_adjacent_segments([a, b])
    assert len(merged) == 2