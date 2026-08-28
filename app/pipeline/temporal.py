from __future__ import annotations

from app.models.schemas import Segment


def snap_segments(segments: list[Segment], cuts: list[float], duration_s: float) -> list[Segment]:
    def snap(t: float) -> float:
        if not cuts:
            return t
        nearest = min(cuts, key=lambda c: abs(c - t))
        if abs(nearest - t) <= 0.75:
            return nearest
        return t

    out: list[Segment] = []
    for s in segments:
        start = max(0.0, snap(s.start_s))
        end = min(duration_s, snap(s.end_s))
        if end <= start:
            end = min(duration_s, start + 0.5)
        s.start_s = round(start, 2)
        s.end_s = round(end, 2)
        out.append(s)
    return merge_adjacent_segments(out)


def merge_adjacent_segments(segments: list[Segment], gap_s: float = 15.0) -> list[Segment]:
    """Glue back-to-back same-type pitches (e.g. L'Oréal 0:36–1:49 and 1:55–2:43)."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (s.start_s, s.end_s))
    out: list[Segment] = [segs[0]]
    mergeable = {"self_promo", "midroll_sponsor_read", "other", "affiliate"}
    for s in segs[1:]:
        prev = out[-1]
        close = s.start_s - prev.end_s <= gap_s
        same_type = s.ad_type == prev.ad_type and s.ad_type in mergeable
        same_brand = (not s.brand and not prev.brand) or (s.brand == prev.brand)
        same_pres = s.presentation == prev.presentation
        if close and same_type and same_brand and same_pres:
            prev.end_s = round(max(prev.end_s, s.end_s), 2)
            prev.start_s = round(min(prev.start_s, s.start_s), 2)
            prev.confidence = max(prev.confidence, s.confidence)
            prev.evidence.frame_timestamps = sorted(
                set(prev.evidence.frame_timestamps + s.evidence.frame_timestamps)
            )[:12]
            if s.evidence.transcript_span and s.evidence.transcript_span not in prev.evidence.transcript_span:
                prev.evidence.transcript_span = (
                    (prev.evidence.transcript_span + " " + s.evidence.transcript_span).strip()
                )[:400]
            if s.evidence.transcript_lang and not prev.evidence.transcript_lang:
                prev.evidence.transcript_lang = s.evidence.transcript_lang
            if s.evidence.ocr_text and s.evidence.ocr_text not in prev.evidence.ocr_text:
                prev.evidence.ocr_text = (
                    (prev.evidence.ocr_text + " " + s.evidence.ocr_text).strip()
                )[:500]
            for sig in s.evidence.signals_used:
                if sig not in prev.evidence.signals_used:
                    prev.evidence.signals_used.append(sig)
            if s.brand and not prev.brand:
                prev.brand = s.brand
        else:
            out.append(s)
    return out


def segment_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union