from __future__ import annotations

from app.config import OCR_SCORE_THRESHOLD, OVERLAY_GAP_S
from app.pipeline.types import Candidate, OcrHit


def candidates_from_ocr(hits: list[OcrHit], video_end_s: float) -> list[Candidate]:
    hot = [h for h in hits if h.score >= OCR_SCORE_THRESHOLD]
    if not hot:
        return []
    hot.sort(key=lambda h: h.timestamp_s)

    def _tokens(text: str) -> set[str]:
        return {w.lower() for w in text.split() if len(w) > 3}

    groups: list[list[OcrHit]] = [[hot[0]]]
    for h in hot[1:]:
        prev = groups[-1][-1]
        close = h.timestamp_s - prev.timestamp_s <= OVERLAY_GAP_S
        related = bool(_tokens(h.text) & _tokens(prev.text))
        if close or related:
            groups[-1].append(h)
        else:
            groups.append([h])

    out: list[Candidate] = []
    for g in groups:
        start = g[0].timestamp_s
        end = g[-1].timestamp_s
        # persistent overlay that is still firing near the tail → extend to file end
        if video_end_s - end <= OVERLAY_GAP_S:
            end = video_end_s
        triggers: list[str] = []
        texts: list[str] = []
        frames: list[float] = []
        score = 0
        for h in g:
            triggers.extend(h.triggers)
            texts.append(h.text)
            frames.append(h.timestamp_s)
            score += h.score
        out.append(
            Candidate(
                start_s=start,
                end_s=end,
                source="ocr",
                raw_triggers=sorted(set(triggers)),
                texts=texts,
                frame_timestamps=frames,
                score_hint=score,
            )
        )
    return out


def union_candidates(groups: list[list[Candidate]]) -> list[Candidate]:
    flat = [c for g in groups for c in g]
    flat.sort(key=lambda c: c.start_s)
    if not flat:
        return []
    merged: list[Candidate] = [flat[0]]
    for c in flat[1:]:
        prev = merged[-1]
        gap = c.start_s - prev.end_s
        overlap = min(prev.end_s, c.end_s) - max(prev.start_s, c.start_s)
        if gap <= OVERLAY_GAP_S or overlap > 0:
            prev.end_s = max(prev.end_s, c.end_s)
            prev.start_s = min(prev.start_s, c.start_s)
            prev.raw_triggers = sorted(set(prev.raw_triggers + c.raw_triggers))
            prev.texts.extend(c.texts)
            prev.frame_timestamps.extend(c.frame_timestamps)
            prev.score_hint += c.score_hint
            if c.source not in prev.source:
                prev.source = f"{prev.source}+{c.source}"
        else:
            merged.append(c)
    return merged


def drop_incidental_ocr(cands: list[Candidate], kind: str, duration_s: float) -> list[Candidate]:
    """R-Incidental: mid-VOD OCR without spoken promo is probably a webpage ad."""
    if kind == "short":
        # Persistent Short overlays (Vista/Avis) are one event; never incidental-drop them.
        return cands
    spoken = [c for c in cands if "asr" in c.source or "intent" in c.source]
    kept: list[Candidate] = []
    for c in cands:
        if "asr" in c.source or "intent" in c.source:
            kept.append(c)
            continue
        # Opening slate: OCR-only block very early in the video (sponsor card / preroll banner).
        # Allow up to the first 8 s so a 5 s "Sponsored by X" frame isn't thrown away.
        opening_slate = c.start_s <= 8.0 and (c.end_s - c.start_s) <= 30.0
        # End-card / closing sponsor: last 20 s of the video.
        ending = c.start_s >= max(0.0, duration_s - 20.0)
        overlaps_asr = any(min(c.end_s, a.end_s) - max(c.start_s, a.start_s) > 0 for a in spoken)
        blob = " ".join(c.texts).lower()
        editorial_page = any(
            w in blob
            for w in ("trump", "election", "voters", "white house", "lok sabha", "census")
        )
        if editorial_page and not overlaps_asr:
            continue
        if opening_slate or ending or overlaps_asr:
            kept.append(c)
    return kept