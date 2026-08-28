"""Generic commercial-intent windows. No product catalog."""

from __future__ import annotations

import math
import re
from collections import Counter

from app.pipeline.asr import CaptionLine
from app.pipeline.types import Candidate

WINDOW_S = 25.0
HOP_S = 12.0
TOP_K = 5
MIN_SCORE = 0.18

# Function words / speech-acts that show up in selling, in any vertical.
_INTENT_HITS = [
    "sponsored",
    "partnership",
    "use code",
    "promo code",
    "link in",
    "link below",
    "click the",
    "try this",
    "try it",
    "use this",
    "buy this",
    "buy now",
    "shop now",
    "order now",
    "enroll",
    "enrol",
    "join now",
    "join this",
    "book now",
    "sign up",
    "download the app",
    "i use this",
    "i've been using",
    "my routine",
    "this product",
    "this range",
    "this kit",
    "this course",
    "this program",
    "this programme",
    "works for",
    "changed my",
    "highly recommend",
    "months for free",
    "extra months",
    "for free",
    "you need to",
    "you should try",
    # HI / Hinglish speech-acts — still not SKUs
    "ट्राई कर",
    "ट्राय कर",
    "जरूर ट्राई",
    "लिंक",
    "लिंक डिस्क्रिप्शन",
    "जॉइन कर",
    "जॉइ कर",
    "खरीद",
    "ऑफर",
    "कोड",
    "यह प्रोडक्ट",
    "ये प्रोडक्ट",
    "यह प्रोग्राम",
    "हमारे कोर्स",
    "रूटिन",
]

_PROMO = [
    "this video is sponsored try this product use the link below and buy it",
    "join this program enroll now fees testimonials use code",
    "i use this product every day here is my routine try this range link in description",
    "book now limited offer discount click the link",
    "यह प्रोडक्ट ट्राई करो लिंक डिस्क्रिप्शन में है जरूर यूज करो",
    "हमारे कोर्सेज जॉइन करिए यह प्रोग्राम टेस्टिमोनियल",
]

_ANTI = [
    "welcome to my channel today we discuss the news hello everyone namaste",
    "नमस्कार वेलकम टू मैं हूं आज हम खबरों के बारे में बात करेंगे",
    "the article says the newspaper reported breaking news election voters",
    "like and subscribe comment below smash the like button",
]


def _tokens(text: str) -> list[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+|[\u0900-\u097F]+|[\u0C00-\u0C7F]+|[\u0D00-\u0D7F]+", text)


def _bow(text: str) -> Counter:
    return Counter(_tokens(text))


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_PROMO_BOW = [_bow(p) for p in _PROMO]
_ANTI_BOW = [_bow(p) for p in _ANTI]


def intent_score(text: str) -> tuple[float, list[str]]:
    blob = (text or "").lower()
    hits = [h for h in _INTENT_HITS if h in blob]
    hit_part = min(1.0, 0.22 * len(hits))
    vec = _bow(text)
    promo = max((_cos(vec, p) for p in _PROMO_BOW), default=0.0)
    anti = max((_cos(vec, p) for p in _ANTI_BOW), default=0.0)
    score = hit_part + promo - 0.7 * anti
    why = hits[:6]
    if promo >= 0.08:
        why.append(f"promo_sim={promo:.2f}")
    if anti >= 0.12:
        why.append(f"anti_sim={anti:.2f}")
    return max(0.0, score), why


def slice_windows(lines: list[CaptionLine], duration_s: float) -> list[CaptionLine]:
    if not lines:
        return []
    end = max(duration_s, max(ln.end_s for ln in lines))
    out: list[CaptionLine] = []
    t = 0.0
    while t < end - 1:
        t1 = min(t + WINDOW_S, end)
        text = " ".join(ln.text for ln in lines if ln.start_s < t1 and ln.end_s > t)
        if text.strip():
            out.append(CaptionLine(t, t1, text))
        t += HOP_S
    return out


def candidates_from_intent(lines: list[CaptionLine], duration_s: float) -> list[Candidate]:
    scored: list[tuple[float, CaptionLine, list[str]]] = []
    for w in slice_windows(lines, duration_s):
        s, why = intent_score(w.text)
        if s >= MIN_SCORE:
            scored.append((s, w, why))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked: list[tuple[float, CaptionLine, list[str]]] = []
    for item in scored:
        if any(abs(item[1].start_s - p[1].start_s) < WINDOW_S * 0.6 for p in picked):
            continue
        picked.append(item)
        if len(picked) >= TOP_K:
            break
    picked.sort(key=lambda x: x[1].start_s)
    cands: list[Candidate] = []
    for s, w, why in picked:
        cands.append(
            Candidate(
                start_s=w.start_s,
                end_s=min(w.end_s, duration_s),
                source="intent",
                raw_triggers=why,
                texts=[w.text[:500]],
                frame_timestamps=[],
                score_hint=max(2, int(s * 10)),
            )
        )
    return cands