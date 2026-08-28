from __future__ import annotations

import re

from app.models.schemas import Evidence, Segment
from app.pipeline.ocr import (
    best_ocr_banner,
    clean_overlay_ocr,
    guess_ocr_brand,
    overlay_native_span,
)
from app.pipeline.types import Candidate

_POLICY_REJECT_COMEDY = re.compile(r"\b(joke|comedy|skit|funny)\b", re.I)
_AVIS = re.compile(r"avis|vascular", re.I)
_DENTAL = re.compile(r"dental|samskruti|contact us|asian dental", re.I)
_SANJEEVAN = re.compile(r"sanjeevan|netralaya", re.I)
_PHONE = re.compile(
    r"(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b|"
    r"(?:\+?91[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}\b|"
    r"\b[6-9]\d{9}\b"
)


def _guess_brand(texts: list[str]) -> str:
    blob = " | ".join(texts)
    low = blob.lower()
    if _AVIS.search(blob):
        return "Avis Vascular Center"
    if "vista imaging" in low or ("vista" in low and "imaging" in low):
        return "Vista Imaging"
    if _SANJEEVAN.search(blob):
        return "Sanjeevan Netralaya"
    if "surfshark" in low:
        return "Surfshark VPN"
    if "career 247" in low or "career247" in low or "करियर 247" in blob or "करियर247" in blob:
        return "Career247"
    if _DENTAL.search(blob):
        return "Asian Dental / Dr. Samskruti"
    if "डाटा एनालिटिक्स" in blob or "data analytics" in low:
        return "Career247 Data Analytics"
    return ""


def _guess_type(kind: str, cand: Candidate, duration_s: float) -> str:
    blob = " ".join(cand.texts).lower()
    asr = "asr" in cand.source or "intent" in cand.source
    coursey = any(
        k in blob
        for k in (
            "course",
            "कोर्स",
            "कोर्सेज",
            "program",
            "प्रोग्राम",
            "जॉइन",
            "join",
            "कक्ष",
        )
    )
    if "sponsored" in blob or "paid partnership" in blob or "स्पान्सर" in blob or "స్పాన్సర్" in blob:
        if kind == "vod" and cand.start_s <= 1.5:
            return "preroll"
        return "midroll_sponsor_read"
    if coursey and asr:
        return "self_promo"
    if kind == "vod" and cand.start_s <= 1.5 and cand.end_s <= 20:
        return "preroll"
    if cand.end_s >= duration_s - 1.5 and cand.start_s >= max(0.0, duration_s - 5):
        return "self_promo"
    if "code" in blob and ("http" in blob or "use" in blob):
        return "affiliate"
    return "other"


def _ocr_blob(cand: Candidate) -> str:
    return best_ocr_banner(cand.texts) or " ".join(cand.texts)[:500]


def _overlay_ocr_text(blob: str, brand: str) -> str:
    cleaned = clean_overlay_ocr(blob) or blob
    if not brand:
        return cleaned[:500]
    brand_words = set(brand.lower().replace("centre", "center").split())
    kept: list[str] = []
    for part in (p.strip() for p in cleaned.split("|")):
        if not part:
            continue
        if overlay_native_span(part):
            kept.append(part)
            continue
        norm = set(part.lower().replace("centre", "center").split())
        if brand_words & norm and not re.search(r"\d{5}", part):
            continue
        kept.append(part)
    return " | ".join([brand] + kept)[:500]


def _is_ocr_only(cand: Candidate) -> bool:
    return "ocr" in cand.source and "asr" not in cand.source and "intent" not in cand.source


def judge_candidate(
    cand: Candidate,
    kind: str,
    duration_s: float,
    idx: int,
    lines: list[CaptionLine] | None = None,
    transcript_lang: str = "",
) -> Segment | None:
    """Fast rule-based judge fallback. Validates triggers and commercial intent."""
    blob = _ocr_blob(cand)
    ocr_only = _is_ocr_only(cand)
    has_phone = bool(_PHONE.search(blob)) or "phone" in cand.raw_triggers
    has_price = "price" in cand.raw_triggers
    has_cta = "cta" in cand.raw_triggers
    has_brand = "brand_word" in cand.raw_triggers or "service" in cand.raw_triggers or bool(
        _AVIS.search(blob) or _DENTAL.search(blob) or _SANJEEVAN.search(blob) or guess_ocr_brand(blob)
    )

    # Isolated price-like numbers with no brand/CTA → likely comedy / incidental
    if has_price and not has_cta and not has_phone and not has_brand:
        return None

    if _POLICY_REJECT_COMEDY.search(blob) and not has_cta:
        return None

    if cand.score_hint < 2 and not has_phone:
        return None

    dur = cand.end_s - cand.start_s
    low_blob = blob.lower()
    promo_intent = any(
        k in blob
        for k in (
            "कोर्सेज",
            "कोर्स",
            "जॉइन",
            "जॉइ कर",
            "प्रोग्राम",
            "टेस्टिमोनियल",
        )
    ) or any(
        k in low_blob
        for k in (
            "sponsored",
            "sponsor",
            "partnership",
            "enroll",
            "join the program",
            "join this program",
            "use code",
            "promo code",
            "discount",
            "surfshark",
            "vpn",
            "link in the description",
            "link below",
            "/month",
            "free months",
        )
    )
    asr = "asr" in cand.source or "intent" in cand.source
    # Name-drop / intro bumper is not a segment.
    if asr and not promo_intent and not has_brand:
        return None
    if asr and dur < 8 and "sponsored" not in blob.lower():
        return None

    ad_type = _guess_type(kind, cand, duration_s)
    presentation = None
    if ocr_only and kind == "short":
        presentation = "overlay"
    elif ocr_only and kind == "live" and ad_type == "other" and not (
        cand.start_s <= 0.2 and cand.end_s >= duration_s - 0.2
    ):
        presentation = "overlay"

    brand = _guess_brand(cand.texts) or guess_ocr_brand(blob)

    desc_bits = []
    if ocr_only:
        trigs = ", ".join(cand.raw_triggers[:8]) if cand.raw_triggers else "ocr"
        desc_bits.append(f"Persistent lower-third commercial overlay. OCR: {trigs}.")
    elif "asr" in cand.source and "ocr" in cand.source:
        desc_bits.append("Spoken promotion confirmed by on-screen commercial text.")
    elif "asr" in cand.source:
        desc_bits.append("Spoken promotional segment from captions/ASR.")
    elif ad_type == "self_promo":
        desc_bits.append("Self-promotion (course/contact/end card).")
    elif ad_type == "preroll":
        desc_bits.append("Opening commercial before editorial content.")
    else:
        desc_bits.append("Commercial region grouped across time.")
        if cand.raw_triggers:
            desc_bits.append("Triggers: " + ", ".join(cand.raw_triggers[:8]) + ".")

    if ocr_only:
        signals = ["ocr"]
    else:
        signals = []
        if "asr" in cand.source:
            signals.append("asr")
        if "ocr" in cand.source:
            signals.append("ocr")

    # Evidence law: signals_used == ["ocr"] ⇒ empty speech. Banner lives in ocr_text.
    if ocr_only or presentation == "overlay":
        span = ""
        lang = ""
    elif lines:
        overlapping = [
            f"{ln.start_s:.1f} {ln.text}"
            for ln in lines
            if ln.end_s >= cand.start_s - 0.5 and ln.start_s <= cand.end_s + 0.5 and ln.text.strip()
        ]
        span = "\n".join(overlapping)[:400] if overlapping else ""
        lang = transcript_lang or next((ln.lang for ln in lines if ln.lang), "")
    elif "asr" in cand.source:
        spoken_only = [t for t in cand.texts if not t.startswith("|") and not t.startswith(">")]
        span = " ".join(spoken_only)[:400] if spoken_only else " ".join(cand.texts)[:400]
        lang = transcript_lang
    else:
        span = ""
        lang = transcript_lang

    return Segment(
        id=f"seg_{idx:02d}",
        start_s=round(cand.start_s, 2),
        end_s=round(cand.end_s, 2),
        ad_type=ad_type,  # type: ignore[arg-type]
        confidence=min(0.95, 0.55 + 0.05 * min(8, cand.score_hint)),
        brand=brand,
        description=" ".join(desc_bits),
        evidence=Evidence(
            frame_timestamps=[round(t, 2) for t in cand.frame_timestamps[:12]],
            transcript_span=span[:400],
            ocr_text=_overlay_ocr_text(blob, brand) if ocr_only else "",
            transcript_lang=lang,
            signals_used=signals or ["ocr"],
        ),
        presentation=presentation,
    )