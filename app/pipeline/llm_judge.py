"""LLM is the policy brain. Sensors only retrieve evidence packets."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.config import DATA_DIR
from app.models.schemas import Evidence, Segment
from app.pipeline.asr import CaptionLine
from app.pipeline.ocr import (
    OcrHit,
    best_ocr_banner,
    clean_overlay_ocr,
    guess_ocr_brand,
)
from app.pipeline.types import Candidate

POLICY = """You are the advertising-segment judge for a video detector.

The full video transcript was already scanned by rules. You receive ONLY a
shortlisted candidate: the timed speech inside the span, 20 seconds of speech
before and after, and any OCR text from that span. Do not assume anything
outside this packet.

TASK
Decide whether this candidate is an advertising SEGMENT of the video itself.
If yes, tighten start_s and end_s to the promotional intent, name the brand,
pick ad_type, write a one-sentence description.

DEFINITION
An advertising segment is a contiguous stretch whose PRIMARY PURPOSE is to
commercially promote, recommend, sell, or drive an action toward a product,
service, brand, paid relationship, or the creator's own offering.

IS AN AD
- Spoken pitch for a course/product with join / enroll / fees / testimonials
- Explicit sponsorship or paid partnership
- Opening commercial slate before editorial content
- On-screen offer (price + CTA) that THIS video is delivering
- Persistent branded overlay: hospital/clinic/brand + phone/price/CTA burned into frames.
  Visual ad. Do not invent speech. Hiccups or editorial talk over a Vista/Avis banner is
  NOT the ad — the overlay is. Leave transcript empty for OCR-only overlays.

NOT AN AD
- Channel intro: "Welcome to X, I am Y"
- Channel name said while discussing news
- Creator's own logo / watermark with no phone, price, or CTA
- Ads that appear inside a news article / tweet / webpage the creator is showing
- Comedy or narrative that mentions a price
- Isolated "like and subscribe"

BOUNDARY RULES
- Start when promotional INTENT begins, usually in speech OR when a sponsor overlay first appears
- For a persistent OCR-only overlay (brand + phone throughout the video), use the first frame
  where the overlay appears as start_s and video end as end_s
- Do not start at a later banner if speech already started the pitch
- One continuous pitch = one segment
- A 2-second name-drop is not a segment

TYPES (use one)
preroll | midroll_sponsor_read | self_promo | affiliate | bumper | other

OUTPUT
JSON only, no markdown:
{
  "decision": "AD" | "NON_AD",
  "ad_type": "self_promo" | "midroll_sponsor_read" | "preroll" | "affiliate" | "bumper" | "other" | null,
  "brand": "string",
  "description": "one sentence",
  "start_s": number,
  "end_s": number,
  "confidence": 0.0-1.0,
  "reject_reason": "string or empty"
}
"""


def neighbor_transcript(lines: list[CaptionLine], start_s: float, end_s: float, pad_s: float = 20.0) -> dict:
    before, mid, after = [], [], []
    for ln in lines:
        if ln.end_s < start_s and ln.start_s >= start_s - pad_s:
            before.append(f"{ln.start_s:.1f} {ln.text}")
        elif ln.start_s <= end_s and ln.end_s >= start_s:
            mid.append(f"{ln.start_s:.1f} {ln.text}")
        elif ln.start_s > end_s and ln.start_s <= end_s + pad_s:
            after.append(f"{ln.start_s:.1f} {ln.text}")
    return {
        "before": "\n".join(before[-12:]),
        "span": "\n".join(mid[:40]),
        "after": "\n".join(after[:12]),
    }


_JUDGE_CACHE = DATA_DIR / "cache" / "judge"
_last_http_s = 0.0
MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "3.0"))
_DUMMY_KEYS = {"", "skip", "x-disabled", "none"}


def llm_needed(cands: list[Candidate], kind: str) -> bool:
    """Skip OpenRouter for OCR-only opening slates / phone banners. Save quota for speech."""
    if not cands:
        return False
    for c in cands:
        if "asr" in c.source or "intent" in c.source:
            return True
        if kind == "vod" and c.start_s > 15 and "ocr" in c.source:
            return True  # mid-roll visual vs incidental
    return False


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc)
    code = getattr(exc, "code", None)
    if code in {429, 503}:
        return True
    return "429" in msg or "503" in msg or "Too Many" in msg or "rate limit" in msg.lower()


def _throttle() -> None:
    global _last_http_s
    interval = MIN_INTERVAL_S
    wait = interval - (time.time() - _last_http_s)
    if wait > 0:
        time.sleep(wait)
    _last_http_s = time.time()


def _complete_with_retry(call, tries: int = 4, label: str = "LLM"):
    delay = 1.0
    for i in range(tries):
        try:
            return call()
        except Exception as e:
            if not _is_rate_limit(e) or i == tries - 1:
                raise
            sleep_s = delay + random.random()
            print(f"WARN: {label} 429, retry in {sleep_s:.1f}s ({i + 1}/{tries})")
            time.sleep(sleep_s)
            delay = min(delay * 2, 16)
    raise RuntimeError("retry exhausted")


def _has_key(name: str) -> bool:
    return os.getenv(name, "").strip() not in _DUMMY_KEYS


def _cand_cache_key(cand: Candidate, kind: str) -> str:
    ocr = " ".join(cand.texts)[:400]
    span = ""  # overlay packets have empty speech; asr texts are in cand.texts
    raw = f"{kind}|{cand.source}|{cand.start_s:.2f}|{cand.end_s:.2f}|{ocr}|{span[:200]}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    return _JUDGE_CACHE / f"{key}.json"


def _load_cached_out(cand: Candidate, kind: str) -> dict | None:
    p = _cache_path(_cand_cache_key(cand, kind))
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _store_cached_out(cand: Candidate, kind: str, out: dict) -> None:
    try:
        _JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
        _cache_path(_cand_cache_key(cand, kind)).write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def ocr_in_span(hits: list[OcrHit], start_s: float, end_s: float) -> str:
    rows = []
    for h in hits:
        if start_s - 1 <= h.timestamp_s <= end_s + 1 and h.text:
            rows.append(f"{h.timestamp_s:.1f} score={h.score} {h.text[:220]}")
    return "\n".join(rows[:25])


def _is_ocr_only(cand: Candidate) -> bool:
    return "ocr" in cand.source and "asr" not in cand.source and "intent" not in cand.source


def _ocr_blob(cand: Candidate) -> str:
    return best_ocr_banner(cand.texts) or " ".join(cand.texts)[:500]


def build_packet(
    cand: Candidate,
    kind: str,
    duration_s: float,
    lines: list[CaptionLine],
    hits: list[OcrHit],
) -> dict:
    ocr_only = _is_ocr_only(cand)
    # OCR-only overlays: never send spoken captions (avoids Telugu hiccup ASR in the packet).
    if ocr_only:
        nb = {"before": "", "span": "", "after": ""}
    else:
        nb = neighbor_transcript(lines, cand.start_s, cand.end_s)
    return {
        "kind": kind,
        "video_duration_s": duration_s,
        "candidate": {
            "start_s": cand.start_s,
            "end_s": cand.end_s,
            "source": cand.source,
            "rule_triggers": cand.raw_triggers[:12],
        },
        "transcript_before": "" if ocr_only else nb["before"],
        "transcript_span": "" if ocr_only else nb["span"],
        "transcript_after": "" if ocr_only else nb["after"],
        "ocr_in_span": ocr_in_span(hits, cand.start_s, cand.end_s),
    }


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 45) -> dict:
    _throttle()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:240]
        except Exception:
            pass
        raise RuntimeError(f"HTTP Error {e.code}: {e.reason} {body}".strip()) from e


def _parse_model_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            return json.loads(m.group(1))
        raise


def call_openai(packet: dict) -> dict:
    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": POLICY},
            {
                "role": "user",
                "content": (
                    "Decide if this candidate is an advertising segment.\n"
                    "Return JSON: {decision: AD|NON_AD, ad_type: preroll|midroll_sponsor_read|"
                    "self_promo|affiliate|bumper|other|null, brand, description, start_s, end_s, "
                    "confidence, reject_reason}.\n\n"
                    + json.dumps(packet, ensure_ascii=False)
                ),
            },
        ],
    }
    raw = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        body,
    )
    return _parse_model_json(raw["choices"][0]["message"]["content"])


def call_gemini(packet: dict) -> dict:
    key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    )
    body = {
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        "contents": [
            {
                "parts": [
                    {
                        "text": POLICY
                        + "\nReturn JSON: {decision, ad_type, brand, description, start_s, end_s, confidence, reject_reason}\n\n"
                        + json.dumps(packet, ensure_ascii=False)
                    }
                ]
            }
        ],
    }
    raw = _post_json(url, {"Content-Type": "application/json"}, body)
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_model_json(text)


def call_openrouter(packet: dict) -> dict:
    key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2:free")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": POLICY},
            {
                "role": "user",
                "content": "Candidate packet:\n" + json.dumps(packet, ensure_ascii=False),
            },
        ],
    }
    raw = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ad-segment-detector",
        },
        body,
    )
    return _parse_model_json(raw["choices"][0]["message"]["content"])


def llm_available() -> str | None:
    if _has_key("OPENROUTER_API_KEY"):
        return "openrouter"
    if _has_key("OPENAI_API_KEY"):
        return "openai"
    if _has_key("GEMINI_API_KEY"):
        return "gemini"
    return None


def _extract_transcript_span(
    lines: list[CaptionLine] | None,
    start_s: float,
    end_s: float,
    max_len: int = 500,
) -> str:
    if not lines:
        return ""
    # Collect all timed caption lines overlapping [start_s, end_s]
    overlapping = [
        f"{ln.start_s:.1f} {ln.text}"
        for ln in lines
        if ln.end_s >= start_s - 0.5 and ln.start_s <= end_s + 0.5 and ln.text.strip()
    ]
    if overlapping:
        return "\n".join(overlapping)[:max_len]
    return ""


def _format_segment(
    out: dict,
    cand: Candidate,
    duration_s: float,
    idx: int,
    lines: list[CaptionLine] | None = None,
    transcript_lang: str = "",
    kind: str = "vod",
) -> Segment | None:
    if str(out.get("decision", "")).upper() != "AD":
        return None
    start = float(out.get("start_s", cand.start_s))
    end = float(out.get("end_s", cand.end_s))
    start = max(0.0, min(start, duration_s))
    end = max(start + 0.3, min(end, duration_s))
    ad_type = out.get("ad_type") or "other"
    allowed = {
        "preroll",
        "midroll_sponsor_read",
        "product_placement",
        "self_promo",
        "affiliate",
        "platform_inserted",
        "bumper",
        "other",
    }
    if ad_type not in allowed:
        ad_type = "other"
    ocr_only = _is_ocr_only(cand)
    if ocr_only:
        signals = ["ocr"]
        raw = _ocr_blob(cand)
        spoken_span = ""
        brand = (str(out.get("brand") or "")[:80]) or guess_ocr_brand(raw)
        ocr_text = clean_overlay_ocr(raw) or raw
        if brand and brand.split()[0].lower() not in ocr_text.lower():
            ocr_text = f"{brand} | {ocr_text}" if ocr_text else brand
        lang = ""
        presentation = "overlay" if kind == "short" else None
        desc = str(out.get("description") or "").strip()
        if not desc:
            trigs = ", ".join(cand.raw_triggers[:8]) if cand.raw_triggers else "ocr"
            desc = f"Persistent lower-third commercial overlay. OCR: {trigs}."
    else:
        signals = []
        if "asr" in cand.source:
            signals.append("asr")
        if "intent" in cand.source:
            signals.append("intent")
        if "ocr" in cand.source:
            signals.append("ocr")
        signals.append("llm")
        spoken_span = _extract_transcript_span(lines, start, end)
        ocr_text = ""
        lang = transcript_lang or next((ln.lang for ln in (lines or []) if ln.lang), "")
        presentation = None
        brand = str(out.get("brand") or "")[:80]
        desc = str(out.get("description") or "LLM policy decision.")[:400]

    return Segment(
        id=f"seg_{idx:02d}",
        start_s=round(start, 2),
        end_s=round(end, 2),
        ad_type=ad_type,
        confidence=float(out.get("confidence") or 0.8),
        brand=brand,
        description=desc[:400],
        evidence=Evidence(
            frame_timestamps=[f for f in cand.frame_timestamps if start - 1 <= f <= end + 1],
            transcript_span=spoken_span,
            ocr_text=ocr_text,
            transcript_lang=lang,
            signals_used=signals,
        ),
        presentation=presentation,
    )


def _providers_single(packet: dict) -> dict:
    """OpenRouter (retries) → OpenAI gpt-4o-mini → caller uses rule judge."""
    last: Exception | None = None
    if _has_key("OPENROUTER_API_KEY"):
        try:
            return _complete_with_retry(lambda: call_openrouter(packet), label="OpenRouter")
        except Exception as e:
            last = e
            if _has_key("OPENAI_API_KEY"):
                why = "429" if _is_rate_limit(e) else str(e)[:80]
                print(f"WARN: OpenRouter failed ({why}), falling back to OpenAI gpt-4o-mini")
            elif _is_rate_limit(e):
                print("WARN: OpenRouter 429, using rule judge")
                raise
            else:
                raise
    if _has_key("OPENAI_API_KEY"):
        try:
            return _complete_with_retry(lambda: call_openai(packet), label="OpenAI")
        except Exception as e:
            last = e
            if _is_rate_limit(e):
                print("WARN: OpenAI 429, using rule judge")
            else:
                print(f"WARN: OpenAI failed ({e}); using rule judge")
            raise
    raise last or RuntimeError("no LLM key")


def _providers_batch(batch_payload: dict) -> dict:
    """OpenRouter (retries) → OpenAI gpt-4o-mini → caller uses rule judge."""
    last: Exception | None = None
    if _has_key("OPENROUTER_API_KEY"):
        try:
            return _complete_with_retry(
                lambda: call_openrouter_batch(batch_payload), label="OpenRouter"
            )
        except Exception as e:
            last = e
            if _has_key("OPENAI_API_KEY"):
                why = "429" if _is_rate_limit(e) else str(e)[:80]
                print(f"WARN: OpenRouter failed ({why}), falling back to OpenAI gpt-4o-mini")
            elif _is_rate_limit(e):
                print("WARN: OpenRouter 429, using rule judge")
                raise
            else:
                raise
    if _has_key("OPENAI_API_KEY"):
        try:
            return _complete_with_retry(
                lambda: call_openai_batch(batch_payload), label="OpenAI"
            )
        except Exception as e:
            last = e
            if _is_rate_limit(e):
                print("WARN: OpenAI 429, using rule judge")
            else:
                print(f"WARN: OpenAI failed ({e}); using rule judge")
            raise
    raise last or RuntimeError("no LLM key")


def judge_with_llm(
    cand: Candidate,
    kind: str,
    duration_s: float,
    idx: int,
    lines: list[CaptionLine],
    hits: list[OcrHit],
) -> Segment | None:
    if not llm_needed([cand], kind):
        from app.pipeline.judge import judge_candidate

        return judge_candidate(cand, kind, duration_s, idx, lines)
    cached = _load_cached_out(cand, kind)
    if cached is not None:
        return _format_segment(cached, cand, duration_s, idx, lines=lines, kind=kind)
    packet = build_packet(cand, kind, duration_s, lines, hits)
    out = _providers_single(packet)
    _store_cached_out(cand, kind, out)
    return _format_segment(out, cand, duration_s, idx, lines=lines, kind=kind)


BATCH_POLICY = (
    POLICY
    + """
BATCH MODE
You receive an object with "candidates": [ {"index": 1, ...packet...}, ... ].
Decide each candidate independently.
Return JSON ONLY:
{
  "judgments": [
    {
      "index": 1,
      "decision": "AD" | "NON_AD",
      "ad_type": "self_promo" | "midroll_sponsor_read" | "preroll" | "affiliate" | "bumper" | "other" | null,
      "brand": "string",
      "description": "one sentence",
      "start_s": number,
      "end_s": number,
      "confidence": 0.0-1.0,
      "reject_reason": "string or empty"
    }
  ]
}
"""
)


def call_openrouter_batch(batch_payload: dict) -> dict:
    key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2:free")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": BATCH_POLICY},
            {
                "role": "user",
                "content": "Candidate batch:\n" + json.dumps(batch_payload, ensure_ascii=False),
            },
        ],
    }
    raw = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ad-segment-detector",
        },
        body,
    )
    return _parse_model_json(raw["choices"][0]["message"]["content"])


def call_openai_batch(batch_payload: dict) -> dict:
    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": BATCH_POLICY},
            {
                "role": "user",
                "content": "Candidate batch:\n" + json.dumps(batch_payload, ensure_ascii=False),
            },
        ],
    }
    raw = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        body,
    )
    return _parse_model_json(raw["choices"][0]["message"]["content"])


def call_gemini_batch(batch_payload: dict) -> dict:
    key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    )
    body = {
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        "contents": [
            {
                "parts": [
                    {
                        "text": BATCH_POLICY
                        + "\nReturn JSON: {\"judgments\": [...]}\n\n"
                        + json.dumps(batch_payload, ensure_ascii=False)
                    }
                ]
            }
        ],
    }
    raw = _post_json(url, {"Content-Type": "application/json"}, body)
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_model_json(text)


def _rule_segments(
    cands: list[Candidate],
    kind: str,
    duration_s: float,
    lines: list[CaptionLine],
    transcript_lang: str,
) -> list[Segment]:
    from app.pipeline.judge import judge_candidate

    segments: list[Segment] = []
    for c in cands:
        seg = judge_candidate(
            c, kind, duration_s, len(segments) + 1, lines, transcript_lang=transcript_lang
        )
        if seg:
            segments.append(seg)
    return segments


def judge_candidates_batch(
    cands: list[Candidate],
    kind: str,
    duration_s: float,
    lines: list[CaptionLine],
    hits: list[OcrHit],
    transcript_lang: str = "",
) -> tuple[list[Segment], int, str]:
    """Judge candidates. LLM is optional.

    Returns (segments, model_calls, llm_fallback) where llm_fallback is
    "" | "skipped" | "cache" | "429" | "error".
    """
    if not cands:
        return [], 0, ""

    if not llm_needed(cands, kind):
        print("INFO: llm skipped (ocr-only preroll, rules sufficient)")
        return _rule_segments(cands, kind, duration_s, lines, transcript_lang), 0, "skipped"

    cached_hits = [_load_cached_out(c, kind) for c in cands]
    if all(h is not None for h in cached_hits):
        print("INFO: llm cache hit, 0 model calls")
        segments: list[Segment] = []
        for c, out in zip(cands, cached_hits, strict=True):
            assert out is not None
            seg = _format_segment(
                out, c, duration_s, len(segments) + 1, lines=lines, transcript_lang=transcript_lang, kind=kind
            )
            if seg:
                segments.append(seg)
        return segments, 0, "cache"

    if not llm_available():
        return _rule_segments(cands, kind, duration_s, lines, transcript_lang), 0, "skipped"

    packets = [
        {"index": i + 1, "packet": build_packet(c, kind, duration_s, lines, hits)}
        for i, c in enumerate(cands)
    ]
    batch_payload = {"candidates": packets}

    try:
        res = _providers_batch(batch_payload)
        judgments = res.get("judgments") or res.get("candidates") or []
        j_by_index = {j.get("index"): j for j in judgments if isinstance(j, dict)}
        segments = []
        for i, c in enumerate(cands, start=1):
            out = j_by_index.get(i)
            if out:
                _store_cached_out(c, kind, out)
                seg = _format_segment(
                    out,
                    c,
                    duration_s,
                    len(segments) + 1,
                    lines=lines,
                    transcript_lang=transcript_lang,
                    kind=kind,
                )
                if seg:
                    segments.append(seg)
        return segments, 1, ""
    except Exception as e:
        reason = "429" if _is_rate_limit(e) else "error"
        if reason != "429":
            print(f"WARN: batch LLM judging failed ({e}); falling back to rule judge")
        return _rule_segments(cands, kind, duration_s, lines, transcript_lang), 0, reason