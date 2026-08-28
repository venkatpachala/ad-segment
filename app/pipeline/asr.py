"""Multilingual ASR layer: captions first, optional Whisper later.

Languages we care about on this assignment: English, Hindi, Telugu, Malayalam,
plus Hinglish/Tanglish romanization that YouTube captions often emit.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.config import ASR_HOP_S, ASR_SCORE_THRESHOLD, ASR_WINDOW_S
from app.pipeline.types import Candidate


@dataclass
class CaptionLine:
    start_s: float
    end_s: float
    text: str
    lang: str = ""


@dataclass
class Transcript:
    """Original-language transcript chosen on ingest (never auto-translate)."""

    language: str = ""
    source: str = "none"  # captions-orig | captions | whisper | none
    lines: list[CaptionLine] = field(default_factory=list)


# --- commercial lexicon: script + romanization. Matching is substring, lowercased. ---
_STRONG = [
    # EN
    "sponsored by",
    "paid partnership",
    "use code",
    "promo code",
    "discount code",
    "affiliate",
    "link in bio",
    "link in description",
    "shop now",
    "buy now",
    "enroll now",
    "enrol now",
    "join this program",
    "join the program",
    "join my course",
    "my course",
    "our course",
    "our courses",
    "this program",
    "this programme",
    "surfshark",
    "surf shark",
    "months for free",
    "extra months",
    "for less than $",
    # HI / Hinglish
    "हमारे कोर्स",
    "हमारे कोर्सेज",
    "कोर्सेज के बाद",
    "जॉइन करिए",
    "जॉइ करिए",
    "जॉइन कीजिए",
    "यह प्रोग्राम",
    "ये प्रोग्राम",
    "इस प्रोग्राम",
    "नामांकन",
    "टेस्टिमोनियल",
    "टेस्टिमोनियल्स",
    "fees",
    "enroll",
    "join kar",
    "hamare course",
    "hamare courses",
    "yeh program",
    "yah program",
    "डाटा एनालिटिक्स कोर्स",
    "data analytics course",
    "data analytics with",
    "join the program",
    "मींस जॉब्स",
    # TE
    "స్పాన్సర్",
    "స్పాన్సర్డ్",
    "కోర్సు",
    "కోర్స్",
    "జాయిన్",
    "ఆఫర్",
    "డిస్కౌంట్",
    "ప్రోమో కోడ్",
    "sponsored",
    # ML
    "സ്പോൺസർ",
    "സ്പോൺസേഡ്",
    "കോഴ്സ്",
    "ജോയിൻ",
    "ഓഫർ",
    "ഡിസ്കൗണ്ട്",
]

_MEDIUM = [
    "course",
    "courses",
    "program",
    "programme",
    "कोर्स",
    "कोर्सेज",
    "प्रोग्राम",
    "जॉइन",
    "जॉइ",
    "ऑफर",
    "डिस्काउंट",
    "कूपन",
    "फीस",
    "testimonial",
    "testimonials",
    "data analytics",
    "डाटा एनालिटिक्स",
    "डेटा एनालिटिक्स",
    "scan now",
    "book now",
    "appointment",
    "patreon",
    "membership",
]


_IDENTITY = (
    "नमस्कार",
    "वेलकम टू",
    "welcome to",
    "मैं हूं",
    "i am",
    "i'm",
)


def score_transcript(text: str) -> tuple[int, list[str]]:
    raw = text or ""
    blob = raw.lower()
    # Channel bumper: "Welcome to Career247 I am Prashant" is identity, not an ad.
    if any(p in raw or p in blob for p in _IDENTITY) and not any(
        p in raw for p in ("कोर्सेज", "कोर्स", "जॉइन", "प्रोग्राम", "sponsored", "enroll")
    ):
        return 0, []
    score = 0
    triggers: list[str] = []
    for phrase in _STRONG:
        if phrase.lower() in blob:
            score += 3
            triggers.append(f"strong:{phrase[:24]}")
    for phrase in _MEDIUM:
        if phrase.lower() in blob:
            score += 1
            triggers.append(f"term:{phrase[:24]}")
    # de-dupe trigger labels
    seen: set[str] = set()
    uniq: list[str] = []
    for t in triggers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return score, uniq


def _parse_ts(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


_VTT_TS = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
)
_TAG = re.compile(r"<[^>]+>")


def parse_vtt(text: str, lang: str = "") -> list[CaptionLine]:
    lines: list[CaptionLine] = []
    chunks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for chunk in chunks:
        m = _VTT_TS.search(chunk)
        if not m:
            continue
        body = _VTT_TS.sub("", chunk)
        body = _TAG.sub("", body)
        body = re.sub(r"align:\S+|position:\S+%|line:\S+", " ", body)
        body = html.unescape(body)
        keep: list[str] = []
        for ln in body.split("\n"):
            s = ln.strip()
            if not s or s.startswith("WEBVTT") or s.startswith("NOTE") or s.isdigit():
                continue
            if "-->" in s:
                continue
            keep.append(s)
        if not keep:
            continue
        lines.append(
            CaptionLine(_parse_ts(m.group(1)), _parse_ts(m.group(2)), " ".join(keep), lang=lang)
        )
    return lines


_CAPTION_LANG = re.compile(r"\.([a-z]{2,3})(?:-orig)?\.vtt$", re.I)


def language_from_caption_path(path: Path) -> str:
    """ISO code from a yt-dlp VTT name (`captions.te-orig.vtt` → `te`)."""
    m = _CAPTION_LANG.search(path.name)
    return m.group(1).lower() if m else ""


def caption_track_rank(path: Path) -> tuple[int, int, str]:
    """Lower is better.

    1. `*-orig` (spoken language; never YouTube auto-translate)
    2. `en` (English video must not pick translated `hi`)
    3. other named tracks
    4. unknown
    """
    name = path.name.lower()
    lang = language_from_caption_path(path)
    is_orig = "-orig." in name or name.endswith("-orig.vtt")
    if is_orig:
        # Prefer non-en orig when both exist (en-orig is often a translation sidecar).
        return (0, 0 if lang != "en" else 1, lang)
    if lang == "en":
        return (1, 0, lang)
    if lang:
        return (2, 0, lang)
    return (3, 0, lang)


def select_caption_track(vtts: list[Path]) -> Path | None:
    if not vtts:
        return None
    return sorted(vtts, key=caption_track_rank)[0]


def whisper_transcribe(video_path: Path, run_dir: Path) -> Transcript:
    """Full transcript via OpenAI Whisper when captions are missing."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or not video_path.exists():
        return Transcript()
    audio = run_dir / "audio.mp3"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "64k",
                str(audio),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except Exception as e:
        (run_dir / "whisper.log").write_text(str(e), encoding="utf-8")
        return Transcript()
    try:
        import urllib.request

        boundary = "----adseg"
        header = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.mp3\"\r\n"
            f"Content-Type: audio/mpeg\r\n\r\n"
        ).encode()
        footer = f"\r\n--{boundary}--\r\n".encode()
        data = header + audio.read_bytes() + footer
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        (run_dir / "whisper.log").write_text(str(e), encoding="utf-8")
        return Transcript()
    lang = str(payload.get("language") or "").strip().lower()
    lines: list[CaptionLine] = []
    for seg in payload.get("segments") or []:
        lines.append(
            CaptionLine(
                float(seg.get("start", 0)),
                float(seg.get("end", 0)),
                str(seg.get("text", "")).strip(),
                lang=lang,
            )
        )
    if not lines and payload.get("text"):
        lines.append(CaptionLine(0.0, 0.0, payload["text"], lang=lang))
    (run_dir / "whisper.language").write_text(lang, encoding="utf-8")
    return Transcript(language=lang, source="whisper", lines=lines)


def _as_transcript(prefetched: Transcript | list[CaptionLine] | None) -> Transcript | None:
    if prefetched is None:
        return None
    if isinstance(prefetched, Transcript):
        return prefetched
    lang = next((ln.lang for ln in prefetched if ln.lang), "")
    return Transcript(language=lang, source="captions", lines=list(prefetched))


def load_transcript(
    url: str | None,
    video_path: Path,
    run_dir: Path,
    prefetched: Transcript | list[CaptionLine] | None = None,
) -> Transcript:
    """Original-language transcript: captions.*.*-orig, else Whisper language, else ISO from name."""
    bundle = _as_transcript(prefetched)
    if bundle is None:
        bundle = fetch_transcript(url, run_dir)
    if bundle.lines:
        print(
            f"INFO: ASR source={bundle.source} lang={bundle.language or '?'} cues={len(bundle.lines)}"
        )
        return Transcript(
            language=bundle.language,
            source=bundle.source,
            lines=_dedupe_rolling(bundle.lines),
        )
    if os.getenv("OPENAI_API_KEY"):
        print("INFO: no captions; running Whisper API on audio")
        whispered = whisper_transcribe(video_path, run_dir)
        if whispered.lines:
            print(
                f"INFO: ASR source=whisper lang={whispered.language or '?'} cues={len(whispered.lines)}"
            )
            return whispered
    print("INFO: ASR source=none")
    return Transcript()


def load_full_transcript(
    url: str | None,
    video_path: Path,
    run_dir: Path,
    prefetched_caps: Transcript | list[CaptionLine] | None = None,
) -> list[CaptionLine]:
    """Full timed transcript: YouTube captions first, Whisper if empty.

    If *prefetched_caps* is supplied (e.g. fetched in parallel with the
    video download) it is used directly and ``fetch_captions`` is skipped.
    """
    return load_transcript(url, video_path, run_dir, prefetched=prefetched_caps).lines


def fetch_transcript(url: str | None, run_dir: Path) -> Transcript:
    if not url or not re.search(r"youtube\.com|youtu\.be", url, re.I):
        return Transcript()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(run_dir / "captions")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--skip-download",
        "--write-auto-sub",
        "--write-sub",
        "--sub-langs",
        ".*-orig,en,hi,te,ta,kn,ml,mr,gu,bn,pa",
        "--convert-subs",
        "vtt",
        "--socket-timeout",
        "10",
        "--retries",
        "1",
        "--extractor-args",
        "youtube:player_client=android,web",
        "-o",
        out_tmpl,
        url,
    ]
    log = run_dir / "yt_dlp_subs.log"
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=25)
        log.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.write_text(str(e), encoding="utf-8")
        return Transcript()
    vtts = sorted(run_dir.glob("*.vtt"))
    chosen = select_caption_track(vtts)
    if chosen is None:
        return Transcript()
    lang = language_from_caption_path(chosen)
    is_orig = "-orig." in chosen.name.lower()
    print(f"INFO: caption track selected: {chosen.name} lang={lang} orig={is_orig}")
    lines = parse_vtt(chosen.read_text(encoding="utf-8", errors="ignore"), lang=lang)
    return Transcript(
        language=lang,
        source="captions-orig" if is_orig else "captions",
        lines=lines,
    )


def fetch_captions(url: str | None, run_dir: Path) -> list[CaptionLine]:
    return fetch_transcript(url, run_dir).lines


def windows_from_lines(lines: list[CaptionLine], duration_s: float) -> list[CaptionLine]:
    if not lines:
        return []
    out: list[CaptionLine] = []
    t = 0.0
    end = max(duration_s, lines[-1].end_s)
    while t < end:
        t1 = t + ASR_WINDOW_S
        text = " ".join(ln.text for ln in lines if ln.start_s < t1 and ln.end_s > t)
        lang = next((ln.lang for ln in lines if ln.lang), "")
        out.append(CaptionLine(t, min(t1, end), text, lang=lang))
        t += ASR_HOP_S
    return out


def _dedupe_rolling(lines: list[CaptionLine]) -> list[CaptionLine]:
    """YouTube auto-VTT repeats the previous phrase. Keep the longest unique cue."""
    out: list[CaptionLine] = []
    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        if out and text in out[-1].text:
            continue
        if out and out[-1].text in text and ln.start_s - out[-1].start_s < 3:
            out[-1] = CaptionLine(out[-1].start_s, ln.end_s, text, lang=ln.lang or out[-1].lang)
            continue
        out.append(CaptionLine(ln.start_s, ln.end_s, text, lang=ln.lang))
    return out


def candidates_from_asr(lines: list[CaptionLine], duration_s: float) -> list[Candidate]:
    """Score individual caption lines so editorial sentences do not enter the pack."""
    hot: list[tuple[CaptionLine, int, list[str]]] = []
    for ln in _dedupe_rolling(lines):
        score, trig = score_transcript(ln.text)
        if score >= ASR_SCORE_THRESHOLD:
            hot.append((ln, score, trig))
    if not hot:
        return []
    groups: list[list[tuple[CaptionLine, int, list[str]]]] = [[hot[0]]]
    for item in hot[1:]:
        prev = groups[-1][-1][0]
        if item[0].start_s <= prev.end_s + 45:
            groups[-1].append(item)
        else:
            groups.append([item])
    def _coursey(trigs: list[str], text: str) -> bool:
        blob = " ".join(trigs) + " " + text
        return any(k in blob.lower() for k in ("कोर्स", "कोर्सेज", "course", "program", "प्रोग्राम", "जॉइन", "join"))

    merged_groups: list[list[tuple[CaptionLine, int, list[str]]]] = []
    for g in groups:
        if merged_groups and _coursey(merged_groups[-1][-1][2], merged_groups[-1][-1][0].text) and _coursey(g[0][2], g[0][0].text):
            if g[0][0].start_s - merged_groups[-1][-1][0].end_s <= 150:
                merged_groups[-1].extend(g)
                continue
        merged_groups.append(g)
    groups = merged_groups

    cands: list[Candidate] = []
    for g in groups:
        texts: list[str] = []
        seen: set[str] = set()
        trigs: list[str] = []
        score = 0
        for ln, s, t in g:
            score += s
            trigs.extend(t)
            key = ln.text[:80]
            if key not in seen:
                seen.add(key)
                texts.append(ln.text)
        cands.append(
            Candidate(
                start_s=g[0][0].start_s,
                end_s=min(g[-1][0].end_s, duration_s),
                source="asr",
                raw_triggers=sorted(set(trigs))[:12],
                texts=texts[:10],
                frame_timestamps=[],
                score_hint=score,
            )
        )
    return cands


def refine_asr_start(cand: Candidate, lines: list[CaptionLine]) -> Candidate:
    """Move start to the first caption line inside the span that scores as commercial."""
    interior = [ln for ln in lines if ln.start_s >= cand.start_s - 1 and ln.start_s <= cand.end_s]
    for ln in interior:
        s, _ = score_transcript(ln.text)
        if s >= ASR_SCORE_THRESHOLD:
            cand.start_s = ln.start_s
            break
    return cand