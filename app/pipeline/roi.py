"""ROI extraction, MAE-based duplicate skip, and chrome/brand filters for live news.

Design
------
Live news L-bar bugs live in the top-right (TR) and bottom-right (BR) corners.
Channel watermarks (Asianet, LIVE, clock) live in the same corners and must be
suppressed *without* a brand keyword list.

Strategy
--------
1. Crop only TR / BR (28 % × 22 % of frame) — skip 94 % of pixels per frame.
2. Downscale crop to 64×36 gray and compare MAE vs the same ROI from the last
   tick.  If MAE < 8 the L-bar has not changed → reuse previous OCR text.
3. Normalize OCR text (lower, strip pure-digit tokens like a clock, collapse
   whitespace).
4. Suppress channel chrome using a small configurable token set.
5. Accept a text as a brand candidate when it is ≥ 4 chars, not chrome, and
   contains at least one Latin token (G-Mart, BHIM, KCL are all Latin on
   Malayalam news channels).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

try:
    import pytesseract
    from PIL import Image as _PILImage

    _TESS_OK: bool | None = None  # lazy check
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    _PILImage = None  # type: ignore[assignment]
    _TESS_OK = False

# normalize_ocr_text is the canonical implementation in ocr.py; imported here
# to avoid duplication. roi.py is always loaded after ocr.py in any pipeline.
from app.pipeline.ocr import normalize_ocr_text  # noqa: E402  (circular-safe; no runtime cycle)


# ---------------------------------------------------------------------------
# ROI definitions (fraction of frame W × H: x1, y1, x2, y2)
# ---------------------------------------------------------------------------

class Roi(NamedTuple):
    name: str
    x1: float  # left edge  (0–1)
    y1: float  # top edge   (0–1)
    x2: float  # right edge (0–1)
    y2: float  # bottom edge (0–1)


ROI_LEFT = Roi("left", 0.00, 0.00, 0.38, 1.00)     # left L-bar column (MS Dhoni, L-bar ads)
ROI_BOTTOM = Roi("bottom", 0.00, 0.75, 1.00, 1.00)  # bottom ticker / banner
ROI_TR = Roi("tr", 0.70, 0.00, 1.00, 0.25)          # top-right   30 % × 25 %
ROI_BR = Roi("br", 0.70, 0.75, 1.00, 1.00)          # bottom-right 30 % × 25 %
LIVE_ROIS: list[Roi] = [ROI_LEFT, ROI_TR, ROI_BR, ROI_BOTTOM]

# ---------------------------------------------------------------------------
# Chrome / channel-watermark token set  (configurable per tenant later)
# ---------------------------------------------------------------------------

CHANNEL_TOKENS: frozenset[str] = frozenset(
    {
        "asianet",
        "asianetnews",
        "asianetnews.com",
        "news",
        "live",
        "friday",
        "saturday",
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "subscribe",
        "watch",
        "stream",
        "breaking",
        "broadcast",
        "channel",
        "update",
        "reporter",
        "newshour",
        "special",
        "overs",
        "runs",
        "wickets",
        "cricket",
        "aug",
        "august",
        "sep",
        "september",
        "oct",
        "october",
        "nov",
        "november",
        "dec",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
    }
)

# MAE below this → static frame → skip Tesseract on this ROI this tick
MAE_SKIP_THRESHOLD: float = 8.0

# Minimum normalized-text length to even consider it commercial
MIN_TEXT_LEN: int = 4

# Thumbnail size for MAE comparison (must match OCR module's OCR_THUMB_SIZE)
_THUMB_W = 64
_THUMB_H = 36

# ---------------------------------------------------------------------------
# Pure-digit / time-like pattern (clock: "04:37 PM", "10:30:05")
# ---------------------------------------------------------------------------
_CLOCK_RE = re.compile(r"^[\d\s:\.apmAPM]+$")
_DIGIT_ONLY_RE = re.compile(r"^\d+$")


# ---------------------------------------------------------------------------
# Tesseract availability (lazy, cached)
# ---------------------------------------------------------------------------

def _tesseract_ok() -> bool:
    global _TESS_OK
    if _TESS_OK is not None:
        return _TESS_OK
    if pytesseract is None:
        _TESS_OK = False
        return False
    try:
        pytesseract.get_tesseract_version()
        _TESS_OK = True
    except Exception:
        _TESS_OK = False
    return _TESS_OK


def _configure_tesseract_path() -> None:
    """Mirror the path-finding logic in ocr.py."""
    import os
    from app.config import ROOT

    _WIN_TESS = [
        str(ROOT / "tesseract" / "tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\venkat\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]
    env = os.getenv("TESSERACT_CMD")
    if env and Path(env).exists():
        pytesseract.pytesseract.tesseract_cmd = env
        return
    for p in _WIN_TESS:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            return


if pytesseract is not None:
    _configure_tesseract_path()

# ---------------------------------------------------------------------------
# Core image helpers
# ---------------------------------------------------------------------------

def crop_roi(frame_bgr: np.ndarray, roi: Roi) -> np.ndarray:
    """Slice a fractional ROI from a BGR frame (numpy view, zero-copy)."""
    h, w = frame_bgr.shape[:2]
    x1 = int(w * roi.x1)
    y1 = int(h * roi.y1)
    x2 = int(w * roi.x2)
    y2 = int(h * roi.y2)
    # Clamp to frame bounds
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return frame_bgr[y1:y2, x1:x2]


def roi_thumb_gray(crop_bgr: np.ndarray) -> np.ndarray:
    """Downscale crop to 64×36 grayscale for MAE comparison."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two same-shape uint8 arrays."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


# ---------------------------------------------------------------------------
# OCR on an in-memory crop (no temp-file write)
# ---------------------------------------------------------------------------

def ocr_crop(crop_bgr: np.ndarray) -> str:
    """Run pytesseract on a BGR numpy array.  Returns joined text or ''."""
    if not _tesseract_ok() or _PILImage is None:
        return ""
    # Upscale small crops for better Tesseract accuracy
    h, w = crop_bgr.shape[:2]
    if max(w, h) < 300:
        crop_bgr = cv2.resize(crop_bgr, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)
    # Convert BGR → RGB for PIL
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = _PILImage.fromarray(rgb)
    try:
        text = pytesseract.image_to_string(pil_img)
    except Exception:
        return ""
    return " ".join(text.split())


# normalize_ocr_text is imported from ocr.py (canonical implementation).
# is_channel_chrome and is_brand_candidate below use the import directly.


def strip_channel_chrome(text: str) -> str:
    """Remove channel watermarks, clocks, and broadcast tickers from text."""
    if not text:
        return ""
    cleaned = text
    # Clock / date / time with OCR typos: LIVE|07:10PM, LIVE|O7:10PM, LIV E]07-10PM, 07:10 PM, AUG 29 2026, SATURDAY
    cleaned = re.sub(r"(?i)\b(?:LIVE\s*[|\]:\s-]*)?[O0-9]{1,2}[:\-.\s][O0-9]{2}(?:\s*[AP]M)?\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\b(?:LIVE|SATURDAY|FRIDAY|SUNDAY|MONDAY|TUESDAY|WEDNESDAY|THURSDAY)\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+[O0-9]{1,2}\s+[O0-9]{4}\b", " ", cleaned)
    # Channel logos and websites
    cleaned = re.sub(r"(?i)\b(?:asianetnews\.com|asianetnews|asianet\s+news|asianet|newshour|eraser)\b", " ", cleaned)
    return " ".join(cleaned.split())


def is_channel_chrome(text: str) -> bool:
    """True if after stripping channel chrome, nothing meaningful remains."""
    stripped = strip_channel_chrome(text)
    norm = normalize_ocr_text(stripped)
    if not norm or len(norm) < MIN_TEXT_LEN:
        return True
    if _CLOCK_RE.match(norm):
        return True
    cleaned = re.sub(r"[^\w\s]", " ", norm)
    tokens = {t for t in cleaned.split() if not t.isdigit()}
    meaningful = {t for t in tokens if len(t) >= 2}
    if not meaningful:
        return True
    return meaningful.issubset(CHANNEL_TOKENS)


# Latin word (ASCII letters only, length ≥ 3) — G-Mart, KCL, BHIM are Latin
_LATIN_RE = re.compile(r"[A-Za-z]{3,}")


def is_brand_candidate(text: str) -> bool:
    """True if text contains non-chrome Latin commercial content."""
    stripped = strip_channel_chrome(text)
    if not _LATIN_RE.search(stripped):
        return False
    norm = normalize_ocr_text(stripped)
    if len(norm) < MIN_TEXT_LEN:
        return False
    return not is_channel_chrome(text)


# ---------------------------------------------------------------------------
# Per-ROI state carried between ticks
# ---------------------------------------------------------------------------

@dataclass
class RoiTickState:
    """State for one ROI across ticks: last thumbnail + last OCR text."""
    name: str
    last_thumb: np.ndarray | None = field(default=None, repr=False)
    last_text: str = ""
    skipped_count: int = 0
    ocr_count: int = 0


# ---------------------------------------------------------------------------
# Per-tick ROI sensor: seek → crop → MAE → optional OCR
# ---------------------------------------------------------------------------

@dataclass
class TickRoiResult:
    roi_name: str
    raw_text: str          # raw OCR output (empty if skipped)
    norm_text: str         # normalize_ocr_text(raw_text)
    is_chrome: bool
    is_brand: bool
    ocr_called: bool       # True if Tesseract was invoked this tick
    mae_value: float       # MAE vs last tick (0.0 if no previous)


def sense_roi(
    frame_bgr: np.ndarray,
    roi: Roi,
    state: RoiTickState,
) -> TickRoiResult:
    """Crop, MAE-check, optionally OCR one ROI from a frame.

    If MAE < MAE_SKIP_THRESHOLD the frame is visually identical to last tick →
    reuse `state.last_text` without calling Tesseract.
    """
    crop = crop_roi(frame_bgr, roi)
    thumb = roi_thumb_gray(crop)

    # --- MAE skip ---
    mae_val = 0.0
    ocr_called = False
    if state.last_thumb is not None:
        mae_val = mae(thumb, state.last_thumb)
        if mae_val < MAE_SKIP_THRESHOLD:
            # Reuse last OCR
            state.skipped_count += 1
            raw = state.last_text
            norm = normalize_ocr_text(raw)
            return TickRoiResult(
                roi_name=roi.name,
                raw_text=raw,
                norm_text=norm,
                is_chrome=is_channel_chrome(raw),
                is_brand=is_brand_candidate(raw),
                ocr_called=False,
                mae_value=mae_val,
            )

    # --- Tesseract ---
    raw = ocr_crop(crop)
    ocr_called = True
    state.last_thumb = thumb
    state.last_text = raw
    state.ocr_count += 1

    norm = normalize_ocr_text(raw)
    return TickRoiResult(
        roi_name=roi.name,
        raw_text=raw,
        norm_text=norm,
        is_chrome=is_channel_chrome(raw),
        is_brand=is_brand_candidate(raw),
        ocr_called=True,
        mae_value=mae_val,
    )


# ---------------------------------------------------------------------------
# Multi-ROI sensor: pick the "best" (most recent, non-static) frame from the
# 4 seek-timestamps and sense each ROI once.
# ---------------------------------------------------------------------------

def sense_all_rois(
    frames_bgr: list[np.ndarray],
    roi_states: dict[str, RoiTickState],
    rois: list[Roi] = LIVE_ROIS,
) -> list[TickRoiResult]:
    """Sense each ROI on the *last* frame in the tick window.

    We take 4 frames per tick (now-1.5, now-1.0, now-0.5, now).  The ROI
    sensor runs on the most recent frame only.  If the MAE skip fires, all
    earlier frames in the same tick are also redundant — no gain from sensing
    multiple frames when the corner is static.

    Returns one TickRoiResult per ROI.
    """
    if not frames_bgr:
        return []
    # Use the last (most recent) frame
    frame = frames_bgr[-1]
    results: list[TickRoiResult] = []
    for roi in rois:
        state = roi_states.setdefault(roi.name, RoiTickState(name=roi.name))
        result = sense_roi(frame, roi, state)
        results.append(result)
    return results
