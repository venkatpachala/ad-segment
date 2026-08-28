from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
GT_PATH = DATA_DIR / "ground_truth.json"
UPLOADS_DIR = DATA_DIR / "uploads"

# Sampling
SHORT_FPS = 2.0
VOD_BASE_FPS = 0.25
VOD_OPENING_BURST_S = 10.0
VOD_OPENING_FPS = 2.0
LIVE_FPS = 1.0

# Max frames passed to OCR for long VODs (opening burst kept whole; body thinned)
VOD_MAX_FRAMES = 200

# Skip scene-cut detection for clips shorter than this (saves 5–20s on short clips)
SCENE_SKIP_UNDER_S = 60.0

# POST /v1/detect waits this long for a sync 200; longer work returns 202 + job_id.
DETECT_SYNC_S = 15.0

# Live HLS: if the buffer duration does not grow for this long, emit `stalled`.
HLS_STALL_S = 12.0

# OCR: pytesseract spawns tesseract.exe per image, so a small thread pool is
# the right primitive. More than 4 workers on a laptop usually thrashes RAM/disk.
OCR_MAX_WORKERS = 4
# Near-duplicate skip: 64x36 grayscale mean-absolute-error vs last *kept* frame.
# ~8 keeps JPEG twins skipped while Avis-style fades still OCR once.
OCR_SKIP_MAE = 8.0
OCR_THUMB_SIZE = (64, 36)


def ocr_worker_count(n_tasks: int | None = None) -> int:
    n = min(OCR_MAX_WORKERS, os.cpu_count() or OCR_MAX_WORKERS)
    if n_tasks is not None:
        n = min(n, max(1, n_tasks))
    return n

# Overlay merge: same commercial text reappearing within this gap is one event
OVERLAY_GAP_S = 16.0

# Minimum commercial score on an OCR frame to propose a candidate
OCR_SCORE_THRESHOLD = 2

# ASR / captions
ASR_WINDOW_S = 20.0
ASR_HOP_S = 8.0
ASR_SCORE_THRESHOLD = 2

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")