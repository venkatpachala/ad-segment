from __future__ import annotations

from pathlib import Path

import cv2


def scene_cuts(video_path: Path, threshold: float = 0.35, max_cuts: int = 80) -> list[float]:
    """Detect scene-change timestamps using seek-based histogram comparison.

    The original implementation called ``cap.read()`` over every frame and
    discarded non-sampled ones — decoding the full stream even at 2 Hz.
    This version seeks directly to each 500 ms sample position so the decoder
    only touches the frames we actually need.

    For a 10-min 30 fps video that is ~1 200 seek-reads vs ~18 000 sequential
    reads — a 10-15× speed improvement.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

    if duration_s <= 0:
        cap.release()
        return []

    # Sample every 500 ms using seeks — 2 Hz resolution is plenty for cut detection.
    sample_interval_s = 0.5
    n_samples = int(duration_s / sample_interval_s) + 1

    prev_hist = None
    cuts: list[float] = []

    for i in range(n_samples):
        t = i * sample_interval_s
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        small = cv2.resize(frame, (64, 36))
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff >= threshold:
                cuts.append(round(t, 3))
                if len(cuts) >= max_cuts:
                    break

        prev_hist = hist

    cap.release()
    return cuts