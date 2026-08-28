from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from app.config import SHORT_FPS, VOD_BASE_FPS, VOD_OPENING_BURST_S, VOD_OPENING_FPS, LIVE_FPS, VOD_MAX_FRAMES


@dataclass
class SampledFrame:
    timestamp_s: float
    path: Path


def sampling_plan(kind: str, duration_s: float) -> list[float]:
    """Dense frames on short clips. VOD burst + sparse. Live 1 fps.

    Duration wins over the label: a 90s file passed as --kind vod still
    gets Short-density frames so overlays are not skipped.
    """
    if kind == "short" or duration_s <= 180:
        fps = SHORT_FPS
        n = max(1, int(duration_s * fps) + 1)
        return [min(duration_s, i / fps) for i in range(n)]
    if kind == "live":
        fps = LIVE_FPS
        n = max(1, int(duration_s * fps) + 1)
        return [min(duration_s, i / fps) for i in range(n)]

    times: list[float] = []
    t = 0.0
    while t <= duration_s + 1e-6:
        times.append(min(t, duration_s))
        step = 1.0 / VOD_OPENING_FPS if t < VOD_OPENING_BURST_S else 1.0 / VOD_BASE_FPS
        t += step
    # always include t=0 and last frame
    if 0.0 not in times:
        times.insert(0, 0.0)
    if times[-1] < duration_s - 0.05:
        times.append(duration_s)
    times = sorted(set(round(x, 3) for x in times))

    # Cap total frames: keep the full opening burst, uniformly thin the body.
    if len(times) > VOD_MAX_FRAMES:
        opening = [t for t in times if t <= VOD_OPENING_BURST_S]
        body = [t for t in times if t > VOD_OPENING_BURST_S]
        remain = max(1, VOD_MAX_FRAMES - len(opening))
        if body and len(body) > remain:
            step_i = len(body) / remain
            body = [body[int(i * step_i)] for i in range(remain)]
        times = sorted(set(opening + body))

    return times


def extract_frames(video_path: Path, timestamps: list[float], out_dir: Path) -> list[SampledFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[SampledFrame] = []
    for i, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
            ok, img = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{i:04d}_{ts:.2f}s.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frames.append(SampledFrame(ts, path))
    cap.release()
    return frames