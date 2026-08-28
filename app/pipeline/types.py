from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SampledFrame:
    timestamp_s: float
    path: Path


@dataclass
class OcrHit:
    timestamp_s: float
    text: str
    score: int
    triggers: list[str]


@dataclass
class OcrResult:
    hits: list[OcrHit]
    frames_sampled: int = 0
    frames_ocrd: int = 0
    ocr_wall_s: float = 0.0
    workers: int = 0


@dataclass
class Candidate:
    start_s: float
    end_s: float
    source: str
    raw_triggers: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)
    score_hint: int = 0