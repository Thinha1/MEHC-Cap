from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectedObject:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CaptionCandidate:
    text: str
    source: str
    generation_score: float | None = None
    clip_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedCaption:
    caption: CaptionCandidate
    rank: int
    rerank_score: float
    reason: str


@dataclass(frozen=True)
class PipelineResult:
    image_path: Path
    objects: list[DetectedObject]
    candidates: list[CaptionCandidate]
    ranked: list[RankedCaption]

    @property
    def best_caption(self) -> str:
        if not self.ranked:
            return ""
        return self.ranked[0].caption.text
