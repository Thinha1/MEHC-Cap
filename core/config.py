from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class YoloConfig:
    model_name: str = "yolov8l-worldv2.pt"
    confidence_threshold: float = 0.25
    max_objects: int = 20


@dataclass
class ClipScoreConfig:
    model_name: str = "openai/clip-vit-base-patch16"
    device: str | None = None


@dataclass
class GeneratorConfig:
    hf_model_name: str | None = None
    checkpoint_path: Path | None = None
    vocab_path: Path | None = None
    custom_candidates_file: Path | None = None
    max_candidates: int = 5
    num_beams: int = 10
    num_beam_groups: int = 5
    diversity_penalty: float = 0.8
    max_new_tokens: int = 40
    # A dedicated plain (non-diverse) beam search candidate, always included alongside the
    # diverse ones -- see CandidateCaptionGenerator._generate_with_checkpoint. Gives the
    # reranker a "safety net": it can never end up worse than plain beam search would have,
    # since that candidate is always in the pool. 0 disables it (diverse-only, old behavior).
    safe_beam_size: int = 3


@dataclass
class PipelineConfig:
    yolo: YoloConfig
    clipscore: ClipScoreConfig
    generator: GeneratorConfig

    @classmethod
    def defaults(cls) -> "PipelineConfig":
        return cls(
            yolo=YoloConfig(),
            clipscore=ClipScoreConfig(),
            generator=GeneratorConfig(),
        )
