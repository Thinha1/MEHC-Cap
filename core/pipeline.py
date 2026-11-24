from __future__ import annotations

from pathlib import Path

import vocab
from clipscore import ClipScorer
from config import PipelineConfig
from generate import CandidateCaptionGenerator
from formula_reranker import FormulaReranker
from schema import PipelineResult
from yolo_objects import YoloObjectExtractor


class CaptionPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.defaults()
        self.object_extractor = YoloObjectExtractor(
            model_name=self.config.yolo.model_name,
            confidence_threshold=self.config.yolo.confidence_threshold,
            max_objects=self.config.yolo.max_objects,
        )
        # Real class names from the loaded checkpoint (not a hand-typed guess) drive the
        # object-noun vocabulary FormulaReranker uses for S_det/hallucination
        # flagging -- see vocab.set_object_noun_vocab.
        vocab.set_object_noun_vocab(self.object_extractor.class_names())
        self.generator = CandidateCaptionGenerator(
            hf_model_name=self.config.generator.hf_model_name,
            checkpoint_path=self.config.generator.checkpoint_path,
            vocab_path=self.config.generator.vocab_path,
            custom_candidates_file=self.config.generator.custom_candidates_file,
            max_candidates=self.config.generator.max_candidates,
            num_beams=self.config.generator.num_beams,
            num_beam_groups=self.config.generator.num_beam_groups,
            diversity_penalty=self.config.generator.diversity_penalty,
            max_new_tokens=self.config.generator.max_new_tokens,
            safe_beam_size=self.config.generator.safe_beam_size,
        )
        self.clip_scorer = ClipScorer(
            model_name=self.config.clipscore.model_name,
            device=self.config.clipscore.device,
        )
        self.reranker = FormulaReranker(clip_scorer=self.clip_scorer)

    def run(self, image_path: str | Path) -> PipelineResult:
        image_path = Path(image_path)
        objects = self.object_extractor.extract(image_path)
        candidates = self.generator.generate(image_path, objects)
        scored_candidates = self.clip_scorer.score(image_path, candidates)
        ranked = self.reranker.rerank(scored_candidates, objects)
        return PipelineResult(
            image_path=image_path,
            objects=objects,
            candidates=scored_candidates,
            ranked=ranked,
        )
