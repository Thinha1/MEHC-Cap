from __future__ import annotations

import math
import os

from dotenv import load_dotenv

from clipscore import ClipScorer
from schema import CaptionCandidate, DetectedObject, RankedCaption
from vocab import clean_word, is_object_noun, singularize

load_dotenv()


def _float_env(*names: str, default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            try:
                return float(value)
            except ValueError:
                pass
    return default


def _str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip().lower() if value and value.strip() else default


def _normalize(values: list[float], valid: list[bool] | None = None) -> list[float]:
    """Min-max normalize `values` to [0, 1], using only entries where `valid[i]` is True to
    compute the min/max (entries missing their raw score don't skew the range for everyone
    else). Entries that are invalid, or where all valid values are equal (or there's < 2 of
    them, e.g. a single candidate), get the neutral value 0.5 -- this must not change the
    relative ranking, since it's the same additive contribution for every candidate."""
    if valid is None:
        valid = [True] * len(values)
    usable = [v for v, ok in zip(values, valid) if ok]
    if len(usable) < 2 or max(usable) == min(usable):
        return [0.5] * len(values)
    lo, hi = min(usable), max(usable)
    return [(v - lo) / (hi - lo) if ok else 0.5 for v, ok in zip(values, valid)]


def _belief_alpha(sim: float, confidence: float) -> float:
    """The Jeffrey's-rule revision exponent from Sabir et al., "Belief Revision based Caption
    Re-ranker with Visual Semantic Information" (COLING 2022):

        alpha = [(1 - sim(w, c)) / (1 + sim(w, c))] ** (1 - P(c))

    sim is clamped away from -1 (the ratio's pole) and confidence into [0, 1] defensively --
    real inputs (CLIP cosine similarity, YOLO box confidence) already live in range, this only
    guards against floating-point edge cases."""
    sim = min(max(sim, -0.999999), 0.999999)
    confidence = min(max(confidence, 0.0), 1.0)
    ratio = (1.0 - sim) / (1.0 + sim)
    return ratio ** (1.0 - confidence)


class FormulaReranker:
    """Deterministic composite reranker -- no network call, no LLM. Implements the S_total
    formula from "De xuat Co che Re-ranking Tong hop cho Mo hinh Sinh chu thich anh":

        linear:
            S_total(c) = alpha*S_base(c) + beta*S_align(c) + gamma*S_det(c)

        belief_penalty:
            S_total(c) = S_quality(c) - gamma*max(0, tau - S_det(c))
            S_quality(c) = weighted_mean(alpha*S_base(c), beta*S_align(c))

    Each term is min-max normalized across the K candidates of the SAME image before combining
    (see _normalize) -- the raw scores live on very different scales (log-prob, cosine
    similarity, and probability), so combining them raw would make the weights
    meaningless. alpha/beta/gamma are read from FORMULA_ALPHA/BETA/GAMMA in .env and
    renormalized to sum to 1.

    - S_base: mean per-word log-prob from the checkpoint decoder's own teacher-forcing pass
      (candidate.metadata["token_log_probs"], populated in generate.py). Candidates from a
      different source (no such metadata) get the neutral 0.5 for this term.
    - S_align: candidate.clip_score (raw cosine similarity, already computed).
    - S_det: Belief Revision score, following Sabir et al. (COLING 2022) -- see _s_det.
    """

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        formula_mode: str | None = None,
        belief_threshold: float | None = None,
        clip_scorer: ClipScorer | None = None,
        belief_neg_confidence: float | None = None,
    ) -> None:
        self.alpha = alpha if alpha is not None else _float_env("FORMULA_ALPHA", default=1.0 / 3.0)
        self.beta = beta if beta is not None else _float_env("FORMULA_BETA", default=1.0 / 3.0)
        self.gamma = gamma if gamma is not None else _float_env("FORMULA_GAMMA", default=1.0 / 3.0)
        self.formula_mode = formula_mode or _str_env("FORMULA_MODE", default="linear")
        if self.formula_mode not in {"linear", "belief_penalty"}:
            self.formula_mode = "linear"
        self.belief_threshold = (
            belief_threshold
            if belief_threshold is not None
            else _float_env("FORMULA_BELIEF_THRESHOLD", default=0.5)
        )
        self.belief_threshold = min(max(self.belief_threshold, 0.0), 1.0)
        total = self.alpha + self.beta + self.gamma
        if total > 0:
            self.alpha, self.beta, self.gamma = (
                self.alpha / total,
                self.beta / total,
                self.gamma / total,
            )
        # Used by _s_det to compute sim(w, c) between a caption and a detected object's
        # label (see ClipScorer.text_similarity). None means "no live CLIP model available"
        # (e.g. --rerank-only replay from a cached report) -- _s_det then relies entirely on
        # a cached candidate.metadata["vc_sim"] and degrades to the neutral 0.5 for anything not
        # already cached there.
        self.clip_scorer = clip_scorer
        # Informativeness assigned to the "absence" of an object the caption names but the
        # detector never found at all (as opposed to the paper's negative-evidence method, which
        # uses a *low-confidence* detection as evidence of absence). We have no such confidence
        # value for something never detected, so default to 1.0 confidence-of-absence (i.e.
        # P(c)=0 in the formula below) -- treat "never detected" as maximally informative.
        # Tunable via FORMULA_BELIEF_NEG_CONFIDENCE if this proves too aggressive.
        self.belief_neg_confidence = (
            belief_neg_confidence
            if belief_neg_confidence is not None
            else _float_env("FORMULA_BELIEF_NEG_CONFIDENCE", default=0.0)
        )

    def rerank(
        self,
        candidates: list[CaptionCandidate],
        objects: list[DetectedObject],
        start_rank: int = 1,
    ) -> list[RankedCaption]:
        if not candidates:
            return []

        base_raw, base_ok = self._s_base(candidates)
        align_raw = [c.clip_score if c.clip_score is not None else 0.0 for c in candidates]
        det_raw, det_ok = zip(*(self._s_det(c, objects) for c in candidates))
        s_base = _normalize(base_raw, base_ok)
        s_align = _normalize(align_raw)
        s_det = _normalize(list(det_raw), list(det_ok))

        scored: list[tuple[CaptionCandidate, float, str]] = []
        for i, candidate in enumerate(candidates):
            total, reason_extra = self._combine_scores(s_base[i], s_align[i], s_det[i])
            reason = f"S_base={s_base[i]:.3f} S_align={s_align[i]:.3f} S_det={s_det[i]:.3f}{reason_extra}"
            scored.append((candidate, total, reason))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RankedCaption(caption=candidate, rank=rank, rerank_score=score, reason=reason)
            for rank, (candidate, score, reason) in enumerate(scored, start=start_rank)
        ]

    def _combine_scores(self, s_base: float, s_align: float, s_det: float) -> tuple[float, str]:
        if self.formula_mode == "belief_penalty":
            quality_weight = self.alpha + self.beta
            if quality_weight > 0:
                quality = (self.alpha * s_base + self.beta * s_align) / quality_weight
            else:
                quality = 0.5
            penalty = max(0.0, self.belief_threshold - s_det)
            total = quality - self.gamma * penalty
            return total, f" S_quality={quality:.3f} belief_penalty={penalty:.3f}"

        total = self.alpha * s_base + self.beta * s_align + self.gamma * s_det
        return total, ""

    @staticmethod
    def _s_base(candidates: list[CaptionCandidate]) -> tuple[list[float], list[bool]]:
        raw: list[float] = []
        ok: list[bool] = []
        for candidate in candidates:
            log_probs = (candidate.metadata or {}).get("token_log_probs")
            if log_probs:
                raw.append(sum(log_probs) / len(log_probs))
                ok.append(True)
            else:
                raw.append(0.0)
                ok.append(False)
        return raw, ok

    def _s_det(self, candidate: CaptionCandidate, objects: list[DetectedObject]) -> tuple[float, bool]:
        """Belief Revision score, following Sabir et al., "Belief Revision based Caption
        Re-ranker with Visual Semantic Information" (COLING 2022, arXiv:2209.08163). The
        detected objects are treated as visual evidence that revises the caption's own
        probability via Jeffrey's rule of conditioning:

            P(w|c)  = P(w) ** alpha                      (positive evidence: object c is present)
            P(w|!c) = 1 - (1 - P(w)) ** alpha             (negative evidence: c is named but absent)
            alpha   = [(1 - sim(w,c)) / (1 + sim(w,c))] ** (1 - P(c))     (see _belief_alpha)

        - P(w): the caption's own prior probability. The paper initializes this from an
          external LM (GPT-2); we reuse the geometric mean of the checkpoint decoder's own
          per-word probabilities (candidate.metadata["token_log_probs"], the same source
          _s_base reads) since that IS this project's "common observation" prior.
        - sim(w,c): semantic relatedness between the caption and a detected object's label.
          The paper uses SBERT-STS; we reuse the CLIP text encoder already loaded for S_align
          (see ClipScorer.text_similarity) instead of loading a second embedding model. Cached
          per (candidate, label) into candidate.metadata["vc_sim"] so it round-trips through
          --run-pipeline report JSON and needs no live CLIP model during --rerank-only replay.
        - P(c): the YOLO detector's confidence for that object -- the "informativeness" term.
          A confident detection lets the similarity term dominate the revision; a barely-
          confident one leaves P(w) close to unrevised (alpha -> 1).

        Every detected object is applied in turn as positive evidence, each revision's output
        feeding in as the next one's prior -- Jeffrey's rule chained over multiple pieces of
        evidence, matching how the paper revises against several visual-context labels (VC1,
        VC2, VC3, ...) for the same image.

        Object-noun words the caption names but no detected box matches (the project's
        hallucination signal) are then applied as *negative* evidence via P(w|!c), using the
        word itself as its own visual-context label (sim=1, an exact textual match) and
        self.belief_neg_confidence as P(c) -- see its docstring in __init__ for why "never
        detected at all" isn't quite the paper's low-confidence-detection negative evidence and
        how that's approximated here.

        Falls back to (0.0, False) -- caller applies the neutral 0.5 via _normalize -- when the
        candidate has no token_log_probs, or a similarity is neither cached nor computable
        (no live clip_scorer), so it never gets penalized against candidates that could compute
        every term.
        """
        log_probs = (candidate.metadata or {}).get("token_log_probs")
        if not log_probs:
            return 0.0, False

        # exp(mean log-prob): geometric mean per-word probability, kept strictly inside (0, 1)
        # since both belief-revision formulas divide/exponentiate around 1 - P(w).
        p_w = math.exp(sum(log_probs) / len(log_probs))
        p_w = min(max(p_w, 1e-6), 1.0 - 1e-6)

        detected_heads: set[str] = set()
        for obj in objects:
            sim = self._label_similarity(candidate, obj.label)
            if sim is None:
                return 0.0, False
            p_w = p_w ** _belief_alpha(sim, obj.confidence)
            detected_heads.add(singularize(clean_word(obj.label.split()[-1])))

        for word in candidate.text.split():
            cleaned = clean_word(word)
            if not cleaned or not is_object_noun(word) or singularize(cleaned) in detected_heads:
                continue
            alpha = _belief_alpha(sim=1.0, confidence=self.belief_neg_confidence)
            p_w = 1.0 - (1.0 - p_w) ** alpha

        return p_w, True

    def _label_similarity(self, candidate: CaptionCandidate, label: str) -> float | None:
        cache = candidate.metadata.setdefault("vc_sim", {})
        key = label.lower()
        if key in cache:
            return cache[key]
        if self.clip_scorer is None:
            return None
        sim = self.clip_scorer.text_similarity(candidate.text, label)
        cache[key] = sim
        return sim
