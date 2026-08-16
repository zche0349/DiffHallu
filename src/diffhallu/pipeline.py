"""End-to-end, model-agnostic DiffHallu pipeline."""

from __future__ import annotations

from .clustering import rotated_representative_clusterings
from .config import DiffHalluConfig
from .entropy import average_weighted_semantic_entropy
from .generation import generate_candidates
from .interfaces import RewriterBackend, SemanticBackend, TargetModelBackend
from .nli import build_equivalence_matrix
from .prefix import locate_target_context, prepare_prefix
from .rewriter import rewrite_target_sentence
from .schemas import DetectionResult


class DiffHallu:
    """Detect hallucination risk by differential candidate generation."""

    def __init__(
        self,
        *,
        rewriter: RewriterBackend,
        target_model: TargetModelBackend,
        semantic_model: SemanticBackend,
        config: DiffHalluConfig | None = None,
    ) -> None:
        self.rewriter = rewriter
        self.target_model = target_model
        self.semantic_model = semantic_model
        self.config = config or DiffHalluConfig()

    def detect(
        self,
        *,
        prompt: str,
        generated_text: str,
        target_entity: str,
        char_start: int | None = None,
        char_end: int | None = None,
        occurrence_index: int = 0,
    ) -> DetectionResult:
        """Return the continuous DiffHallu risk score for one target entity."""

        context = locate_target_context(
            prompt,
            generated_text,
            target_entity,
            char_start=char_start,
            char_end=char_end,
            occurrence_index=occurrence_index,
        )
        rewrite = rewrite_target_sentence(
            self.rewriter,
            context,
            self.config.rewrite,
        )
        prepared = prepare_prefix(context, rewrite)

        candidates = generate_candidates(
            self.target_model,
            prompt,
            prepared.sampling_prefix,
            self.config.generation,
        )
        answers = tuple(candidate.completion for candidate in candidates)
        mean_logprobs = tuple(candidate.mean_logprob for candidate in candidates)

        equivalence_matrix = build_equivalence_matrix(
            answers,
            self.semantic_model,
        )
        clusterings = rotated_representative_clusterings(equivalence_matrix)
        risk_score, cluster_runs = average_weighted_semantic_entropy(
            clusterings,
            mean_logprobs,
        )

        return DetectionResult(
            target_entity=target_entity,
            entity_start=context.entity_start,
            entity_end=context.entity_end,
            source_sentence=prepared.source_sentence,
            rewritten_sentence=prepared.rewritten_sentence,
            cloze_sentence=prepared.cloze_sentence,
            sampling_prefix=prepared.sampling_prefix,
            candidates=candidates,
            equivalence_matrix=equivalence_matrix,
            cluster_runs=cluster_runs,
            risk_score=risk_score,
        )
