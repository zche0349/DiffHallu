"""Typed inputs and outputs for the DiffHallu method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewriteResult:
    """A target sentence rewritten so that the entity occupies the final slot."""

    rewritten_sentence: str
    cloze_sentence: str


@dataclass(frozen=True)
class PrefixContext:
    """Text fragments used to construct the controlled generation prefix."""

    prompt: str
    generated_text: str
    target_entity: str
    entity_start: int
    entity_end: int
    source_sentence: str
    context_left: str


@dataclass(frozen=True)
class PreparedPrefix:
    """Validated rewrite and the prefix supplied to the target model."""

    source_sentence: str
    rewritten_sentence: str
    cloze_sentence: str
    full_prefix: str
    sampling_prefix: str


@dataclass(frozen=True)
class Candidate:
    """One sampled continuation and its length-normalized generation support."""

    completion: str
    mean_logprob: float
    token_logprobs: tuple[float, ...]
    raw_generation: str


@dataclass(frozen=True)
class ClusterRun:
    """One representative-clustering scan and its weighted semantic entropy."""

    order: tuple[int, ...]
    clusters: tuple[int, ...]
    cluster_weights: dict[int, float]
    entropy: float


@dataclass(frozen=True)
class DetectionResult:
    """Complete method output for one target entity."""

    target_entity: str
    entity_start: int
    entity_end: int
    source_sentence: str
    rewritten_sentence: str
    cloze_sentence: str
    sampling_prefix: str
    candidates: tuple[Candidate, ...]
    equivalence_matrix: tuple[tuple[bool, ...], ...]
    cluster_runs: tuple[ClusterRun, ...]
    risk_score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

