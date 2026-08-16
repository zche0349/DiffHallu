"""Bidirectional semantic-equivalence decisions for candidate continuations."""

from __future__ import annotations

from collections.abc import Sequence

from .interfaces import SemanticBackend


_VALID_LABELS = {"entailment", "neutral", "contradiction"}


def normalize_nli_label(label: str) -> str:
    """Normalize and validate a backend-provided NLI label."""

    normalized = str(label).strip().lower()
    if normalized not in _VALID_LABELS:
        raise ValueError(
            "semantic backend must return entailment, neutral, or contradiction; "
            f"received {label!r}"
        )
    return normalized


def are_semantically_equivalent(
    backend: SemanticBackend,
    answer_a: str,
    answer_b: str,
) -> bool:
    """Apply DiffHallu's lax bidirectional NLI equivalence rule."""

    if answer_a == answer_b:
        return True
    if not answer_a or not answer_b:
        return False

    relation_ab = normalize_nli_label(backend.predict(answer_a, answer_b))
    relation_ba = normalize_nli_label(backend.predict(answer_b, answer_a))

    if "contradiction" in {relation_ab, relation_ba}:
        return False
    if relation_ab == relation_ba == "neutral":
        return False
    return True


def build_equivalence_matrix(
    answers: Sequence[str],
    backend: SemanticBackend,
) -> tuple[tuple[bool, ...], ...]:
    """Build a symmetric pairwise semantic-equivalence matrix."""

    size = len(answers)
    matrix = [[False] * size for _ in range(size)]
    for index in range(size):
        matrix[index][index] = True

    for left in range(size):
        for right in range(left + 1, size):
            equivalent = are_semantically_equivalent(
                backend,
                str(answers[left]),
                str(answers[right]),
            )
            matrix[left][right] = equivalent
            matrix[right][left] = equivalent

    return tuple(tuple(row) for row in matrix)
