"""Weighted semantic entropy over representative clusterings."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .schemas import ClusterRun


def logsumexp(values: Sequence[float]) -> float:
    """Compute log(sum(exp(values))) without numerical overflow."""

    if not values:
        raise ValueError("logsumexp requires at least one value")
    floats = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in floats):
        raise ValueError("logsumexp values must be finite")
    maximum = max(floats)
    return maximum + math.log(sum(math.exp(value - maximum) for value in floats))


def semantic_cluster_weights(
    clusters: Sequence[int],
    mean_logprobs: Sequence[float],
) -> dict[int, float]:
    """Aggregate normalized generation support for every semantic cluster."""

    if len(clusters) != len(mean_logprobs):
        raise ValueError("clusters and mean_logprobs must have equal length")
    if not clusters:
        return {}

    scores = tuple(float(value) for value in mean_logprobs)
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("mean_logprobs must all be finite")

    denominator = logsumexp(scores)
    grouped: dict[int, list[float]] = {}
    for cluster_id, score in zip(clusters, scores):
        grouped.setdefault(int(cluster_id), []).append(score)

    weights = {
        cluster_id: math.exp(logsumexp(cluster_scores) - denominator)
        for cluster_id, cluster_scores in sorted(grouped.items())
    }

    total = sum(weights.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("semantic cluster weights could not be normalized")
    return {cluster_id: weight / total for cluster_id, weight in weights.items()}


def weighted_semantic_entropy(cluster_weights: dict[int, float]) -> float:
    """Compute negative sum(q * log(q)) for a semantic-cluster distribution."""

    entropy = 0.0
    for weight in cluster_weights.values():
        if not 0 < weight <= 1:
            raise ValueError("each semantic cluster weight must be in (0, 1]")
        entropy -= weight * math.log(weight)
    return float(entropy)


def average_weighted_semantic_entropy(
    clustering_runs: Sequence[tuple[Sequence[int], Sequence[int]]],
    mean_logprobs: Sequence[float],
) -> tuple[float, tuple[ClusterRun, ...]]:
    """Compute each scan's entropy and return their arithmetic mean."""

    if not clustering_runs:
        if mean_logprobs:
            raise ValueError("at least one clustering run is required")
        return 0.0, tuple()

    runs: list[ClusterRun] = []
    for order, clusters in clustering_runs:
        weights = semantic_cluster_weights(clusters, mean_logprobs)
        entropy = weighted_semantic_entropy(weights)
        runs.append(
            ClusterRun(
                order=tuple(int(index) for index in order),
                clusters=tuple(int(cluster_id) for cluster_id in clusters),
                cluster_weights=weights,
                entropy=entropy,
            )
        )

    score = sum(run.entropy for run in runs) / len(runs)
    return float(score), tuple(runs)

