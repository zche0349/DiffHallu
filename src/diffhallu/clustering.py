"""Order-robust representative clustering used by DiffHallu."""

from __future__ import annotations

from collections.abc import Sequence


def _validate_matrix(matrix: Sequence[Sequence[bool]]) -> None:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("equivalence_matrix must be square")
    for index in range(size):
        if not matrix[index][index]:
            raise ValueError("equivalence_matrix diagonal must be true")
        for other in range(index + 1, size):
            if bool(matrix[index][other]) != bool(matrix[other][index]):
                raise ValueError("equivalence_matrix must be symmetric")


def representative_clustering(
    equivalence_matrix: Sequence[Sequence[bool]],
    order: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Cluster candidates by comparing each unassigned item with a representative."""

    _validate_matrix(equivalence_matrix)
    size = len(equivalence_matrix)
    scan_order = tuple(range(size)) if order is None else tuple(int(i) for i in order)
    if sorted(scan_order) != list(range(size)):
        raise ValueError("order must be a permutation of all candidate indices")

    clusters = [-1] * size
    cluster_id = 0
    for order_position, representative in enumerate(scan_order):
        if clusters[representative] != -1:
            continue
        clusters[representative] = cluster_id
        for candidate in scan_order[order_position + 1 :]:
            if clusters[candidate] == -1 and equivalence_matrix[representative][candidate]:
                clusters[candidate] = cluster_id
        cluster_id += 1

    return tuple(clusters)


def rotated_representative_clusterings(
    equivalence_matrix: Sequence[Sequence[bool]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return ``(order, clusters)`` for every cyclic scan order."""

    _validate_matrix(equivalence_matrix)
    size = len(equivalence_matrix)
    if size == 0:
        return tuple()

    runs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for offset in range(size):
        order = tuple(range(offset, size)) + tuple(range(0, offset))
        clusters = representative_clustering(equivalence_matrix, order)
        runs.append((order, clusters))
    return tuple(runs)
