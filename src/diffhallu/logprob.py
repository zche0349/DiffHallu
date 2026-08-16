"""Length-normalized generation support used by weighted semantic entropy."""

from __future__ import annotations

import math
from collections.abc import Sequence


def length_normalized_logprob(token_logprobs: Sequence[float]) -> float:
    """Return the arithmetic mean of conditional token log-probabilities."""

    if not token_logprobs:
        raise ValueError("token_logprobs must contain at least one value")

    values = tuple(float(value) for value in token_logprobs)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("token_logprobs must all be finite")
    return sum(values) / len(values)

