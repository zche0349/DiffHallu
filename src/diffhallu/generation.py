"""Candidate generation and teacher-forced support collection."""

from __future__ import annotations

from collections.abc import Sequence

import pysbd

from .config import GenerationConfig
from .interfaces import TargetModelBackend
from .logprob import length_normalized_logprob
from .schemas import Candidate


_DIALOGUE_STOPS = ("\nUser", "\nHuman", "User:", "Human:")


def truncate_first_sentence(text: str) -> str:
    """Return the first generated sentence after removing dialogue spillover."""

    clean = str(text).strip()
    for stop_word in _DIALOGUE_STOPS:
        if stop_word in clean:
            clean = clean.split(stop_word, 1)[0].rstrip()

    if not clean:
        return ""

    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(clean)
    return str(sentences[0]).strip() if sentences else ""


def _validate_generated_count(outputs: Sequence[str], expected: int) -> None:
    if len(outputs) != expected:
        raise ValueError(
            "target model backend returned an unexpected number of generations: "
            f"expected {expected}, received {len(outputs)}"
        )


def generate_candidates(
    backend: TargetModelBackend,
    prompt: str,
    prefix: str,
    config: GenerationConfig,
) -> tuple[Candidate, ...]:
    """Generate, truncate, and score all differential candidates."""

    raw_outputs = tuple(backend.generate(prompt, prefix, config))
    _validate_generated_count(raw_outputs, config.num_samples)

    candidates: list[Candidate] = []
    for index, raw in enumerate(raw_outputs):
        raw_text = str(raw)
        completion = truncate_first_sentence(raw_text)
        if not completion:
            raise ValueError(f"generation {index} is empty after truncation")

        token_logprobs = tuple(
            float(value)
            for value in backend.token_logprobs(prompt, prefix, completion)
        )
        mean_logprob = length_normalized_logprob(token_logprobs)
        candidates.append(
            Candidate(
                completion=completion,
                mean_logprob=mean_logprob,
                token_logprobs=token_logprobs,
                raw_generation=raw_text,
            )
        )

    return tuple(candidates)
