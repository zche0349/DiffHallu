"""Target localization and controlled-prefix construction."""

from __future__ import annotations

from typing import Iterable

import pysbd

from .schemas import PrefixContext, PreparedPrefix, RewriteResult


BLANK_TOKEN = "[BLANK]"


def _sentence_spans(text: str) -> Iterable[tuple[int, int, str]]:
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    for span in segmenter.segment(text):
        yield int(span.start), int(span.end), str(span.sent)


def _resolve_entity_span(
    generated_text: str,
    target_entity: str,
    char_start: int | None,
    char_end: int | None,
    occurrence_index: int,
) -> tuple[int, int]:
    if not target_entity:
        raise ValueError("target_entity must not be empty")

    if char_start is not None or char_end is not None:
        if char_start is None or char_end is None:
            raise ValueError("char_start and char_end must be provided together")
        if not 0 <= char_start < char_end <= len(generated_text):
            raise ValueError("the target character span is outside generated_text")
        if generated_text[char_start:char_end] != target_entity:
            raise ValueError("the supplied character span does not match target_entity")
        return char_start, char_end

    if occurrence_index < 0:
        raise ValueError("occurrence_index must be non-negative")

    starts: list[int] = []
    offset = 0
    while True:
        found = generated_text.find(target_entity, offset)
        if found < 0:
            break
        starts.append(found)
        offset = found + max(len(target_entity), 1)

    if not starts:
        raise ValueError("target_entity was not found in generated_text")
    if occurrence_index >= len(starts):
        raise ValueError(
            f"occurrence_index={occurrence_index} is out of range; "
            f"found {len(starts)} occurrence(s)"
        )

    start = starts[occurrence_index]
    return start, start + len(target_entity)


def locate_target_context(
    prompt: str,
    generated_text: str,
    target_entity: str,
    *,
    char_start: int | None = None,
    char_end: int | None = None,
    occurrence_index: int = 0,
) -> PrefixContext:
    """Locate the target entity and the generated sentence containing it."""

    entity_start, entity_end = _resolve_entity_span(
        generated_text,
        target_entity,
        char_start,
        char_end,
        occurrence_index,
    )

    for sentence_start, sentence_end, sentence in _sentence_spans(generated_text):
        if sentence_start <= entity_start and entity_end <= sentence_end:
            return PrefixContext(
                prompt=prompt,
                generated_text=generated_text,
                target_entity=target_entity,
                entity_start=entity_start,
                entity_end=entity_end,
                source_sentence=sentence,
                context_left=generated_text[:sentence_start],
            )

    raise ValueError("could not locate a sentence containing target_entity")


def validate_rewrite(target_entity: str, result: RewriteResult) -> None:
    """Validate the structural invariants required by prefix construction."""

    rewritten = result.rewritten_sentence.rstrip()
    cloze = result.cloze_sentence.rstrip()

    if not rewritten.endswith(target_entity):
        raise ValueError("rewritten_sentence must end with the exact target entity")
    if not cloze.endswith(BLANK_TOKEN):
        raise ValueError("cloze_sentence must end with [BLANK]")
    if target_entity in cloze:
        raise ValueError("cloze_sentence must not contain the target entity")

    expected = rewritten[: len(rewritten) - len(target_entity)] + BLANK_TOKEN
    if cloze != expected:
        raise ValueError(
            "cloze_sentence must equal rewritten_sentence with only its final "
            "target entity replaced by [BLANK]"
        )


def _join_context(context_left: str, cloze_sentence: str) -> str:
    left = context_left or ""
    right = cloze_sentence.strip()
    if not left:
        return right
    if left.endswith((" ", "\n", "\t")):
        return left + right
    return left + " " + right


def prepare_prefix(context: PrefixContext, rewrite: RewriteResult) -> PreparedPrefix:
    """Construct both the inspectable cloze prefix and the sampling prefix."""

    validate_rewrite(context.target_entity, rewrite)
    full_prefix = _join_context(context.context_left, rewrite.cloze_sentence)
    sampling_prefix = full_prefix[: -len(BLANK_TOKEN)].rstrip() + " "
    return PreparedPrefix(
        source_sentence=context.source_sentence,
        rewritten_sentence=rewrite.rewritten_sentence.rstrip(),
        cloze_sentence=rewrite.cloze_sentence.rstrip(),
        full_prefix=full_prefix,
        sampling_prefix=sampling_prefix,
    )

