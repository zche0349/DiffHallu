"""Prompt construction and response parsing for controlled prefix rewriting."""

from __future__ import annotations

import json
from typing import Any

from .config import RewriteConfig
from .interfaces import RewriterBackend
from .prefix import BLANK_TOKEN, validate_rewrite
from .schemas import PrefixContext, RewriteResult


def build_rewrite_prompt(entity: str, source_sentence: str) -> str:
    """Build the complete controlled-rewrite prompt used by DiffHallu."""

    return f"""You are an expert in controlled sentence rewriting and cloze creation.

Task
Given ONE English sentence and ONE target entity string, output JSON with exactly two keys:

- "rewritten_sentence": ONE complete grammatical English sentence. All information that describes the entity must appear BEFORE the entity. The very last characters must be exactly the entity string (no trailing punctuation, no trailing whitespace).
- "cloze_sentence": identical to rewritten_sentence, except the FINAL entity string is mechanically replaced with the ASCII placeholder [BLANK]. [BLANK] must be the final characters (no trailing punctuation, no trailing whitespace).

Quality & Solvability
- Keep enough unique context FROM THE SOURCE before the entity so the blank is guessable; do not add new facts.
- Avoid tautology/circular definitions.

Formatting & Structure (Crucial)

1) List marker:
If the source starts with an ASCII list marker like "4.", "-", "*", keep that marker at the start of BOTH outputs.

2) Frozen header:
- If the source starts with a Markdown header that does NOT contain the target entity (e.g., "- **Header**: ...entity..."),
  keep the header prefix exactly (e.g., "- **Header**:") and rewrite only the part after ":" so the entity is the final characters.
- If the source starts with the entity as a Markdown header title, e.g., "- **ENTITY**: ..." or "* **ENTITY**: ..." or "1. **ENTITY**: ...",
  then you MUST NOT keep that header title text (the "**ENTITY**:" part), because it would repeat the entity.
  Keep ONLY the list marker ("- " / "* " / "1. "), and rewrite the description as ONE sentence ending with the exact entity string.

Scope pruning
- You may delete text only if it is clearly NOT describing the target entity (topic switch / unrelated continuation).
- Never delete truth-conditional content about the entity (negation, conditions, comparisons, quantities, time ranges, restrictive modifiers).

Output format
{{
"rewritten_sentence":"...",
"cloze_sentence":"..."
}}

Input
Target entity (use exactly): "{entity}"
Original sentence: "{source_sentence}"

Few-shot (invented; ASCII only)

Input:
Target entity (use exactly): "Zephyr Mobility"
Original sentence: "4. **Zephyr Mobility** - Zephyr Mobility develops lidar perception stacks."
Output:
{{
"rewritten_sentence":"4. A developer of lidar perception stacks is Zephyr Mobility",
"cloze_sentence":"4. A developer of lidar perception stacks is [BLANK]"
}}

Input:
Target entity (use exactly): "Director Nila Verma"
Original sentence: "- **Director Nila Verma**: Oversaw the field inspection, reviewed the safety logs, and approved the final report."
Output:
{{
"rewritten_sentence":"- The person who oversaw the field inspection, reviewed the safety logs, and approved the final report was Director Nila Verma",
"cloze_sentence":"- The person who oversaw the field inspection, reviewed the safety logs, and approved the final report was [BLANK]"
}}

Input:
Target entity (use exactly): "patient records"
Original sentence: "- **Clinical Trials**: Led the multi-site trial, analyzing the patient records and coordinating with overseas labs."
Output:
{{
"rewritten_sentence":"- **Clinical Trials**: While leading the multi-site trial and coordinating with overseas labs, the analysis focused on patient records",
"cloze_sentence":"- **Clinical Trials**: While leading the multi-site trial and coordinating with overseas labs, the analysis focused on [BLANK]"
}}

Input:
Target entity (use exactly): "Ontario Court of Appeal"
Original sentence: "The case then proceeded to the Ontario Court of Appeal, which also upheld the conviction."
Output:
{{
"rewritten_sentence":"The court that the case then proceeded to and that also upheld the conviction was Ontario Court of Appeal",
"cloze_sentence":"The court that the case then proceeded to and that also upheld the conviction was [BLANK]"
}}

Input:
Target entity (use exactly): "Mira Sato"
Original sentence: "The internal report was compiled by Mira Sato, who also designed the follow-up survey."
Output:
{{
"rewritten_sentence":"The person who compiled the internal report and also designed the follow-up survey was Mira Sato",
"cloze_sentence":"The person who compiled the internal report and also designed the follow-up survey was [BLANK]"
}}

Input:
Target entity (use exactly): "Aether BioSensors"
Original sentence: "Aether BioSensors manufactures portable glucose monitors, but the newsletter also includes unrelated event listings."
Output:
{{
"rewritten_sentence":"A manufacturer of portable glucose monitors is Aether BioSensors",
"cloze_sentence":"A manufacturer of portable glucose monitors is [BLANK]"
}}
"""


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from a possibly wrapped response."""

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def parse_rewrite_response(raw_response: str) -> RewriteResult:
    """Parse the API's raw response into DiffHallu's rewrite schema."""

    if not raw_response or not raw_response.strip():
        raise ValueError("prefix-rewrite API returned an empty response")

    candidate = _extract_first_json_object(raw_response.strip())
    if candidate is None:
        raise ValueError("prefix-rewrite API response does not contain a JSON object")

    try:
        payload: Any = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("prefix-rewrite API returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("prefix-rewrite API JSON must be an object")

    rewritten = str(payload.get("rewritten_sentence") or "").strip()
    cloze = str(payload.get("cloze_sentence") or "").strip()
    blank_position = cloze.rfind(BLANK_TOKEN)
    if blank_position >= 0:
        cloze = cloze[: blank_position + len(BLANK_TOKEN)].rstrip()

    if not rewritten or not cloze:
        raise ValueError(
            "prefix-rewrite API JSON must contain rewritten_sentence and cloze_sentence"
        )
    return RewriteResult(rewritten_sentence=rewritten, cloze_sentence=cloze)


def rewrite_target_sentence(
    backend: RewriterBackend,
    context: PrefixContext,
    config: RewriteConfig,
) -> RewriteResult:
    """Build the method prompt, call the user's API, parse, and validate it."""

    prompt = build_rewrite_prompt(context.target_entity, context.source_sentence)
    raw_response = backend.complete(prompt, config)
    result = parse_rewrite_response(raw_response)
    validate_rewrite(context.target_entity, result)
    return result

