"""Reference conversation templates used by the original DiffHallu pipeline.

The templates are provided for reproducibility only. They do not load or
configure a model. A user-provided target-model backend decides whether to use
one of these templates or the deployed tokenizer's native chat template.
"""

from __future__ import annotations

from dataclasses import dataclass


HELPFUL_ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant."

COMMON_STOP_TOKENS = (
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
)


@dataclass(frozen=True)
class ConversationTemplate:
    """A named model-family prompt template and its system instruction."""

    family: str
    prompt_format: str
    system_prompt: str | None

    def render(self, instruction: str, prefix: str) -> str:
        """Render the conversation prompt followed by one fixed text prefix."""

        if not instruction:
            raise ValueError("instruction must not be empty")
        prefix_clean = prefix.replace("[BLANK]", "").rstrip() + " "
        return self.prompt_format.format(instruction=instruction) + prefix_clean


CONVERSATION_TEMPLATES: dict[str, ConversationTemplate] = {
    "llama-3": ConversationTemplate(
        family="llama-3",
        system_prompt=HELPFUL_ASSISTANT_SYSTEM_PROMPT,
        prompt_format=(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{HELPFUL_ASSISTANT_SYSTEM_PROMPT}"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|><|start_header_id|>assistant"
            "<|end_header_id|>\n\n"
        ),
    ),
    "mistral": ConversationTemplate(
        family="mistral",
        system_prompt=None,
        prompt_format="[INST] {instruction} [/INST]",
    ),
    "qwen": ConversationTemplate(
        family="qwen",
        system_prompt=HELPFUL_ASSISTANT_SYSTEM_PROMPT,
        prompt_format=(
            "<|im_start|>system\n"
            f"{HELPFUL_ASSISTANT_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    ),
    "gemma": ConversationTemplate(
        family="gemma",
        system_prompt=None,
        prompt_format=(
            "<bos><start_of_turn>user\n{instruction}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
    ),
}


def get_conversation_template(model_family: str) -> ConversationTemplate:
    """Return one reference template by its explicit model-family key."""

    key = model_family.strip().lower()
    try:
        return CONVERSATION_TEMPLATES[key]
    except KeyError as error:
        supported = ", ".join(sorted(CONVERSATION_TEMPLATES))
        raise ValueError(
            f"unsupported model_family={model_family!r}; choose one of: {supported}"
        ) from error


def infer_model_family(model_name_or_path: str) -> str:
    """Infer a provided reference-template key from a model name or path."""

    name = model_name_or_path.lower()
    if "llama-3" in name or "llama3" in name or "meta-llama" in name:
        return "llama-3"
    if "mistral" in name:
        return "mistral"
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    raise ValueError(f"cannot infer a conversation template for {model_name_or_path!r}")


def render_conversation_prompt(
    model_family: str,
    instruction: str,
    prefix: str,
) -> str:
    """Render one of the supplied reference templates."""

    return get_conversation_template(model_family).render(instruction, prefix)
