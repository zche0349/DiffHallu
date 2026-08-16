"""Model interfaces required by DiffHallu.

Users may implement these protocols with Transformers, vLLM, SGLang, an API,
or an internal inference system. DiffHallu does not configure or deploy models.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .config import GenerationConfig, RewriteConfig


@runtime_checkable
class RewriterBackend(Protocol):
    """Calls a user-owned API for prefix rewriting."""

    def complete(self, prompt: str, config: RewriteConfig) -> str:
        """Send DiffHallu's prompt to an API and return its raw text response."""

        ...


@runtime_checkable
class TargetModelBackend(Protocol):
    """Generates continuations and exposes teacher-forced token log-probabilities."""

    def generate(
        self,
        prompt: str,
        prefix: str,
        config: GenerationConfig,
    ) -> Sequence[str]:
        """Generate ``config.num_samples`` continuations from one fixed prefix."""

        ...

    def token_logprobs(
        self,
        prompt: str,
        prefix: str,
        completion: str,
    ) -> Sequence[float]:
        """Return one conditional log-probability per token in ``completion``."""

        ...


@runtime_checkable
class SemanticBackend(Protocol):
    """Predicts the directional NLI relation between two continuations."""

    def predict(self, premise: str, hypothesis: str) -> str:
        """Return ``entailment``, ``neutral``, or ``contradiction``."""

        ...
