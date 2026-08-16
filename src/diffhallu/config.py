"""Method parameters used by DiffHallu.

This module intentionally contains no model name, model path, endpoint, device,
credential, or deployment configuration. Model integration belongs to the
backend implementations supplied by users.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewriteConfig:
    """Default parameters passed to the user-provided prefix-rewrite API."""

    max_tokens: int = 768
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


@dataclass(frozen=True)
class GenerationConfig:
    """Default candidate-generation parameters used by the original pipeline."""

    num_samples: int = 10
    max_new_tokens: int = 50
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


@dataclass(frozen=True)
class DiffHalluConfig:
    """Algorithm configuration independent of any concrete model backend."""

    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
