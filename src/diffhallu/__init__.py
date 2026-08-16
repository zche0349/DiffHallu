"""DiffHallu: differential testing for long-form hallucination detection."""

from .config import DiffHalluConfig, GenerationConfig, RewriteConfig
from .interfaces import RewriterBackend, SemanticBackend, TargetModelBackend
from .pipeline import DiffHallu
from .schemas import Candidate, DetectionResult, RewriteResult
from .templates import (
    COMMON_STOP_TOKENS,
    CONVERSATION_TEMPLATES,
    ConversationTemplate,
    get_conversation_template,
    infer_model_family,
    render_conversation_prompt,
)

__all__ = [
    "Candidate",
    "COMMON_STOP_TOKENS",
    "CONVERSATION_TEMPLATES",
    "ConversationTemplate",
    "DetectionResult",
    "DiffHallu",
    "DiffHalluConfig",
    "GenerationConfig",
    "RewriteConfig",
    "RewriteResult",
    "RewriterBackend",
    "SemanticBackend",
    "TargetModelBackend",
    "get_conversation_template",
    "infer_model_family",
    "render_conversation_prompt",
]

__version__ = "0.1.0"
