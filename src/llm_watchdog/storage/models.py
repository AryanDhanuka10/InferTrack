"""
models.py — Core data structures for llm-watchdog
==================================================
CallLog is the single record written to SQLite for every LLM call.
Using stdlib dataclasses — no Pydantic, no SQLAlchemy.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CallLog:
    """
    One row in the logs table. Represents a single LLM API call.

    Fields:
        id:               Auto-assigned by SQLite (None before insert)
        timestamp:        When the call was made (UTC)
        provider:         'openai', 'anthropic', 'ollama', 'unknown'
        model:            e.g. 'qwen2.5:0.5b', 'gpt-4o'
        input_tokens:     From response.usage.prompt_tokens
        output_tokens:    From response.usage.completion_tokens
        total_tokens:     input + output
        cost_usd:         Calculated from prices.json (0.0 for Ollama)
        latency_ms:       Wall clock time in milliseconds
        tag:              Optional label (e.g. 'summarize', 'search')
        user_id:          Optional user identifier for budget tracking
        session_id:       Optional session grouping
        success:          False if the call raised an exception
        error_msg:        Exception message if success=False, else None
    """

    # Identity
    id: Optional[int] = field(default=None)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Provider info
    provider: str = "unknown"
    model: str = "unknown"

    # Token usage (from response.usage — no tiktoken)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost (0.0 for Ollama)
    cost_usd: float = 0.0

    # Performance
    latency_ms: float = 0.0

    # Tagging
    tag: str = "default"
    user_id: str = "anonymous"
    session_id: Optional[str] = None

    # Status
    success: bool = True
    error_msg: Optional[str] = None

    def __post_init__(self) -> None:
        """Auto-compute total_tokens if not provided."""
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens
