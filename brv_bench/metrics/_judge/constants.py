"""Shared constants for the LLM judge and justifier subsystems."""

# ── Token limits ──────────────────────────────────────────────────────

# Default max tokens for raw_call across all backends.
RAW_CALL_DEFAULT_MAX_TOKENS = 512

# Max tokens allocated for a judge verdict response.
JUDGE_MAX_TOKENS = 8192

# Max tokens allocated for a justifier response.
JUSTIFIER_MAX_TOKENS = 32768

# ── Anthropic ─────────────────────────────────────────────────────────

# Default model for the Anthropic backend.
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"

# Minimum thinking-budget tokens for Claude models that use manual extended
# thinking (non-adaptive).  Enough for simple correct/incorrect reasoning.
ANTHROPIC_THINKING_BUDGET = 1024

# Effort level for adaptive-thinking models (Opus/Sonnet 4.6).
ANTHROPIC_ADAPTIVE_EFFORT = "low"

# Adaptive thinking: Opus 4.6 and Sonnet 4.6.
# budget_tokens / type:"enabled" is deprecated on both.
# Effort level is a separate output_config parameter — NOT inside thinking.
# Supported effort levels: low | medium | high | max (max: Opus 4.6 only).
ANTHROPIC_ADAPTIVE_MODELS = ("claude-opus-4-6", "claude-sonnet-4-6")

# All other Claude 3.7+ / Claude 4 models use manual extended thinking.
# (Haiku 4.5, Sonnet 4.5, Opus 4.5, Opus 4.1, Sonnet 4.0, Opus 4.0, Sonnet 3.7)
ANTHROPIC_ENABLED_PREFIXES = (
    "claude-3-7",
    "claude-haiku-4",
    "claude-sonnet-4",  # matches 4.0 / 4.5; 4.6 caught above
    "claude-opus-4",  # matches 4.0 / 4.1 / 4.5; 4.6 caught above
)

# ── OpenAI ────────────────────────────────────────────────────────────

# Default model for the OpenAI backend.
OPENAI_DEFAULT_MODEL = "gpt-4o-2024-08-06"

# ── Gemini ────────────────────────────────────────────────────────────

# Default model for the Gemini backend.
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

# thinking_budget value that disables Gemini thinking entirely.
GEMINI_THINKING_BUDGET_DISABLED = 0
