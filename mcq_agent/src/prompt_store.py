"""Session-scoped prompt overrides via contextvars (safe for concurrent Streamlit users)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import src.prompts as prompts_module

# Editable prompt keys exposed in the Streamlit UI (system prompts + retrieval
# queries only — USER_PROMPT templates are not editable).
PROMPT_GROUPS: dict[str, list[str]] = {
    "Batch blueprint": [
        "BATCH_BLUEPRINT_SYSTEM_PROMPT",
    ],
    "Batch generation": [
        "BATCH_GENERATION_SYSTEM_PROMPT",
    ],
    "Content accuracy check": [
        "CONTENT_CHECK_SYSTEM_PROMPT",
    ],
    "MCQ quality check": [
        "QUALITY_CHECK_SYSTEM_PROMPT",
    ],
    "Duplicate check": [
        "DUPLICATE_CHECK_SYSTEM_PROMPT",
    ],
    "Revision": [
        "REVISION_SYSTEM_PROMPT",
    ],
    "Retrieval queries": [
        "RETRIEVAL_QUERY_MISCONCEPTIONS",
        "RETRIEVAL_QUERY_BEST_PRACTICES",
    ],
    "Unused / legacy (single-item path)": [
        "BLUEPRINT_SYSTEM_PROMPT",
        "GENERATION_SYSTEM_PROMPT",
    ],
}

ALL_PROMPT_KEYS: tuple[str, ...] = tuple(
    key for keys in PROMPT_GROUPS.values() for key in keys
)

_prompt_overrides: ContextVar[dict[str, str] | None] = ContextVar(
    "prompt_overrides",
    default=None,
)


def get_defaults() -> dict[str, str]:
    """Return the built-in prompt strings from ``src.prompts``."""
    defaults: dict[str, str] = {}
    for key in ALL_PROMPT_KEYS:
        value = getattr(prompts_module, key, None)
        if value is None:
            raise KeyError(f"Unknown prompt key: {key}")
        defaults[key] = str(value)
    return defaults


def get_prompt(key: str) -> str:
    """Resolve a prompt, preferring the active context override when set."""
    overrides = _prompt_overrides.get()
    if overrides and key in overrides:
        return overrides[key]
    value = getattr(prompts_module, key, None)
    if value is None:
        raise KeyError(f"Unknown prompt key: {key}")
    return str(value)


@contextmanager
def prompt_overrides(overrides: dict[str, str] | None) -> Iterator[None]:
    """Apply prompt overrides for the current context (e.g. one Streamlit run)."""
    if not overrides:
        yield
        return
    # Only keep known editable keys.
    filtered = {k: v for k, v in overrides.items() if k in ALL_PROMPT_KEYS}
    token = _prompt_overrides.set(filtered)
    try:
        yield
    finally:
        _prompt_overrides.reset(token)
