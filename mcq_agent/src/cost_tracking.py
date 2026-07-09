"""LangChain callback for token usage and estimated USD cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# USD per 1M tokens. Estimates only; update as provider pricing changes.
# Format: model_name -> (input_per_1m, output_per_1m) for chat;
# embedding models use (input_per_1m, 0.0).
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
}


def _normalize_model_name(model: str | None) -> str:
    if not model:
        return ""
    name = model.strip().lower()
    # Strip common date/version suffixes like gpt-4o-mini-2024-07-18
    for known in MODEL_PRICING_PER_1M:
        if name == known or name.startswith(known + "-"):
            return known
    return name


@dataclass
class UsageTotals:
    """Accumulated token counts and estimated cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float = 0.0
    unknown_model_tokens: int = 0
    models_seen: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": (
                self.prompt_tokens + self.completion_tokens + self.embedding_tokens
            ),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "unknown_model_tokens": self.unknown_model_tokens,
            "models_seen": sorted(self.models_seen),
            "pricing_known": self.unknown_model_tokens == 0,
        }


class UsageCostCallback(BaseCallbackHandler):
    """Accumulate LLM and embedding token usage and estimate USD cost."""

    def __init__(self) -> None:
        super().__init__()
        self.totals = UsageTotals()

    def _add_chat_cost(
        self,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        normalized = _normalize_model_name(model)
        if normalized:
            self.totals.models_seen.add(normalized)
        self.totals.prompt_tokens += prompt_tokens
        self.totals.completion_tokens += completion_tokens
        rates = MODEL_PRICING_PER_1M.get(normalized)
        if rates is None:
            self.totals.unknown_model_tokens += prompt_tokens + completion_tokens
            return
        input_rate, output_rate = rates
        self.totals.estimated_cost_usd += (
            prompt_tokens * input_rate + completion_tokens * output_rate
        ) / 1_000_000

    def _add_embedding_cost(self, model: str | None, tokens: int) -> None:
        normalized = _normalize_model_name(model)
        if normalized:
            self.totals.models_seen.add(normalized)
        self.totals.embedding_tokens += tokens
        rates = MODEL_PRICING_PER_1M.get(normalized)
        if rates is None:
            self.totals.unknown_model_tokens += tokens
            return
        input_rate, _ = rates
        self.totals.estimated_cost_usd += (tokens * input_rate) / 1_000_000

    @staticmethod
    def _extract_token_usage(response: LLMResult) -> tuple[int, int, str | None]:
        prompt_tokens = 0
        completion_tokens = 0
        model: str | None = None

        # Prefer llm_output usage (classic OpenAI path).
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            prompt_tokens = int(
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or 0
            )
            completion_tokens = int(
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or 0
            )
        model = llm_output.get("model_name") or llm_output.get("model")

        # Fall back to per-generation usage_metadata / response_metadata.
        if prompt_tokens == 0 and completion_tokens == 0:
            for generations in response.generations:
                for gen in generations:
                    message = getattr(gen, "message", None)
                    if message is None:
                        continue
                    meta = getattr(message, "usage_metadata", None) or {}
                    if meta:
                        prompt_tokens += int(
                            meta.get("input_tokens") or meta.get("prompt_tokens") or 0
                        )
                        completion_tokens += int(
                            meta.get("output_tokens")
                            or meta.get("completion_tokens")
                            or 0
                        )
                    resp_meta = getattr(message, "response_metadata", None) or {}
                    if not model:
                        model = resp_meta.get("model_name") or resp_meta.get("model")
                    token_usage = resp_meta.get("token_usage") or {}
                    if token_usage and prompt_tokens == 0 and completion_tokens == 0:
                        prompt_tokens += int(token_usage.get("prompt_tokens") or 0)
                        completion_tokens += int(
                            token_usage.get("completion_tokens") or 0
                        )

        return prompt_tokens, completion_tokens, model

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        prompt_tokens, completion_tokens, model = self._extract_token_usage(response)
        if prompt_tokens or completion_tokens:
            self._add_chat_cost(model, prompt_tokens, completion_tokens)

    def on_embedding_end(
        self,
        response: list[list[float]],
        **kwargs: Any,
    ) -> None:
        # LangChain OpenAI embeddings often put usage on kwargs / run metadata.
        # When unavailable, estimate roughly from character length if provided.
        run_meta = kwargs.get("metadata") or {}
        usage = run_meta.get("token_usage") or run_meta.get("usage") or {}
        tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        model = run_meta.get("model") or run_meta.get("model_name")
        if tokens:
            self._add_embedding_cost(model, tokens)

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of current totals for UI display."""
        return self.totals.as_dict()
