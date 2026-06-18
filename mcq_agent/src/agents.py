"""LLM agent helpers for generation, evaluation, and revision."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.prompts import (
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_USER_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    REVISION_SYSTEM_PROMPT,
    REVISION_USER_PROMPT,
)
from src.schemas import EvaluationResult, MCQQuestion
from src.utils import dumps_json, get_model_config

logger = logging.getLogger("mcq_agent.agents")


def build_chat_model() -> ChatOpenAI:
    """Create an OpenAI-compatible chat model from environment settings."""
    config = get_model_config()
    if not config["api_key"]:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to mcq_agent/.env before running."
        )
    kwargs: dict[str, Any] = {
        "model": config["model"],
        "api_key": config["api_key"],
        "temperature": config["temperature"],
    }
    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]
    return ChatOpenAI(**kwargs)


def _invoke_structured(model: ChatOpenAI, system: str, user: str, schema: type):
    """Invoke the model with structured output parsing."""
    structured = model.with_structured_output(schema)
    return structured.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )


def generate_mcq(
    *,
    topic: str,
    learning_objective: str,
    difficulty: str,
    question_number: int,
    num_questions: int,
    exam_context: list[str],
    model: ChatOpenAI | None = None,
) -> MCQQuestion:
    """Generate a single MCQ using retrieved exam examples as style context."""
    llm = model or build_chat_model()
    context_text = (
        "\n\n---\n\n".join(exam_context)
        if exam_context
        else "[No exam examples retrieved. Use standard chemistry exam style.]"
    )
    user_prompt = GENERATION_USER_PROMPT.format(
        topic=topic,
        learning_objective=learning_objective,
        difficulty=difficulty,
        question_number=question_number,
        num_questions=num_questions,
        exam_context=context_text,
    )
    question = _invoke_structured(llm, GENERATION_SYSTEM_PROMPT, user_prompt, MCQQuestion)
    question.source_context_used = exam_context[:]
    question.learning_objective = learning_objective
    return question


def evaluate_mcq(
    *,
    question: MCQQuestion,
    learning_objective: str,
    difficulty: str,
    best_practices_context: list[str],
    model: ChatOpenAI | None = None,
) -> EvaluationResult:
    """Evaluate an MCQ against retrieved best-practice guidelines."""
    llm = model or build_chat_model()
    context_text = (
        "\n\n---\n\n".join(best_practices_context)
        if best_practices_context
        else "[No best-practice documents retrieved. Apply standard MCQ design rules.]"
    )
    user_prompt = EVALUATION_USER_PROMPT.format(
        learning_objective=learning_objective,
        difficulty=difficulty,
        mcq_json=dumps_json(question),
        best_practices_context=context_text,
    )
    return _invoke_structured(llm, EVALUATION_SYSTEM_PROMPT, user_prompt, EvaluationResult)


def revise_mcq(
    *,
    question: MCQQuestion,
    evaluation: EvaluationResult,
    topic: str,
    learning_objective: str,
    difficulty: str,
    revision_round: int,
    model: ChatOpenAI | None = None,
) -> MCQQuestion:
    """Revise an MCQ using evaluator feedback while preserving the learning objective."""
    llm = model or build_chat_model()
    issues_text = "\n".join(f"- {issue}" for issue in evaluation.issues) or "- None listed"
    user_prompt = REVISION_USER_PROMPT.format(
        topic=topic,
        learning_objective=learning_objective,
        difficulty=difficulty,
        mcq_json=dumps_json(question),
        evaluator_feedback=evaluation.feedback,
        issues=issues_text,
        revision_round=revision_round,
    )
    revised = _invoke_structured(llm, REVISION_SYSTEM_PROMPT, user_prompt, MCQQuestion)
    revised.source_context_used = question.source_context_used
    revised.learning_objective = learning_objective
    revised.revision_rounds = revision_round
    revised.evaluator_feedback = evaluation.feedback
    return revised
