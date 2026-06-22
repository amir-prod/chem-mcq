"""LLM agent helpers for generation, evaluation, and revision."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.prompts import (
    BLUEPRINT_SYSTEM_PROMPT,
    BLUEPRINT_USER_PROMPT,
    CONTENT_CHECK_SYSTEM_PROMPT,
    CONTENT_CHECK_USER_PROMPT,
    DUPLICATE_CHECK_SYSTEM_PROMPT,
    DUPLICATE_CHECK_USER_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    QUALITY_CHECK_SYSTEM_PROMPT,
    QUALITY_CHECK_USER_PROMPT,
    REVISION_SYSTEM_PROMPT,
    REVISION_USER_PROMPT,
)
from src.schemas import (
    CompletedMCQRecord,
    DuplicateCheckResult,
    MCQQuestion,
    QuestionBlueprint,
    RejectedQuestionRecord,
    StructuredEvaluation,
)
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


def _summarize_completed(records: list[CompletedMCQRecord]) -> str:
    if not records:
        return "[None yet]"
    lines = []
    for index, record in enumerate(records, start=1):
        mcq = record.mcq
        lines.append(
            f"{index}. {mcq.question[:120]}… "
            f"(misconception: {record.question_blueprint.target_misconception})"
        )
    return "\n".join(lines)


def _summarize_rejected(records: list[RejectedQuestionRecord]) -> str:
    if not records:
        return "[None]"
    lines = []
    for index, record in enumerate(records, start=1):
        blueprint = record.question_blueprint
        if blueprint:
            lines.append(
                f"{index}. {blueprint.correct_answer_concept} "
                f"(misconception: {blueprint.target_misconception})"
            )
        else:
            lines.append(f"{index}. {record.rejection_reason}")
    return "\n".join(lines)


def _summarize_completed_for_duplicate(records: list[CompletedMCQRecord]) -> str:
    if not records:
        return "[No completed questions yet]"
    lines = []
    for index, record in enumerate(records, start=1):
        mcq = record.mcq
        lines.append(
            f"ID {index}: stem={mcq.question[:100]}…; "
            f"correct={mcq.correct_answer}; "
            f"concept={record.question_blueprint.correct_answer_concept}; "
            f"misconception={record.question_blueprint.target_misconception}"
        )
    return "\n".join(lines)


def create_question_blueprint(
    *,
    topic: str,
    learning_objective: str,
    difficulty: str,
    question_number: int,
    num_questions: int,
    completed_questions: list[CompletedMCQRecord],
    rejected_questions: list[RejectedQuestionRecord],
    model: ChatOpenAI | None = None,
) -> QuestionBlueprint:
    """Create a structured plan for the next MCQ."""
    llm = model or build_chat_model()
    user_prompt = BLUEPRINT_USER_PROMPT.format(
        topic=topic,
        learning_objective=learning_objective,
        difficulty=difficulty,
        question_number=question_number,
        num_questions=num_questions,
        completed_summaries=_summarize_completed(completed_questions),
        rejected_summaries=_summarize_rejected(rejected_questions),
    )
    return _invoke_structured(llm, BLUEPRINT_SYSTEM_PROMPT, user_prompt, QuestionBlueprint)


def generate_mcq(
    *,
    blueprint: QuestionBlueprint,
    question_number: int,
    num_questions: int,
    exam_context: list[str],
    best_practices_context: list[str],
    model: ChatOpenAI | None = None,
) -> MCQQuestion:
    """Generate a single MCQ from blueprint, exam style, and best-practice constraints."""
    llm = model or build_chat_model()
    exam_text = (
        "\n\n---\n\n".join(exam_context)
        if exam_context
        else "[No exam examples retrieved. Use standard chemistry exam style.]"
    )
    practices_text = (
        "\n\n---\n\n".join(best_practices_context)
        if best_practices_context
        else "[No best-practice documents retrieved. Apply standard MCQ design rules.]"
    )
    user_prompt = GENERATION_USER_PROMPT.format(
        blueprint_json=dumps_json(blueprint),
        exam_context=exam_text,
        best_practices_context=practices_text,
    )
    question = _invoke_structured(llm, GENERATION_SYSTEM_PROMPT, user_prompt, MCQQuestion)
    question.source_context_used = exam_context[:]
    question.learning_objective = blueprint.learning_objective
    return question


def check_content_accuracy(
    *,
    question: MCQQuestion,
    blueprint: QuestionBlueprint,
    learning_objective: str,
    difficulty: str,
    model: ChatOpenAI | None = None,
) -> StructuredEvaluation:
    """Verify scientific correctness and answer key validity."""
    llm = model or build_chat_model()
    user_prompt = CONTENT_CHECK_USER_PROMPT.format(
        learning_objective=learning_objective,
        difficulty=difficulty,
        blueprint_json=dumps_json(blueprint),
        mcq_json=dumps_json(question),
    )
    return _invoke_structured(
        llm, CONTENT_CHECK_SYSTEM_PROMPT, user_prompt, StructuredEvaluation
    )


def check_mcq_quality(
    *,
    question: MCQQuestion,
    blueprint: QuestionBlueprint,
    learning_objective: str,
    difficulty: str,
    best_practices_context: list[str],
    model: ChatOpenAI | None = None,
) -> StructuredEvaluation:
    """Evaluate MCQ-writing quality against best-practice rubric."""
    llm = model or build_chat_model()
    context_text = (
        "\n\n---\n\n".join(best_practices_context)
        if best_practices_context
        else "[No best-practice documents retrieved. Apply standard MCQ design rules.]"
    )
    user_prompt = QUALITY_CHECK_USER_PROMPT.format(
        learning_objective=learning_objective,
        difficulty=difficulty,
        blueprint_json=dumps_json(blueprint),
        mcq_json=dumps_json(question),
        best_practices_context=context_text,
    )
    return _invoke_structured(
        llm, QUALITY_CHECK_SYSTEM_PROMPT, user_prompt, StructuredEvaluation
    )


def check_duplicate(
    *,
    question: MCQQuestion,
    blueprint: QuestionBlueprint,
    completed_questions: list[CompletedMCQRecord],
    model: ChatOpenAI | None = None,
) -> DuplicateCheckResult:
    """Detect duplicate or near-duplicate questions against completed batch."""
    if not completed_questions:
        return DuplicateCheckResult(is_duplicate=False)

    llm = model or build_chat_model()
    user_prompt = DUPLICATE_CHECK_USER_PROMPT.format(
        mcq_json=dumps_json(question),
        target_misconception=blueprint.target_misconception,
        completed_questions_summary=_summarize_completed_for_duplicate(completed_questions),
    )
    return _invoke_structured(
        llm, DUPLICATE_CHECK_SYSTEM_PROMPT, user_prompt, DuplicateCheckResult
    )


def revise_mcq(
    *,
    question: MCQQuestion,
    blueprint: QuestionBlueprint,
    content_evaluation: StructuredEvaluation,
    quality_evaluation: StructuredEvaluation,
    duplicate_evaluation: DuplicateCheckResult,
    topic: str,
    learning_objective: str,
    difficulty: str,
    revision_round: int,
    model: ChatOpenAI | None = None,
) -> MCQQuestion:
    """Revise an MCQ using structured feedback from all reviewers."""
    llm = model or build_chat_model()

    def _format_eval(label: str, evaluation: StructuredEvaluation) -> str:
        issues = "\n".join(f"- {issue}" for issue in evaluation.issues) or "- None"
        blocking = (
            "\n".join(f"- {err}" for err in evaluation.blocking_errors) or "- None"
        )
        return (
            f"{label} approved={evaluation.approved}, score={evaluation.score}\n"
            f"Issues:\n{issues}\n"
            f"Blocking errors:\n{blocking}\n"
            f"Revision instructions: {evaluation.revision_instructions or 'None'}"
        )

    duplicate_feedback = (
        f"is_duplicate={duplicate_evaluation.is_duplicate}; "
        f"reason={duplicate_evaluation.similarity_reason or 'None'}; "
        f"instructions={duplicate_evaluation.revision_instructions or 'None'}"
    )
    user_prompt = REVISION_USER_PROMPT.format(
        topic=topic,
        learning_objective=learning_objective,
        difficulty=difficulty,
        blueprint_json=dumps_json(blueprint),
        mcq_json=dumps_json(question),
        content_feedback=_format_eval("Content", content_evaluation),
        quality_feedback=_format_eval("Quality", quality_evaluation),
        duplicate_feedback=duplicate_feedback,
        revision_round=revision_round,
    )
    revised = _invoke_structured(llm, REVISION_SYSTEM_PROMPT, user_prompt, MCQQuestion)
    revised.source_context_used = question.source_context_used
    revised.learning_objective = learning_objective
    revised.revision_rounds = revision_round
    revised.evaluator_feedback = (
        f"Content: {content_evaluation.revision_instructions}; "
        f"Quality: {quality_evaluation.revision_instructions}; "
        f"Duplicate: {duplicate_evaluation.revision_instructions}"
    ).strip("; ")
    return revised
