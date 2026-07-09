"""LLM agent helpers for generation, evaluation, and revision."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.prompt_store import get_prompt
from src.schemas import (
    CompletedMCQRecord,
    DuplicateCheckResult,
    MCQQuestion,
    MCQQuestionBatch,
    QuestionBlueprint,
    QuestionBlueprintBatch,
    RejectedQuestionRecord,
    StructuredEvaluation,
)
from src.utils import dumps_json, get_model_config

logger = logging.getLogger("mcq_agent.agents")


def build_chat_model(
    callbacks: Sequence[BaseCallbackHandler] | None = None,
) -> ChatOpenAI:
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
    if callbacks:
        kwargs["callbacks"] = list(callbacks)
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


def _format_misconceptions_context(misconceptions_context: list[str] | None) -> str:
    if misconceptions_context:
        return "\n\n---\n\n".join(misconceptions_context)
    return "[No literature-backed misconceptions retrieved. Use topic knowledge.]"


def _format_best_practices_context(best_practices_context: list[str] | None) -> str:
    if best_practices_context:
        return "\n\n---\n\n".join(best_practices_context)
    return "[No best-practice documents retrieved. Apply standard MCQ design rules.]"


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
    misconceptions_context: list[str] | None = None,
    model: ChatOpenAI | None = None,
) -> QuestionBlueprint:
    """Create a structured plan for the next MCQ."""
    llm = model or build_chat_model()
    user_prompt = get_prompt("BLUEPRINT_USER_PROMPT").format(
        topic=topic,
        learning_objective=learning_objective,
        difficulty=difficulty,
        question_number=question_number,
        num_questions=num_questions,
        completed_summaries=_summarize_completed(completed_questions),
        rejected_summaries=_summarize_rejected(rejected_questions),
        misconceptions_context=_format_misconceptions_context(misconceptions_context),
    )
    return _invoke_structured(
        llm, get_prompt("BLUEPRINT_SYSTEM_PROMPT"), user_prompt, QuestionBlueprint
    )


def create_batch_blueprints(
    *,
    topic: str,
    learning_objective: str,
    difficulty: str,
    start_question_number: int,
    num_questions: int,
    batch_count: int,
    completed_questions: list[CompletedMCQRecord],
    rejected_questions: list[RejectedQuestionRecord],
    misconceptions_context: list[str] | None = None,
    model: ChatOpenAI | None = None,
) -> QuestionBlueprintBatch:
    """Create structured plans for a batch of MCQs."""
    llm = model or build_chat_model()
    user_prompt = get_prompt("BATCH_BLUEPRINT_USER_PROMPT").format(
        topic=topic,
        learning_objective=learning_objective,
        difficulty=difficulty,
        batch_count=batch_count,
        start_question_number=start_question_number,
        num_questions=num_questions,
        completed_summaries=_summarize_completed(completed_questions),
        rejected_summaries=_summarize_rejected(rejected_questions),
        misconceptions_context=_format_misconceptions_context(misconceptions_context),
    )
    batch = _invoke_structured(
        llm,
        get_prompt("BATCH_BLUEPRINT_SYSTEM_PROMPT"),
        user_prompt,
        QuestionBlueprintBatch,
    )
    if len(batch.blueprints) != batch_count:
        raise ValueError(
            f"Expected {batch_count} blueprints, got {len(batch.blueprints)}"
        )
    return batch


def generate_mcq(
    *,
    blueprint: QuestionBlueprint,
    misconceptions_context: list[str],
    best_practices_context: list[str],
    model: ChatOpenAI | None = None,
) -> MCQQuestion:
    """Generate a single MCQ from blueprint, misconceptions, and best-practice constraints."""
    llm = model or build_chat_model()
    user_prompt = get_prompt("GENERATION_USER_PROMPT").format(
        blueprint_json=dumps_json(blueprint),
        misconceptions_context=_format_misconceptions_context(misconceptions_context),
        best_practices_context=_format_best_practices_context(best_practices_context),
    )
    question = _invoke_structured(
        llm, get_prompt("GENERATION_SYSTEM_PROMPT"), user_prompt, MCQQuestion
    )
    question.source_context_used = misconceptions_context[:]
    question.learning_objective = blueprint.learning_objective
    return question


def generate_batch_mcqs(
    *,
    blueprints: list[QuestionBlueprint],
    misconceptions_context: list[str],
    best_practices_context: list[str],
    model: ChatOpenAI | None = None,
) -> MCQQuestionBatch:
    """Generate a batch of MCQs from blueprints and retrieved context."""
    llm = model or build_chat_model()
    user_prompt = get_prompt("BATCH_GENERATION_USER_PROMPT").format(
        blueprints_json=dumps_json(blueprints),
        batch_count=len(blueprints),
        misconceptions_context=_format_misconceptions_context(misconceptions_context),
        best_practices_context=_format_best_practices_context(best_practices_context),
    )
    batch = _invoke_structured(
        llm,
        get_prompt("BATCH_GENERATION_SYSTEM_PROMPT"),
        user_prompt,
        MCQQuestionBatch,
    )
    if len(batch.questions) != len(blueprints):
        raise ValueError(
            f"Expected {len(blueprints)} questions, got {len(batch.questions)}"
        )

    finalized: list[MCQQuestion] = []
    for blueprint, question in zip(blueprints, batch.questions, strict=True):
        question.source_context_used = misconceptions_context[:]
        question.learning_objective = blueprint.learning_objective
        finalized.append(question)
    return MCQQuestionBatch(questions=finalized)


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
    user_prompt = get_prompt("CONTENT_CHECK_USER_PROMPT").format(
        learning_objective=learning_objective,
        difficulty=difficulty,
        blueprint_json=dumps_json(blueprint),
        mcq_json=dumps_json(question),
    )
    return _invoke_structured(
        llm, get_prompt("CONTENT_CHECK_SYSTEM_PROMPT"), user_prompt, StructuredEvaluation
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
    user_prompt = get_prompt("QUALITY_CHECK_USER_PROMPT").format(
        learning_objective=learning_objective,
        difficulty=difficulty,
        blueprint_json=dumps_json(blueprint),
        mcq_json=dumps_json(question),
        best_practices_context=_format_best_practices_context(best_practices_context),
    )
    return _invoke_structured(
        llm, get_prompt("QUALITY_CHECK_SYSTEM_PROMPT"), user_prompt, StructuredEvaluation
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
    user_prompt = get_prompt("DUPLICATE_CHECK_USER_PROMPT").format(
        mcq_json=dumps_json(question),
        target_misconception=blueprint.target_misconception,
        completed_questions_summary=_summarize_completed_for_duplicate(completed_questions),
    )
    return _invoke_structured(
        llm, get_prompt("DUPLICATE_CHECK_SYSTEM_PROMPT"), user_prompt, DuplicateCheckResult
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
    user_prompt = get_prompt("REVISION_USER_PROMPT").format(
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
    revised = _invoke_structured(
        llm, get_prompt("REVISION_SYSTEM_PROMPT"), user_prompt, MCQQuestion
    )
    revised.source_context_used = question.source_context_used
    revised.learning_objective = learning_objective
    revised.revision_rounds = revision_round
    revised.evaluator_feedback = (
        f"Content: {content_evaluation.revision_instructions}; "
        f"Quality: {quality_evaluation.revision_instructions}; "
        f"Duplicate: {duplicate_evaluation.revision_instructions}"
    ).strip("; ")
    return revised
