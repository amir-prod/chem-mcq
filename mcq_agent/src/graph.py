"""LangGraph workflow for RAG-backed MCQ generation, evaluation, and revision."""

from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph

from src.agents import (
    build_chat_model,
    check_content_accuracy,
    check_duplicate,
    check_mcq_quality,
    create_batch_blueprints,
    generate_batch_mcqs,
    revise_mcq,
)
from src.prompts import (
    RETRIEVAL_QUERY_BEST_PRACTICES,
    RETRIEVAL_QUERY_MISCONCEPTIONS,
)
from src.schemas import (
    CompletedMCQRecord,
    DuplicateCheckResult,
    RejectedQuestionRecord,
    StemMediaType,
    StructuredEvaluation,
    WorkflowState,
    allocate_stem_media_types,
    compute_max_total_attempts,
)
from src.utils import (
    BEST_PRACTICES_COLLECTION,
    DEFAULT_BATCH_SIZE,
    MISCONCEPTIONS_COLLECTION,
    persist_workflow_outputs,
)
from src.vectorstore import VectorStoreManager, ensure_indexes

logger = logging.getLogger("mcq_agent.graph")


class GraphDependencies:
    """Injectable dependencies so the workflow stays modular and testable."""

    def __init__(
        self,
        vector_manager: VectorStoreManager | None = None,
        rebuild_indexes: bool = False,
        *,
        output_json: Path | None = None,
        output_md: Path | None = None,
        output_rejected_json: Path | None = None,
    ) -> None:
        self.vector_manager = ensure_indexes(
            vector_manager,
            rebuild=rebuild_indexes,
        )
        self.model = build_chat_model()
        self.output_json = output_json
        self.output_md = output_md
        self.output_rejected_json = output_rejected_json

    def checkpoint(self, state: WorkflowState, *, completed=None, rejected=None) -> None:
        """Persist approved/rejected questions so far (no-op if paths unset)."""
        if not self.output_json or not self.output_md or not self.output_rejected_json:
            return
        completed_questions = (
            completed
            if completed is not None
            else list(state.get("completed_questions", []))
        )
        rejected_questions = (
            rejected
            if rejected is not None
            else list(state.get("rejected_questions", []))
        )
        persist_workflow_outputs(
            topic=state["topic"],
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            completed_questions=completed_questions,
            rejected_questions=rejected_questions,
            json_path=self.output_json,
            md_path=self.output_md,
            rejected_json_path=self.output_rejected_json,
        )


def _blueprint_hint(state: WorkflowState) -> str:
    blueprint = state.get("question_blueprint")
    if blueprint is None:
        batch_blueprints = state.get("batch_blueprints", [])
        batch_index = state.get("batch_index", 0)
        if batch_index < len(batch_blueprints):
            blueprint = batch_blueprints[batch_index]
    if blueprint is None:
        return ""
    return (
        f"{blueprint.cognitive_level} {blueprint.correct_answer_concept} "
        f"{blueprint.target_misconception}"
    )


def _retrieval_query_misconceptions(state: WorkflowState) -> str:
    return RETRIEVAL_QUERY_MISCONCEPTIONS.format(
        topic=state["topic"],
        learning_objective=state["learning_objective"],
        blueprint_hint=_blueprint_hint(state),
    )


def _retrieval_query_best_practices(state: WorkflowState) -> str:
    return RETRIEVAL_QUERY_BEST_PRACTICES.format(
        topic=state["topic"],
        blueprint_hint=_blueprint_hint(state),
    )


def _revision_rounds(state: WorkflowState) -> int:
    candidate = state.get("candidate_question")
    return candidate.revision_rounds if candidate else 0


def _attempts_exceeded(state: WorkflowState) -> bool:
    attempts = state.get("generation_attempts", 0)
    cap = state.get("max_total_attempts", 1)
    return attempts >= cap


def _remaining_questions(state: WorkflowState) -> int:
    completed = len(state.get("completed_questions", []))
    return max(state.get("num_questions", 1) - completed, 0)


def _current_batch_count(state: WorkflowState) -> int:
    batch_size = state.get("batch_size", DEFAULT_BATCH_SIZE)
    return min(batch_size, _remaining_questions(state))


def route_after_quality_gate(
    state: WorkflowState,
) -> Literal["finalize_question", "revise_mcq", "reject_or_regenerate"]:
    """
    Combine content, quality, and duplicate checks.

    - All pass -> finalize
    - Any fail with revision budget left -> revise
    - Any fail after max rounds -> reject (do NOT finalize)
    """
    if _attempts_exceeded(state):
        logger.error("Generation attempt cap reached; rejecting current question.")
        return "reject_or_regenerate"

    content = state.get("content_evaluation")
    quality = state.get("quality_evaluation")
    duplicate = state.get("duplicate_evaluation")
    max_rounds = state.get("max_revision_rounds", 3)
    rounds = _revision_rounds(state)

    if content is None or quality is None or duplicate is None:
        if rounds < max_rounds:
            return "revise_mcq"
        return "reject_or_regenerate"

    all_approved = (
        content.approved and quality.approved and not duplicate.is_duplicate
    )
    if all_approved:
        return "finalize_question"
    if rounds < max_rounds:
        return "revise_mcq"
    return "reject_or_regenerate"


def route_after_advance(
    state: WorkflowState,
) -> Literal["select_batch_question", "retrieve_misconceptions", "__end__"]:
    """Advance to the next question in the batch or start a new batch."""
    completed = len(state.get("completed_questions", []))
    target = state.get("num_questions", 1)
    if completed >= target:
        return "__end__"
    if _attempts_exceeded(state):
        logger.error(
            "Generation attempt cap reached with %d/%d questions completed.",
            completed,
            target,
        )
        return "__end__"

    batch_index = state.get("batch_index", 0)
    batch_len = len(state.get("batch_blueprints", []))
    if batch_index < batch_len:
        return "select_batch_question"
    return "retrieve_misconceptions"


def make_retrieve_misconceptions_node(deps: GraphDependencies):
    def retrieve_misconceptions(state: WorkflowState) -> dict:
        if _remaining_questions(state) == 0:
            return {}
        query = _retrieval_query_misconceptions(state)
        context = deps.vector_manager.retrieve(MISCONCEPTIONS_COLLECTION, query)
        if not context:
            logger.warning(
                "No misconceptions retrieved; blueprint will rely on model knowledge."
            )
        return {
            "misconceptions_context": context,
            "batch_index": 0,
            "batch_blueprints": [],
            "batch_candidates": [],
            "best_practices_context": [],
        }

    return retrieve_misconceptions


def make_create_batch_blueprints_node(deps: GraphDependencies):
    def create_batch_blueprints_node(state: WorkflowState) -> dict:
        remaining = _remaining_questions(state)
        if remaining == 0:
            return {}

        if _attempts_exceeded(state):
            return {
                "error": (
                    f"Exceeded max generation attempts ({state.get('max_total_attempts')}). "
                    f"Completed {len(state.get('completed_questions', []))} of "
                    f"{state.get('num_questions', 1)} requested."
                )
            }

        batch_count = _current_batch_count(state)
        if batch_count == 0:
            return {}

        start_question_number = len(state.get("completed_questions", [])) + 1
        media_plan = state.get("stem_media_plan") or allocate_stem_media_types(
            state["num_questions"]
        )
        start_index = start_question_number - 1
        assigned_media_types = media_plan[start_index : start_index + batch_count]
        if len(assigned_media_types) < batch_count:
            # Safety pad if plan is shorter than expected (should not happen).
            pad = [StemMediaType.TEXT] * (batch_count - len(assigned_media_types))
            assigned_media_types = list(assigned_media_types) + pad

        batch = create_batch_blueprints(
            topic=state["topic"],
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            start_question_number=start_question_number,
            num_questions=state["num_questions"],
            batch_count=batch_count,
            completed_questions=state.get("completed_questions", []),
            rejected_questions=state.get("rejected_questions", []),
            misconceptions_context=state.get("misconceptions_context", []),
            assigned_media_types=assigned_media_types,
            model=deps.model,
        )
        return {
            "batch_start_index": start_question_number,
            "batch_blueprints": batch.blueprints,
            "batch_candidates": [],
            "batch_index": 0,
            "generation_attempts": state.get("generation_attempts", 0) + batch_count,
            "question_blueprint": None,
            "candidate_question": None,
            "content_evaluation": None,
            "quality_evaluation": None,
            "duplicate_evaluation": None,
        }

    return create_batch_blueprints_node


def make_retrieve_best_practices_node(deps: GraphDependencies):
    def retrieve_best_practices(state: WorkflowState) -> dict:
        query = _retrieval_query_best_practices(state)
        context = deps.vector_manager.retrieve(BEST_PRACTICES_COLLECTION, query)
        if not context:
            logger.warning(
                "No best-practice documents retrieved; reviewers will use defaults."
            )
        return {"best_practices_context": context}

    return retrieve_best_practices


def make_generate_batch_mcqs_node(deps: GraphDependencies):
    def generate_batch_mcqs_node(state: WorkflowState) -> dict:
        blueprints = state.get("batch_blueprints", [])
        if not blueprints:
            return {"error": "No batch blueprints available for generation."}

        batch = generate_batch_mcqs(
            blueprints=blueprints,
            misconceptions_context=state.get("misconceptions_context", []),
            best_practices_context=state.get("best_practices_context", []),
            model=deps.model,
        )
        return {
            "batch_candidates": batch.questions,
            "candidate_question": None,
            "content_evaluation": None,
            "quality_evaluation": None,
            "duplicate_evaluation": None,
        }

    return generate_batch_mcqs_node


def select_batch_question_node(state: WorkflowState) -> dict:
    """Select the current question from the active batch for evaluation."""
    batch_index = state.get("batch_index", 0)
    blueprints = state.get("batch_blueprints", [])
    candidates = state.get("batch_candidates", [])
    if batch_index >= len(blueprints) or batch_index >= len(candidates):
        return {"error": "Batch index out of range for question selection."}

    start_index = state.get("batch_start_index", 1)
    return {
        "question_blueprint": blueprints[batch_index],
        "candidate_question": candidates[batch_index],
        "current_question_index": start_index + batch_index,
        "content_evaluation": None,
        "quality_evaluation": None,
        "duplicate_evaluation": None,
    }


def make_content_accuracy_check_node(deps: GraphDependencies):
    def content_accuracy_check_node(state: WorkflowState) -> dict:
        question = state.get("candidate_question")
        blueprint = state.get("question_blueprint")
        if question is None or blueprint is None:
            return {"error": "No candidate question or blueprint for content check."}

        evaluation = check_content_accuracy(
            question=question,
            blueprint=blueprint,
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            model=deps.model,
        )
        return {"content_evaluation": evaluation}

    return content_accuracy_check_node


def make_mcq_quality_check_node(deps: GraphDependencies):
    def mcq_quality_check_node(state: WorkflowState) -> dict:
        question = state.get("candidate_question")
        blueprint = state.get("question_blueprint")
        if question is None or blueprint is None:
            return {"error": "No candidate question or blueprint for quality check."}

        evaluation = check_mcq_quality(
            question=question,
            blueprint=blueprint,
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            best_practices_context=state.get("best_practices_context", []),
            model=deps.model,
        )
        return {"quality_evaluation": evaluation}

    return mcq_quality_check_node


def make_duplicate_check_node(deps: GraphDependencies):
    def duplicate_check_node(state: WorkflowState) -> dict:
        question = state.get("candidate_question")
        blueprint = state.get("question_blueprint")
        if question is None or blueprint is None:
            return {"error": "No candidate question or blueprint for duplicate check."}

        result = check_duplicate(
            question=question,
            blueprint=blueprint,
            completed_questions=state.get("completed_questions", []),
            model=deps.model,
        )
        return {"duplicate_evaluation": result}

    return duplicate_check_node


def quality_gate_node(state: WorkflowState) -> dict:
    """Log combined gate status (routing handled by conditional edges)."""
    content = state.get("content_evaluation")
    quality = state.get("quality_evaluation")
    duplicate = state.get("duplicate_evaluation")
    if content and quality and duplicate:
        logger.info(
            "Quality gate Q%d: content=%s quality=%s duplicate=%s rounds=%d",
            state.get("current_question_index", 0),
            content.approved,
            quality.approved,
            duplicate.is_duplicate,
            _revision_rounds(state),
        )
    return {}


def make_revise_mcq_node(deps: GraphDependencies):
    def revise_mcq_node(state: WorkflowState) -> dict:
        question = state.get("candidate_question")
        blueprint = state.get("question_blueprint")
        content = state.get("content_evaluation")
        quality = state.get("quality_evaluation")
        duplicate = state.get("duplicate_evaluation")
        if (
            question is None
            or blueprint is None
            or content is None
            or quality is None
            or duplicate is None
        ):
            return {"error": "Cannot revise without question, blueprint, and evaluations."}

        next_round = question.revision_rounds + 1
        revised = revise_mcq(
            question=question,
            blueprint=blueprint,
            content_evaluation=content,
            quality_evaluation=quality,
            duplicate_evaluation=duplicate,
            topic=state["topic"],
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            revision_round=next_round,
            model=deps.model,
        )

        batch_index = state.get("batch_index", 0)
        batch_candidates = list(state.get("batch_candidates", []))
        if batch_index < len(batch_candidates):
            batch_candidates[batch_index] = revised

        return {
            "candidate_question": revised,
            "batch_candidates": batch_candidates,
            "content_evaluation": None,
            "quality_evaluation": None,
            "duplicate_evaluation": None,
        }

    return revise_mcq_node


def finalize_question_node(state: WorkflowState) -> dict:
    """Append an approved question to completed outputs and clear working state."""
    return _finalize_question(state, checkpoint=None)


def make_finalize_question_node(deps: GraphDependencies):
    def finalize_with_checkpoint(state: WorkflowState) -> dict:
        return _finalize_question(state, checkpoint=deps.checkpoint)

    return finalize_with_checkpoint


def _finalize_question(state: WorkflowState, checkpoint) -> dict:
    """Append an approved question to completed outputs and clear working state."""
    question = state.get("candidate_question")
    blueprint = state.get("question_blueprint")
    content = state.get("content_evaluation")
    quality = state.get("quality_evaluation")
    duplicate = state.get("duplicate_evaluation")

    if (
        question is None
        or blueprint is None
        or content is None
        or quality is None
        or duplicate is None
    ):
        return {"error": "Cannot finalize without question, blueprint, and evaluations."}

    if not (content.approved and quality.approved and not duplicate.is_duplicate):
        return {"error": "Finalize called but quality gate did not pass."}

    finalized_mcq = question.model_copy(update={"approved": True})
    record = CompletedMCQRecord(
        mcq=finalized_mcq,
        question_blueprint=blueprint,
        revision_rounds=finalized_mcq.revision_rounds,
        content_evaluation=content,
        quality_evaluation=quality,
        duplicate_evaluation=duplicate,
    )
    completed = list(state.get("completed_questions", [])) + [record]
    rejected = list(state.get("rejected_questions", []))
    if checkpoint is not None:
        try:
            checkpoint(state, completed=completed, rejected=rejected)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to checkpoint outputs after finalize.")

    return {
        "completed_questions": [record],
        "candidate_question": None,
        "question_blueprint": None,
        "content_evaluation": None,
        "quality_evaluation": None,
        "duplicate_evaluation": None,
    }


def reject_or_regenerate_node(state: WorkflowState) -> dict:
    """Reject a failed question and prepare for the next batch slot."""
    return _reject_or_regenerate(state, checkpoint=None)


def make_reject_or_regenerate_node(deps: GraphDependencies):
    def reject_with_checkpoint(state: WorkflowState) -> dict:
        return _reject_or_regenerate(state, checkpoint=deps.checkpoint)

    return reject_with_checkpoint


def _reject_or_regenerate(state: WorkflowState, checkpoint) -> dict:
    """Reject a failed question and prepare for the next batch slot."""
    question = state.get("candidate_question")
    blueprint = state.get("question_blueprint")
    content = state.get("content_evaluation")
    quality = state.get("quality_evaluation")
    duplicate = state.get("duplicate_evaluation")
    max_rounds = state.get("max_revision_rounds", 3)

    reasons: list[str] = []
    if content and not content.approved:
        reasons.append("content accuracy failed")
    if quality and not quality.approved:
        reasons.append("MCQ quality failed")
    if duplicate and duplicate.is_duplicate:
        reasons.append(f"duplicate: {duplicate.similarity_reason}")

    rejection_reason = (
        "; ".join(reasons) if reasons else f"max revision rounds ({max_rounds}) reached"
    )
    logger.warning(
        "Rejecting question slot %d after %d revision rounds: %s",
        state.get("current_question_index", 0),
        _revision_rounds(state),
        rejection_reason,
    )

    rejected_record = RejectedQuestionRecord(
        question_blueprint=blueprint,
        last_candidate=question,
        rejection_reason=rejection_reason,
        revision_rounds=_revision_rounds(state),
        content_evaluation=content,
        quality_evaluation=quality,
        duplicate_evaluation=duplicate,
    )
    completed = list(state.get("completed_questions", []))
    rejected = list(state.get("rejected_questions", [])) + [rejected_record]
    if checkpoint is not None:
        try:
            checkpoint(state, completed=completed, rejected=rejected)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to checkpoint outputs after reject.")

    return {
        "rejected_questions": [rejected_record],
        "candidate_question": None,
        "question_blueprint": None,
        "content_evaluation": None,
        "quality_evaluation": None,
        "duplicate_evaluation": None,
    }


def advance_batch_node(state: WorkflowState) -> dict:
    """Move to the next question slot within the current batch."""
    return {"batch_index": state.get("batch_index", 0) + 1}


def build_mcq_graph(
    deps: GraphDependencies | None = None,
    *,
    rebuild_indexes: bool = False,
    output_json: Path | None = None,
    output_md: Path | None = None,
    output_rejected_json: Path | None = None,
):
    """Compile the LangGraph workflow."""
    dependencies = deps or GraphDependencies(
        rebuild_indexes=rebuild_indexes,
        output_json=output_json,
        output_md=output_md,
        output_rejected_json=output_rejected_json,
    )

    class MCQGraphState(WorkflowState, total=False):
        completed_questions: Annotated[list[CompletedMCQRecord], operator.add]
        rejected_questions: Annotated[list[RejectedQuestionRecord], operator.add]

    graph = StateGraph(MCQGraphState)

    graph.add_node(
        "retrieve_misconceptions",
        make_retrieve_misconceptions_node(dependencies),
    )
    graph.add_node(
        "create_batch_blueprints",
        make_create_batch_blueprints_node(dependencies),
    )
    graph.add_node(
        "retrieve_best_practices",
        make_retrieve_best_practices_node(dependencies),
    )
    graph.add_node(
        "generate_batch_mcqs",
        make_generate_batch_mcqs_node(dependencies),
    )
    graph.add_node("select_batch_question", select_batch_question_node)
    graph.add_node(
        "content_accuracy_check",
        make_content_accuracy_check_node(dependencies),
    )
    graph.add_node("mcq_quality_check", make_mcq_quality_check_node(dependencies))
    graph.add_node("duplicate_check", make_duplicate_check_node(dependencies))
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("revise_mcq", make_revise_mcq_node(dependencies))
    graph.add_node("finalize_question", make_finalize_question_node(dependencies))
    graph.add_node("reject_or_regenerate", make_reject_or_regenerate_node(dependencies))
    graph.add_node("advance_batch", advance_batch_node)

    graph.add_edge(START, "retrieve_misconceptions")
    graph.add_edge("retrieve_misconceptions", "create_batch_blueprints")
    graph.add_edge("create_batch_blueprints", "retrieve_best_practices")
    graph.add_edge("retrieve_best_practices", "generate_batch_mcqs")
    graph.add_edge("generate_batch_mcqs", "select_batch_question")
    graph.add_edge("select_batch_question", "content_accuracy_check")
    graph.add_edge("content_accuracy_check", "mcq_quality_check")
    graph.add_edge("mcq_quality_check", "duplicate_check")
    graph.add_edge("duplicate_check", "quality_gate")

    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "finalize_question": "finalize_question",
            "revise_mcq": "revise_mcq",
            "reject_or_regenerate": "reject_or_regenerate",
        },
    )
    graph.add_edge("revise_mcq", "content_accuracy_check")
    graph.add_edge("finalize_question", "advance_batch")
    graph.add_edge("reject_or_regenerate", "advance_batch")

    graph.add_conditional_edges(
        "advance_batch",
        route_after_advance,
        {
            "select_batch_question": "select_batch_question",
            "retrieve_misconceptions": "retrieve_misconceptions",
            "__end__": END,
        },
    )

    return graph.compile()


def run_workflow(
    *,
    topic: str,
    learning_objective: str,
    difficulty: str,
    num_questions: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_revision_rounds: int = 3,
    rebuild_indexes: bool = False,
    output_json: Path | None = None,
    output_md: Path | None = None,
    output_rejected_json: Path | None = None,
) -> dict:
    """
    Execute the compiled graph and return the final (or partial) state.

    Streams state updates so Ctrl+C / unexpected stops can still return whatever
    was approved so far. Approved and rejected items are also checkpointed to
    disk after each finalize/reject when output paths are provided.
    """
    app = build_mcq_graph(
        rebuild_indexes=rebuild_indexes,
        output_json=output_json,
        output_md=output_md,
        output_rejected_json=output_rejected_json,
    )
    max_total = compute_max_total_attempts(num_questions, max_revision_rounds)
    initial_state: WorkflowState = {
        "topic": topic,
        "learning_objective": learning_objective,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "batch_size": batch_size,
        "max_revision_rounds": max_revision_rounds,
        "max_total_attempts": max_total,
        "generation_attempts": 0,
        "batch_start_index": 1,
        "batch_index": 0,
        "batch_blueprints": [],
        "batch_candidates": [],
        "current_question_index": 0,
        "question_blueprint": None,
        "misconceptions_context": [],
        "best_practices_context": [],
        "completed_questions": [],
        "rejected_questions": [],
        "candidate_question": None,
        "content_evaluation": None,
        "quality_evaluation": None,
        "duplicate_evaluation": None,
        "stem_media_plan": allocate_stem_media_types(num_questions),
        "error": None,
    }

    last_state: dict = dict(initial_state)
    try:
        for state in app.stream(initial_state, stream_mode="values"):
            last_state = state
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user after %d approved / %d rejected question(s).",
            len(last_state.get("completed_questions", [])),
            len(last_state.get("rejected_questions", [])),
        )
        last_state = {
            **last_state,
            "error": last_state.get("error") or "Interrupted by user",
        }
    return last_state
