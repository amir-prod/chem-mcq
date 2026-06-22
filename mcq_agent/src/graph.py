"""LangGraph workflow for RAG-backed MCQ generation, evaluation, and revision."""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph

from src.agents import (
    build_chat_model,
    check_content_accuracy,
    check_duplicate,
    check_mcq_quality,
    create_question_blueprint,
    generate_mcq,
    revise_mcq,
)
from src.prompts import RETRIEVAL_QUERY_BEST_PRACTICES, RETRIEVAL_QUERY_EXAM
from src.schemas import (
    CompletedMCQRecord,
    DuplicateCheckResult,
    MCQQuestion,
    QuestionBlueprint,
    RejectedQuestionRecord,
    StructuredEvaluation,
    WorkflowState,
    compute_max_total_attempts,
)
from src.utils import BEST_PRACTICES_COLLECTION, EXAM_COLLECTION
from src.vectorstore import VectorStoreManager, ensure_indexes

logger = logging.getLogger("mcq_agent.graph")


class GraphDependencies:
    """Injectable dependencies so the workflow stays modular and testable."""

    def __init__(
        self,
        vector_manager: VectorStoreManager | None = None,
        rebuild_indexes: bool = False,
    ) -> None:
        self.vector_manager = ensure_indexes(
            vector_manager,
            rebuild=rebuild_indexes,
        )
        self.model = build_chat_model()


def _blueprint_hint(state: WorkflowState) -> str:
    blueprint = state.get("question_blueprint")
    if blueprint is None:
        return ""
    return (
        f"{blueprint.cognitive_level} {blueprint.correct_answer_concept} "
        f"{blueprint.target_misconception}"
    )


def _retrieval_query_exam(state: WorkflowState) -> str:
    return RETRIEVAL_QUERY_EXAM.format(
        topic=state["topic"],
        learning_objective=state["learning_objective"],
        difficulty=state["difficulty"],
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


def route_after_finalize(
    state: WorkflowState,
) -> Literal["create_question_blueprint", "__end__"]:
    """Start a new blueprint when more approved questions are needed."""
    completed = len(state.get("completed_questions", []))
    target = state.get("num_questions", 1)
    if completed < target:
        if _attempts_exceeded(state):
            logger.error(
                "Generation attempt cap reached with %d/%d questions completed.",
                completed,
                target,
            )
            return "__end__"
        return "create_question_blueprint"
    return "__end__"


def route_after_reject(
    state: WorkflowState,
) -> Literal["create_question_blueprint", "__end__"]:
    """After rejection, try a fresh blueprint unless attempt cap is hit."""
    completed = len(state.get("completed_questions", []))
    target = state.get("num_questions", 1)
    if completed >= target:
        return "__end__"
    if _attempts_exceeded(state):
        return "__end__"
    return "create_question_blueprint"


def make_create_question_blueprint_node(deps: GraphDependencies):
    def create_question_blueprint_node(state: WorkflowState) -> dict:
        if _attempts_exceeded(state):
            return {
                "error": (
                    f"Exceeded max generation attempts ({state.get('max_total_attempts')}). "
                    f"Completed {len(state.get('completed_questions', []))} of "
                    f"{state.get('num_questions', 1)} requested."
                )
            }

        index = len(state.get("completed_questions", [])) + 1
        blueprint = create_question_blueprint(
            topic=state["topic"],
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            question_number=index,
            num_questions=state["num_questions"],
            completed_questions=state.get("completed_questions", []),
            rejected_questions=state.get("rejected_questions", []),
            model=deps.model,
        )
        return {
            "current_question_index": index,
            "question_blueprint": blueprint,
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "candidate_question": None,
            "content_evaluation": None,
            "quality_evaluation": None,
            "duplicate_evaluation": None,
            "exam_examples_context": [],
            "best_practices_context": [],
        }

    return create_question_blueprint_node


def make_retrieve_exam_examples_node(deps: GraphDependencies):
    def retrieve_exam_examples(state: WorkflowState) -> dict:
        query = _retrieval_query_exam(state)
        context = deps.vector_manager.retrieve(EXAM_COLLECTION, query)
        if not context:
            logger.warning("No exam examples retrieved; generation will rely on defaults.")
        return {"exam_examples_context": context}

    return retrieve_exam_examples


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


def make_generate_mcq_node(deps: GraphDependencies):
    def generate_mcq_node(state: WorkflowState) -> dict:
        blueprint = state.get("question_blueprint")
        if blueprint is None:
            return {"error": "No question blueprint available for generation."}

        question = generate_mcq(
            blueprint=blueprint,
            question_number=state.get("current_question_index", 1),
            num_questions=state["num_questions"],
            exam_context=state.get("exam_examples_context", []),
            best_practices_context=state.get("best_practices_context", []),
            model=deps.model,
        )
        return {
            "candidate_question": question,
            "content_evaluation": None,
            "quality_evaluation": None,
            "duplicate_evaluation": None,
        }

    return generate_mcq_node


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
        return {
            "candidate_question": revised,
            "content_evaluation": None,
            "quality_evaluation": None,
            "duplicate_evaluation": None,
        }

    return revise_mcq_node


def finalize_question_node(state: WorkflowState) -> dict:
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
    return {
        "completed_questions": [record],
        "candidate_question": None,
        "question_blueprint": None,
        "content_evaluation": None,
        "quality_evaluation": None,
        "duplicate_evaluation": None,
        "exam_examples_context": [],
        "best_practices_context": [],
    }


def reject_or_regenerate_node(state: WorkflowState) -> dict:
    """Reject a failed question and prepare for a new blueprint (no finalize)."""
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

    rejected = RejectedQuestionRecord(
        question_blueprint=blueprint,
        last_candidate=question,
        rejection_reason=rejection_reason,
        revision_rounds=_revision_rounds(state),
        content_evaluation=content,
        quality_evaluation=quality,
        duplicate_evaluation=duplicate,
    )
    return {
        "rejected_questions": [rejected],
        "candidate_question": None,
        "question_blueprint": None,
        "content_evaluation": None,
        "quality_evaluation": None,
        "duplicate_evaluation": None,
        "exam_examples_context": [],
        "best_practices_context": [],
    }


def build_mcq_graph(
    deps: GraphDependencies | None = None,
    *,
    rebuild_indexes: bool = False,
):
    """Compile the LangGraph workflow."""
    dependencies = deps or GraphDependencies(rebuild_indexes=rebuild_indexes)

    class MCQGraphState(WorkflowState, total=False):
        completed_questions: Annotated[list[CompletedMCQRecord], operator.add]
        rejected_questions: Annotated[list[RejectedQuestionRecord], operator.add]

    graph = StateGraph(MCQGraphState)

    graph.add_node(
        "create_question_blueprint",
        make_create_question_blueprint_node(dependencies),
    )
    graph.add_node(
        "retrieve_exam_examples",
        make_retrieve_exam_examples_node(dependencies),
    )
    graph.add_node(
        "retrieve_best_practices",
        make_retrieve_best_practices_node(dependencies),
    )
    graph.add_node("generate_mcq", make_generate_mcq_node(dependencies))
    graph.add_node(
        "content_accuracy_check",
        make_content_accuracy_check_node(dependencies),
    )
    graph.add_node("mcq_quality_check", make_mcq_quality_check_node(dependencies))
    graph.add_node("duplicate_check", make_duplicate_check_node(dependencies))
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("revise_mcq", make_revise_mcq_node(dependencies))
    graph.add_node("finalize_question", finalize_question_node)
    graph.add_node("reject_or_regenerate", reject_or_regenerate_node)

    # Per-question pipeline: blueprint -> retrieve both -> generate -> review chain.
    graph.add_edge(START, "create_question_blueprint")
    graph.add_edge("create_question_blueprint", "retrieve_exam_examples")
    graph.add_edge("retrieve_exam_examples", "retrieve_best_practices")
    graph.add_edge("retrieve_best_practices", "generate_mcq")
    graph.add_edge("generate_mcq", "content_accuracy_check")
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

    graph.add_conditional_edges(
        "finalize_question",
        route_after_finalize,
        {
            "create_question_blueprint": "create_question_blueprint",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "reject_or_regenerate",
        route_after_reject,
        {
            "create_question_blueprint": "create_question_blueprint",
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
    max_revision_rounds: int = 3,
    rebuild_indexes: bool = False,
) -> dict:
    """Execute the compiled graph and return the final state."""
    app = build_mcq_graph(rebuild_indexes=rebuild_indexes)
    max_total = compute_max_total_attempts(num_questions, max_revision_rounds)
    initial_state: WorkflowState = {
        "topic": topic,
        "learning_objective": learning_objective,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "max_revision_rounds": max_revision_rounds,
        "max_total_attempts": max_total,
        "generation_attempts": 0,
        "current_question_index": 0,
        "question_blueprint": None,
        "exam_examples_context": [],
        "best_practices_context": [],
        "completed_questions": [],
        "rejected_questions": [],
        "candidate_question": None,
        "content_evaluation": None,
        "quality_evaluation": None,
        "duplicate_evaluation": None,
        "error": None,
    }
    return app.invoke(initial_state)
