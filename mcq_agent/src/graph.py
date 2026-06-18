"""LangGraph workflow for RAG-backed MCQ generation, evaluation, and revision."""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph

from src.agents import build_chat_model, evaluate_mcq, generate_mcq, revise_mcq
from src.prompts import RETRIEVAL_QUERY_BEST_PRACTICES, RETRIEVAL_QUERY_EXAM
from src.schemas import EvaluationResult, MCQQuestion, WorkflowState
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


def _retrieval_query_exam(state: WorkflowState) -> str:
    return RETRIEVAL_QUERY_EXAM.format(
        topic=state["topic"],
        learning_objective=state["learning_objective"],
        difficulty=state["difficulty"],
    )


def _retrieval_query_best_practices(state: WorkflowState) -> str:
    return RETRIEVAL_QUERY_BEST_PRACTICES.format(topic=state["topic"])


def make_retrieve_exam_examples_node(deps: GraphDependencies):
    def retrieve_exam_examples(state: WorkflowState) -> dict:
        query = _retrieval_query_exam(state)
        context = deps.vector_manager.retrieve(EXAM_COLLECTION, query)
        if not context:
            logger.warning("No exam examples retrieved; generation will rely on defaults.")
        return {"exam_context": context}

    return retrieve_exam_examples


def make_generate_mcq_node(deps: GraphDependencies):
    def generate_mcq_node(state: WorkflowState) -> dict:
        index = state.get("current_question_index", 0) + 1
        question = generate_mcq(
            topic=state["topic"],
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            question_number=index,
            num_questions=state["num_questions"],
            exam_context=state.get("exam_context", []),
            model=deps.model,
        )
        return {
            "current_question_index": index,
            "current_question": question,
            "evaluation": None,
        }

    return generate_mcq_node


def make_retrieve_best_practices_node(deps: GraphDependencies):
    def retrieve_best_practices(state: WorkflowState) -> dict:
        query = _retrieval_query_best_practices(state)
        context = deps.vector_manager.retrieve(BEST_PRACTICES_COLLECTION, query)
        if not context:
            logger.warning(
                "No best-practice documents retrieved; evaluator will use defaults."
            )
        return {"best_practices_context": context}

    return retrieve_best_practices


def make_evaluate_mcq_node(deps: GraphDependencies):
    def evaluate_mcq_node(state: WorkflowState) -> dict:
        question = state["current_question"]
        if question is None:
            return {"error": "No current question to evaluate."}

        evaluation = evaluate_mcq(
            question=question,
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            best_practices_context=state.get("best_practices_context", []),
            model=deps.model,
        )
        updated = question.model_copy(
            update={
                "evaluator_feedback": evaluation.feedback,
                "approved": evaluation.approved,
            }
        )
        return {"current_question": updated, "evaluation": evaluation}

    return evaluate_mcq_node


def make_revise_mcq_node(deps: GraphDependencies):
    def revise_mcq_node(state: WorkflowState) -> dict:
        question = state["current_question"]
        evaluation = state.get("evaluation")
        if question is None or evaluation is None:
            return {"error": "Cannot revise without a question and evaluation."}

        next_round = question.revision_rounds + 1
        revised = revise_mcq(
            question=question,
            evaluation=evaluation,
            topic=state["topic"],
            learning_objective=state["learning_objective"],
            difficulty=state["difficulty"],
            revision_round=next_round,
            model=deps.model,
        )
        return {"current_question": revised}

    return revise_mcq_node


def finalize_question_node(state: WorkflowState) -> dict:
    """Append the current question to completed outputs and clear working state."""
    question = state.get("current_question")
    if question is None:
        return {"error": "No current question to finalize."}

    max_rounds = state.get("max_revision_rounds", 3)
    finalized = question.model_copy()
    if not finalized.approved and finalized.revision_rounds >= max_rounds:
        finalized.evaluator_feedback = (
            f"{finalized.evaluator_feedback} "
            f"[Max revision rounds ({max_rounds}) reached.]"
        ).strip()

    return {
        "completed_questions": [finalized],
        "current_question": None,
        "evaluation": None,
    }


def final_quality_check_node(state: WorkflowState) -> dict:
    """Quality gate after evaluation (routing handled by conditional edges)."""
    question = state.get("current_question")
    evaluation = state.get("evaluation")
    if question and evaluation:
        logger.info(
            "Quality check Q%d: approved=%s, revision_rounds=%d",
            state.get("current_question_index", 0),
            evaluation.approved,
            question.revision_rounds,
        )
    return {}


def route_after_quality_check(state: WorkflowState) -> Literal["revise_mcq", "finalize_question"]:
    """
    Decide whether to revise again or finalize the current question.

    Loops between evaluate_mcq and revise_mcq until approved or max rounds.
    """
    question = state.get("current_question")
    evaluation = state.get("evaluation")
    max_rounds = state.get("max_revision_rounds", 3)

    if question is None or evaluation is None:
        return "finalize_question"
    if evaluation.approved:
        return "finalize_question"
    if question.revision_rounds >= max_rounds:
        return "finalize_question"
    return "revise_mcq"


def route_after_finalize(state: WorkflowState) -> Literal["generate_mcq", "__end__"]:
    """Generate another question or finish when the batch is complete."""
    completed = len(state.get("completed_questions", []))
    target = state.get("num_questions", 1)
    if completed < target:
        return "generate_mcq"
    return "__end__"


def build_mcq_graph(
    deps: GraphDependencies | None = None,
    *,
    rebuild_indexes: bool = False,
):
    """Compile the LangGraph workflow."""
    dependencies = deps or GraphDependencies(rebuild_indexes=rebuild_indexes)

    # Local state type with list reducer for completed questions.
    class MCQGraphState(WorkflowState, total=False):
        completed_questions: Annotated[list[MCQQuestion], operator.add]

    graph = StateGraph(MCQGraphState)

    graph.add_node("retrieve_exam_examples", make_retrieve_exam_examples_node(dependencies))
    graph.add_node("generate_mcq", make_generate_mcq_node(dependencies))
    graph.add_node("retrieve_best_practices", make_retrieve_best_practices_node(dependencies))
    graph.add_node("evaluate_mcq", make_evaluate_mcq_node(dependencies))
    graph.add_node("final_quality_check", final_quality_check_node)
    graph.add_node("revise_mcq", make_revise_mcq_node(dependencies))
    graph.add_node("finalize_question", finalize_question_node)

    # Linear setup through first evaluation pass.
    graph.add_edge(START, "retrieve_exam_examples")
    graph.add_edge("retrieve_exam_examples", "generate_mcq")
    graph.add_edge("generate_mcq", "retrieve_best_practices")
    graph.add_edge("retrieve_best_practices", "evaluate_mcq")
    graph.add_edge("evaluate_mcq", "final_quality_check")

    # Quality gate: revise loop or finalize.
    graph.add_conditional_edges(
        "final_quality_check",
        route_after_quality_check,
        {
            "revise_mcq": "revise_mcq",
            "finalize_question": "finalize_question",
        },
    )
    graph.add_edge("revise_mcq", "evaluate_mcq")

    # Batch loop across questions.
    graph.add_conditional_edges(
        "finalize_question",
        route_after_finalize,
        {
            "generate_mcq": "generate_mcq",
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
    initial_state: WorkflowState = {
        "topic": topic,
        "learning_objective": learning_objective,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "max_revision_rounds": max_revision_rounds,
        "current_question_index": 0,
        "exam_context": [],
        "best_practices_context": [],
        "completed_questions": [],
        "current_question": None,
        "evaluation": None,
        "error": None,
    }
    return app.invoke(initial_state)
