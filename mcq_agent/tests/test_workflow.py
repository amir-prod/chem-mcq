"""Tests for workflow routing, schemas, and quality-gate behavior."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.graph import (
    finalize_question_node,
    reject_or_regenerate_node,
    route_after_finalize,
    route_after_quality_gate,
    route_after_reject,
    run_workflow,
)
from src.schemas import (
    CognitiveLevel,
    CompletedMCQRecord,
    Difficulty,
    DuplicateCheckResult,
    MCQOptions,
    MCQQuestion,
    QuestionBlueprint,
    RejectedQuestionRecord,
    StructuredEvaluation,
    compute_max_total_attempts,
)


def _sample_blueprint() -> QuestionBlueprint:
    return QuestionBlueprint(
        topic="stoichiometry",
        learning_objective="Balance chemical equations",
        difficulty="medium",
        cognitive_level="application",
        target_misconception="Confusing coefficients with subscripts",
        correct_answer_concept="Coefficients scale entire formulas",
        expected_reasoning="Student identifies mole ratios from balanced equation",
        question_style_notes="Single-concept calculation item",
    )


def _sample_mcq(**overrides) -> MCQQuestion:
    defaults = dict(
        question="What is the coefficient of O2 when balancing CH4 + O2 -> CO2 + H2O?",
        options=MCQOptions(A="1", B="2", C="3", D="4"),
        correct_answer="B",
        explanation="Two oxygen molecules are needed on the left.",
        learning_objective="Balance chemical equations",
        difficulty=Difficulty.MEDIUM,
        cognitive_level=CognitiveLevel.APPLICATION,
        revision_rounds=0,
        approved=False,
    )
    defaults.update(overrides)
    return MCQQuestion(**defaults)


def _approved_eval() -> StructuredEvaluation:
    return StructuredEvaluation(approved=True, score=0.95, issues=[])


def _failed_eval(instructions: str = "Fix the stem") -> StructuredEvaluation:
    return StructuredEvaluation(
        approved=False,
        score=0.4,
        issues=["Ambiguous stem"],
        revision_instructions=instructions,
        blocking_errors=["Multiple correct answers possible"],
    )


def _base_state(**overrides):
    state = {
        "topic": "stoichiometry",
        "learning_objective": "Balance chemical equations",
        "difficulty": "medium",
        "num_questions": 2,
        "max_revision_rounds": 2,
        "max_total_attempts": compute_max_total_attempts(2, 2),
        "generation_attempts": 1,
        "current_question_index": 1,
        "question_blueprint": _sample_blueprint(),
        "candidate_question": _sample_mcq(),
        "content_evaluation": _approved_eval(),
        "quality_evaluation": _approved_eval(),
        "duplicate_evaluation": DuplicateCheckResult(is_duplicate=False),
        "completed_questions": [],
        "rejected_questions": [],
    }
    state.update(overrides)
    return state


class StructuredEvaluationSchemaTests(unittest.TestCase):
    def test_round_trip_json(self):
        evaluation = _failed_eval("Clarify distractor C")
        payload = evaluation.model_dump()
        restored = StructuredEvaluation.model_validate(payload)
        self.assertFalse(restored.approved)
        self.assertEqual(restored.revision_instructions, "Clarify distractor C")
        self.assertEqual(len(restored.blocking_errors), 1)

        raw = json.dumps(payload)
        parsed = StructuredEvaluation.model_validate_json(raw)
        self.assertEqual(parsed.issues, ["Ambiguous stem"])


class QualityGateRoutingTests(unittest.TestCase):
    def test_all_approved_routes_to_finalize(self):
        route = route_after_quality_gate(_base_state())
        self.assertEqual(route, "finalize_question")

    def test_content_failure_with_rounds_left_routes_to_revise(self):
        state = _base_state(content_evaluation=_failed_eval())
        self.assertEqual(route_after_quality_gate(state), "revise_mcq")

    def test_quality_failure_with_rounds_left_routes_to_revise(self):
        state = _base_state(quality_evaluation=_failed_eval())
        self.assertEqual(route_after_quality_gate(state), "revise_mcq")

    def test_duplicate_routes_to_revise_when_rounds_remain(self):
        state = _base_state(
            duplicate_evaluation=DuplicateCheckResult(
                is_duplicate=True,
                similarity_reason="Same misconception as Q1",
                most_similar_question_id=1,
                revision_instructions="Target a different scenario",
            )
        )
        self.assertEqual(route_after_quality_gate(state), "revise_mcq")

    def test_failed_after_max_rounds_routes_to_reject_not_finalize(self):
        state = _base_state(
            candidate_question=_sample_mcq(revision_rounds=2),
            quality_evaluation=_failed_eval(),
        )
        self.assertEqual(route_after_quality_gate(state), "reject_or_regenerate")

    def test_max_rounds_does_not_finalize_unapproved(self):
        state = _base_state(
            candidate_question=_sample_mcq(revision_rounds=2),
            content_evaluation=_failed_eval(),
            quality_evaluation=_failed_eval(),
        )
        self.assertNotEqual(route_after_quality_gate(state), "finalize_question")

    def test_attempt_cap_routes_to_reject(self):
        state = _base_state(
            generation_attempts=100,
            max_total_attempts=10,
        )
        self.assertEqual(route_after_quality_gate(state), "reject_or_regenerate")


class FinalizeAndRejectNodeTests(unittest.TestCase):
    def test_finalize_only_when_all_checks_pass(self):
        result = finalize_question_node(_base_state())
        self.assertNotIn("error", result)
        self.assertEqual(len(result["completed_questions"]), 1)
        self.assertTrue(result["completed_questions"][0].mcq.approved)

    def test_finalize_refuses_unapproved_candidate(self):
        state = _base_state(quality_evaluation=_failed_eval())
        result = finalize_question_node(state)
        self.assertIn("error", result)
        self.assertNotIn("completed_questions", result)

    def test_reject_adds_to_rejected_not_completed(self):
        state = _base_state(
            candidate_question=_sample_mcq(revision_rounds=2),
            quality_evaluation=_failed_eval(),
        )
        result = reject_or_regenerate_node(state)
        self.assertEqual(len(result["rejected_questions"]), 1)
        self.assertIn("quality", result["rejected_questions"][0].rejection_reason)
        self.assertIsNone(result.get("completed_questions"))


class BatchLoopRoutingTests(unittest.TestCase):
    def test_finalize_loops_to_blueprint_when_more_needed(self):
        state = _base_state(completed_questions=[MagicMock(spec=CompletedMCQRecord)])
        self.assertEqual(route_after_finalize(state), "create_question_blueprint")

    def test_finalize_ends_when_batch_complete(self):
        completed = [MagicMock(spec=CompletedMCQRecord) for _ in range(2)]
        state = _base_state(completed_questions=completed)
        self.assertEqual(route_after_finalize(state), "__end__")

    def test_reject_retries_blueprint_when_under_target(self):
        state = _base_state(rejected_questions=[MagicMock(spec=RejectedQuestionRecord)])
        self.assertEqual(route_after_reject(state), "create_question_blueprint")

    def test_reject_ends_when_attempt_cap_hit(self):
        state = _base_state(
            generation_attempts=20,
            max_total_attempts=10,
            rejected_questions=[MagicMock(spec=RejectedQuestionRecord)],
        )
        self.assertEqual(route_after_reject(state), "__end__")


class LoopGuardTests(unittest.TestCase):
    def test_compute_max_total_attempts_scales_with_batch(self):
        self.assertEqual(compute_max_total_attempts(3, 2), 18)

    @patch("src.graph.build_mcq_graph")
    def test_workflow_stops_without_infinite_loop_on_repeated_reject(
        self, mock_build_graph
    ):
        """Simulate graph returning after attempt cap without hanging."""
        mock_app = MagicMock()
        mock_build_graph.return_value = mock_app
        mock_app.invoke.return_value = {
            "completed_questions": [],
            "rejected_questions": [MagicMock(spec=RejectedQuestionRecord)] * 3,
            "error": "Exceeded max generation attempts (8). Completed 0 of 1 requested.",
            "generation_attempts": 8,
            "max_total_attempts": 8,
        }

        result = run_workflow(
            topic="acids",
            learning_objective="Identify conjugate pairs",
            difficulty="medium",
            num_questions=1,
            max_revision_rounds=2,
        )
        self.assertEqual(result["generation_attempts"], 8)
        self.assertIn("error", result)
        mock_app.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
