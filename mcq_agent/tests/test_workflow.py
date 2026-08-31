"""Tests for workflow routing, schemas, and quality-gate behavior."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from src.graph import (
    _current_batch_count,
    _remaining_questions,
    advance_batch_node,
    finalize_question_node,
    reject_or_regenerate_node,
    route_after_advance,
    route_after_quality_gate,
    run_workflow,
    _finalize_question,
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
    StemMediaType,
    StructuredEvaluation,
    allocate_stem_media_types,
    compute_max_total_attempts,
)
from src.utils import get_embedding_config, get_model_config, persist_workflow_outputs, DEFAULT_OPENAI_BASE_URL



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
        "batch_size": 5,
        "max_revision_rounds": 2,
        "max_total_attempts": compute_max_total_attempts(2, 2),
        "generation_attempts": 1,
        "batch_start_index": 1,
        "batch_index": 0,
        "batch_blueprints": [_sample_blueprint()],
        "batch_candidates": [_sample_mcq()],
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


class BatchRoutingTests(unittest.TestCase):
    def test_remaining_questions_accounts_for_completed(self):
        state = _base_state(completed_questions=[MagicMock(spec=CompletedMCQRecord)])
        self.assertEqual(_remaining_questions(state), 1)

    def test_current_batch_count_uses_remaining_when_smaller_than_batch_size(self):
        state = _base_state(num_questions=7, batch_size=5, completed_questions=[])
        self.assertEqual(_current_batch_count(state), 5)

        state = _base_state(
            num_questions=7,
            batch_size=5,
            completed_questions=[MagicMock(spec=CompletedMCQRecord) for _ in range(5)],
        )
        self.assertEqual(_current_batch_count(state), 2)

    def test_advance_batch_increments_index(self):
        result = advance_batch_node(_base_state(batch_index=0))
        self.assertEqual(result["batch_index"], 1)

    def test_advance_routes_to_next_question_in_batch(self):
        state = _base_state(
            batch_index=1,
            batch_blueprints=[_sample_blueprint(), _sample_blueprint()],
            batch_candidates=[_sample_mcq(), _sample_mcq()],
        )
        self.assertEqual(route_after_advance(state), "select_batch_question")

    def test_advance_routes_to_new_batch_when_current_batch_exhausted(self):
        state = _base_state(
            batch_index=2,
            batch_blueprints=[_sample_blueprint(), _sample_blueprint()],
            batch_candidates=[_sample_mcq(), _sample_mcq()],
            completed_questions=[MagicMock(spec=CompletedMCQRecord)],
            num_questions=5,
        )
        self.assertEqual(route_after_advance(state), "retrieve_misconceptions")

    def test_advance_ends_when_target_reached(self):
        state = _base_state(
            batch_index=1,
            batch_blueprints=[_sample_blueprint()],
            completed_questions=[MagicMock(spec=CompletedMCQRecord) for _ in range(2)],
            num_questions=2,
        )
        self.assertEqual(route_after_advance(state), "__end__")

    def test_advance_ends_when_attempt_cap_hit(self):
        state = _base_state(
            generation_attempts=20,
            max_total_attempts=10,
            batch_index=1,
            batch_blueprints=[_sample_blueprint(), _sample_blueprint()],
        )
        self.assertEqual(route_after_advance(state), "__end__")


class BatchGenerationFallbackTests(unittest.TestCase):
    def test_generate_batch_fills_missing_questions_one_by_one(self):
        from src.agents import generate_batch_mcqs
        from src.schemas import MCQQuestionBatch

        blueprints = [_sample_blueprint(), _sample_blueprint(), _sample_blueprint()]
        partial = MCQQuestionBatch(questions=[_sample_mcq()])
        filled = [_sample_mcq(question="filled-2"), _sample_mcq(question="filled-3")]

        with patch("src.agents._invoke_structured", return_value=partial), patch(
            "src.agents.generate_mcq", side_effect=filled
        ) as mock_single:
            result = generate_batch_mcqs(
                blueprints=blueprints,
                misconceptions_context=["ctx"],
                best_practices_context=["bp"],
                model=MagicMock(),
            )

        self.assertEqual(len(result.questions), 3)
        self.assertEqual(mock_single.call_count, 2)
        self.assertEqual(result.questions[1].question, "filled-2")
        self.assertEqual(result.questions[2].question, "filled-3")


class ModelEmbeddingConfigTests(unittest.TestCase):
    def test_embedding_uses_dedicated_key_and_ignores_local_chat_base(self):
        env = {
            "OPENAI_API_KEY": "sk-openwebui",
            "OPENAI_BASE_URL": "http://localhost:8080/api/v1",
            "OPENAI_MODEL": "qwen36-35b",
            "EMBEDDING_API_KEY": "sk-openai-embed",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
            "OPENAI_TEMPERATURE": "0.3",
        }
        with patch("src.utils.load_dotenv"), patch.dict(os.environ, env, clear=False):
            for key in ("EMBEDDING_BASE_URL",):
                os.environ.pop(key, None)
            chat = get_model_config()
            embed = get_embedding_config()

        self.assertEqual(chat["api_key"], "sk-openwebui")
        self.assertEqual(chat["base_url"], "http://localhost:8080/api/v1")
        self.assertEqual(chat["model"], "qwen36-35b")
        self.assertEqual(embed["api_key"], "sk-openai-embed")
        self.assertEqual(embed["base_url"], DEFAULT_OPENAI_BASE_URL)
        self.assertEqual(embed["model"], "text-embedding-3-large")

    def test_embedding_falls_back_to_openai_key_when_no_embedding_key(self):
        env = {
            "OPENAI_API_KEY": "sk-shared",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        }
        with patch("src.utils.load_dotenv"), patch.dict(os.environ, env, clear=False):
            os.environ.pop("EMBEDDING_API_KEY", None)
            os.environ.pop("EMBEDDING_BASE_URL", None)
            embed = get_embedding_config()

        self.assertEqual(embed["api_key"], "sk-shared")
        self.assertEqual(embed["base_url"], "https://api.openai.com/v1")

    def test_explicit_embedding_base_url_empty_means_openai_default(self):
        env = {
            "OPENAI_API_KEY": "sk-openwebui",
            "OPENAI_BASE_URL": "http://localhost:8080/api/v1",
            "EMBEDDING_API_KEY": "sk-openai-embed",
            "EMBEDDING_BASE_URL": "",
        }
        with patch("src.utils.load_dotenv"), patch.dict(os.environ, env, clear=False):
            embed = get_embedding_config()

        self.assertEqual(embed["api_key"], "sk-openai-embed")
        self.assertEqual(embed["base_url"], DEFAULT_OPENAI_BASE_URL)


class StemMediaAllocationTests(unittest.TestCase):
    def test_n1_is_text_only(self):
        plan = allocate_stem_media_types(1)
        self.assertEqual(plan, [StemMediaType.TEXT])

    def test_n2_has_one_table_and_one_figure(self):
        plan = allocate_stem_media_types(2)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan.count(StemMediaType.TABLE), 1)
        self.assertEqual(plan.count(StemMediaType.FIGURE), 1)

    def test_n5_has_one_table_one_figure_three_text(self):
        plan = allocate_stem_media_types(5)
        self.assertEqual(len(plan), 5)
        self.assertEqual(plan.count(StemMediaType.TABLE), 1)
        self.assertEqual(plan.count(StemMediaType.FIGURE), 1)
        self.assertEqual(plan.count(StemMediaType.TEXT), 3)

    def test_n7_scales_to_about_twenty_percent(self):
        plan = allocate_stem_media_types(7)
        self.assertEqual(len(plan), 7)
        self.assertEqual(plan.count(StemMediaType.TABLE), max(1, round(7 * 0.2)))
        self.assertEqual(plan.count(StemMediaType.FIGURE), max(1, round(7 * 0.2)))
        self.assertEqual(
            plan.count(StemMediaType.TEXT),
            7 - plan.count(StemMediaType.TABLE) - plan.count(StemMediaType.FIGURE),
        )

    def test_schema_defaults_to_text_media(self):
        blueprint = _sample_blueprint()
        mcq = _sample_mcq()
        self.assertEqual(blueprint.stem_media_type, StemMediaType.TEXT)
        self.assertEqual(mcq.stem_media_type, StemMediaType.TEXT)
        self.assertEqual(mcq.media_content, "")


class CheckpointPersistenceTests(unittest.TestCase):
    def test_persist_returns_none_when_empty(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = persist_workflow_outputs(
                topic="t",
                learning_objective="lo",
                difficulty="hard",
                completed_questions=[],
                rejected_questions=[],
                json_path=root / "out.json",
                md_path=root / "out.md",
                rejected_json_path=root / "out_rejected.json",
            )
            self.assertIsNone(result)
            self.assertFalse((root / "out.json").exists())

    def test_finalize_checkpoint_writes_partial_outputs(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "partial.json"
            md_path = root / "partial.md"
            rejected_path = root / "partial_rejected.json"
            written = {}

            def checkpoint(state, *, completed=None, rejected=None):
                written["completed"] = completed
                written["rejected"] = rejected
                persist_workflow_outputs(
                    topic=state["topic"],
                    learning_objective=state["learning_objective"],
                    difficulty=state["difficulty"],
                    completed_questions=completed or [],
                    rejected_questions=rejected or [],
                    json_path=json_path,
                    md_path=md_path,
                    rejected_json_path=rejected_path,
                )

            result = _finalize_question(_base_state(), checkpoint=checkpoint)
            self.assertEqual(len(result["completed_questions"]), 1)
            self.assertEqual(len(written["completed"]), 1)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            payload = json_path.read_text(encoding="utf-8")
            self.assertIn("stoichiometry", payload)


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
        final = {
            "completed_questions": [],
            "rejected_questions": [MagicMock(spec=RejectedQuestionRecord)] * 3,
            "error": "Exceeded max generation attempts (8). Completed 0 of 1 requested.",
            "generation_attempts": 8,
            "max_total_attempts": 8,
        }
        mock_app.stream.return_value = iter([final])

        result = run_workflow(
            topic="acids",
            learning_objective="Identify conjugate pairs",
            difficulty="medium",
            num_questions=1,
            batch_size=5,
            max_revision_rounds=2,
        )
        self.assertEqual(result["generation_attempts"], 8)
        self.assertIn("error", result)
        mock_app.stream.assert_called_once()


if __name__ == "__main__":
    unittest.main()
