"""Pydantic models for MCQ workflow state and structured outputs."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CognitiveLevel(str, Enum):
    RECALL = "recall"
    APPLICATION = "application"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"


class MCQOptions(BaseModel):
    """Four-option multiple-choice answer set."""

    A: str
    B: str
    C: str
    D: str


class MCQQuestion(BaseModel):
    """Structured representation of a generated multiple-choice question."""

    question: str
    options: MCQOptions
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str
    learning_objective: str
    difficulty: Difficulty
    cognitive_level: CognitiveLevel
    source_context_used: list[str] = Field(default_factory=list)
    evaluator_feedback: str = ""
    revision_rounds: int = 0
    approved: bool = False

    @field_validator("correct_answer", mode="before")
    @classmethod
    def normalize_correct_answer(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip().upper()
        return value


class QuestionBlueprint(BaseModel):
    """Structured plan for an MCQ before generation."""

    topic: str
    learning_objective: str
    difficulty: str
    cognitive_level: str
    target_misconception: str
    correct_answer_concept: str
    expected_reasoning: str
    question_style_notes: str = ""


class StructuredEvaluation(BaseModel):
    """Structured output from content or quality evaluators."""

    approved: bool
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    issues: list[str] = Field(default_factory=list)
    revision_instructions: str = ""
    blocking_errors: list[str] = Field(default_factory=list)


class DuplicateCheckResult(BaseModel):
    """Result of duplicate / near-duplicate detection."""

    is_duplicate: bool
    similarity_reason: str = ""
    most_similar_question_id: int | None = None
    revision_instructions: str = ""


class EvaluationResult(BaseModel):
    """Legacy evaluator output (kept for backward compatibility)."""

    approved: bool
    feedback: str
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class CompletedMCQRecord(BaseModel):
    """A finalized question with full audit trail."""

    mcq: MCQQuestion
    question_blueprint: QuestionBlueprint
    revision_rounds: int
    content_evaluation: StructuredEvaluation
    quality_evaluation: StructuredEvaluation
    duplicate_evaluation: DuplicateCheckResult


class RejectedQuestionRecord(BaseModel):
    """A question that failed after max revision rounds."""

    question_blueprint: QuestionBlueprint | None = None
    last_candidate: MCQQuestion | None = None
    rejection_reason: str
    revision_rounds: int = 0
    content_evaluation: StructuredEvaluation | None = None
    quality_evaluation: StructuredEvaluation | None = None
    duplicate_evaluation: DuplicateCheckResult | None = None


class GenerationRequest(BaseModel):
    """User inputs for MCQ generation."""

    topic: str
    learning_objective: str
    difficulty: Difficulty = Difficulty.MEDIUM
    num_questions: int = Field(default=1, ge=1, le=50)
    max_revision_rounds: int = Field(default=3, ge=0, le=10)


class DocumentMetadata(BaseModel):
    """Metadata attached to ingested documents."""

    source_folder: str
    filename: str
    document_type: Literal["md", "json", "png", "image"]
    collection: Literal["exam_examples", "best_practices"]
    relative_path: str


class IngestedDocument(BaseModel):
    """A single chunk-ready document from the ingestion pipeline."""

    content: str
    metadata: DocumentMetadata


class WorkflowState(TypedDict, total=False):
    """LangGraph state shared across workflow nodes."""

    # User request
    topic: str
    learning_objective: str
    difficulty: str
    num_questions: int
    max_revision_rounds: int
    max_total_attempts: int

    # Retrieval context (per-question cycle)
    exam_examples_context: list[str]
    best_practices_context: list[str]

    # Per-question working state
    current_question_index: int
    question_blueprint: QuestionBlueprint | None
    candidate_question: MCQQuestion | None
    content_evaluation: StructuredEvaluation | None
    quality_evaluation: StructuredEvaluation | None
    duplicate_evaluation: DuplicateCheckResult | None

    # Accumulated outputs
    completed_questions: Annotated[list[CompletedMCQRecord], lambda left, right: left + right]
    rejected_questions: Annotated[list[RejectedQuestionRecord], lambda left, right: left + right]

    # Loop guards
    generation_attempts: int

    # Control flags
    error: str | None


class MCQBatchOutput(BaseModel):
    """Final batch output written to disk."""

    topic: str
    learning_objective: str
    difficulty: str
    num_questions: int
    questions: list[CompletedMCQRecord]
    rejected_questions: list[RejectedQuestionRecord] = Field(default_factory=list)


def compute_max_total_attempts(num_questions: int, max_revision_rounds: int) -> int:
    """Upper bound on blueprint/generation cycles to prevent infinite loops."""
    return num_questions * (max_revision_rounds + 4)
