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


class EvaluationResult(BaseModel):
    """Evaluator agent output for a single MCQ."""

    approved: bool
    feedback: str
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


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

    # Retrieval context (cached for the run)
    exam_context: list[str]
    best_practices_context: list[str]

    # Per-question working state
    current_question_index: int
    current_question: MCQQuestion | None
    evaluation: EvaluationResult | None

    # Accumulated outputs
    completed_questions: Annotated[list[MCQQuestion], lambda left, right: left + right]

    # Control flags
    error: str | None


class MCQBatchOutput(BaseModel):
    """Final batch output written to disk."""

    topic: str
    learning_objective: str
    difficulty: str
    num_questions: int
    questions: list[MCQQuestion]
