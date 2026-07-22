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


class StemMediaType(str, Enum):
    """How the stem presents supporting media (text-native only)."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


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
    stem_media_type: StemMediaType = StemMediaType.TEXT
    media_content: str = ""
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
    stem_media_type: StemMediaType = StemMediaType.TEXT


class QuestionBlueprintBatch(BaseModel):
    """Structured plans for a batch of MCQs before generation."""

    blueprints: list[QuestionBlueprint] = Field(min_length=1)


class MCQQuestionBatch(BaseModel):
    """Structured batch of generated MCQs."""

    questions: list[MCQQuestion] = Field(min_length=1)


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
    collection: Literal["best_practices", "misconceptions"]
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
    batch_size: int
    max_revision_rounds: int
    max_total_attempts: int

    # Retrieval context (per batch cycle)
    misconceptions_context: list[str]
    best_practices_context: list[str]

    # Batch working state
    batch_start_index: int
    batch_blueprints: list[QuestionBlueprint]
    batch_candidates: list[MCQQuestion]
    batch_index: int

    # Per-question working state (view into current batch slot)
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

    # Precomputed stem media mix for the full run (length == num_questions)
    stem_media_plan: list[StemMediaType]

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


def allocate_stem_media_types(num_questions: int) -> list[StemMediaType]:
    """
    Assign stem media types across a full generation run.

    Target mix: ~20% table, ~20% figure, remainder text.
    For n >= 5, round(n * 0.2) of each media type (at least 1).
    For 2 <= n < 5, one table and one figure (remainder text).
    For n == 1, text only.
    Types are spread across slots to avoid clustering media items.
    """
    if num_questions < 1:
        return []

    if num_questions == 1:
        return [StemMediaType.TEXT]

    if num_questions < 5:
        n_table = 1
        n_figure = 1 if num_questions >= 2 else 0
        if n_table + n_figure > num_questions:
            n_figure = num_questions - n_table
    else:
        n_table = max(1, round(num_questions * 0.2))
        n_figure = max(1, round(num_questions * 0.2))

    if n_table + n_figure > num_questions:
        n_figure = max(0, num_questions - n_table)
        if n_table + n_figure > num_questions:
            n_table = num_questions
            n_figure = 0

    n_text = num_questions - n_table - n_figure
    media = [StemMediaType.TABLE] * n_table + [StemMediaType.FIGURE] * n_figure
    return _spread_media_slots(media, num_questions)


def _spread_media_slots(
    media: list[StemMediaType], num_questions: int
) -> list[StemMediaType]:
    """Place table/figure slots at evenly spaced indices; fill the rest with text."""
    if not media:
        return [StemMediaType.TEXT] * num_questions

    result: list[StemMediaType] = [StemMediaType.TEXT] * num_questions
    step = num_questions / (len(media) + 1)
    positions: list[int] = []
    for i in range(1, len(media) + 1):
        pos = min(num_questions - 1, max(0, round(i * step) - 1))
        while pos in positions:
            pos = (pos + 1) % num_questions
        positions.append(pos)

    for pos, media_type in zip(sorted(positions), media, strict=True):
        result[pos] = media_type
    return result
