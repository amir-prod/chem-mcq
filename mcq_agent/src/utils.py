"""Shared utilities: paths, configuration, logging, and output writers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.schemas import MCQBatchOutput, MCQQuestion

# Project root is the mcq_agent directory (parent of src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MCQ_DATA_DIR", PROJECT_ROOT / "data"))
EXAMS_DIR = Path(os.getenv("MCQ_EXAMS_DIR", DATA_DIR / "mds_exams"))
BEST_PRACTICES_DIR = Path(os.getenv("MCQ_BEST_PRACTICES_DIR", DATA_DIR / "mds_best_practices"))
VECTORSTORE_DIR = Path(os.getenv("MCQ_VECTORSTORE_DIR", PROJECT_ROOT / ".chroma"))
OUTPUT_DIR = Path(os.getenv("MCQ_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

EXAM_COLLECTION = "exam_examples"
BEST_PRACTICES_COLLECTION = "best_practices"

SUPPORTED_EXTENSIONS = {".md", ".json", ".png"}


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure root logger and return the mcq_agent logger."""
    load_dotenv(PROJECT_ROOT / ".env")
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("mcq_agent")


def get_model_config() -> dict[str, str | float]:
    """Load OpenAI-compatible model settings from environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL") or None,
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
    }


def get_embedding_config() -> dict[str, str]:
    """Load embedding model settings from environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL") or None,
    }


def get_retrieval_config() -> dict[str, int]:
    """Default retrieval settings (overridable via env)."""
    return {
        "exam_k": int(os.getenv("RETRIEVAL_EXAM_K", "6")),
        "best_practices_k": int(os.getenv("RETRIEVAL_BEST_PRACTICES_K", "5")),
        "chunk_size": int(os.getenv("INGESTION_CHUNK_SIZE", "1200")),
        "chunk_overlap": int(os.getenv("INGESTION_CHUNK_OVERLAP", "200")),
    }


def ensure_directories() -> None:
    """Create required runtime directories if missing."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_data_directories(logger: logging.Logger | None = None) -> None:
    """Raise FileNotFoundError when expected data folders are absent."""
    log = logger or logging.getLogger("mcq_agent")
    for label, path in (
        ("exams", EXAMS_DIR),
        ("best practices", BEST_PRACTICES_DIR),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label} directory: {path}. "
                "Place data under mcq_agent/data/ or set MCQ_DATA_DIR."
            )
        if not any(path.rglob("*")):
            log.warning("Data directory is empty: %s", path)


def write_outputs(
    batch: MCQBatchOutput,
    json_path: Path | None = None,
    md_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write generated MCQs to JSON and Markdown files."""
    ensure_directories()
    json_path = json_path or OUTPUT_DIR / "generated_mcqs.json"
    md_path = md_path or OUTPUT_DIR / "generated_mcqs.md"

    json_path.write_text(
        batch.model_dump_json(indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_format_markdown(batch), encoding="utf-8")
    return json_path, md_path


def _format_markdown(batch: MCQBatchOutput) -> str:
    """Render MCQ batch as human-readable Markdown."""
    lines = [
        "# Generated Multiple-Choice Questions",
        "",
        f"**Topic:** {batch.topic}",
        f"**Learning objective:** {batch.learning_objective}",
        f"**Difficulty:** {batch.difficulty}",
        f"**Count:** {len(batch.questions)}",
        "",
    ]
    for index, question in enumerate(batch.questions, start=1):
        lines.extend(_format_question_markdown(index, question))
    return "\n".join(lines)


def _format_question_markdown(index: int, question: MCQQuestion) -> list[str]:
    status = "Approved" if question.approved else "Not approved (max revisions reached)"
    lines = [
        f"## Question {index}",
        "",
        f"**Status:** {status}  ",
        f"**Revision rounds:** {question.revision_rounds}  ",
        f"**Cognitive level:** {question.cognitive_level.value}  ",
        "",
        question.question,
        "",
    ]
    for letter in ("A", "B", "C", "D"):
        marker = " ✓" if letter == question.correct_answer else ""
        lines.append(f"- **{letter}.** {getattr(question.options, letter)}{marker}")
    lines.extend(
        [
            "",
            f"**Explanation:** {question.explanation}",
            "",
            f"**Evaluator feedback:** {question.evaluator_feedback or 'None'}",
            "",
            "---",
            "",
        ]
    )
    return lines


def dumps_json(data: object) -> str:
    """JSON dump helper that handles Pydantic models."""
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=2)
    return json.dumps(data, indent=2, default=str)
