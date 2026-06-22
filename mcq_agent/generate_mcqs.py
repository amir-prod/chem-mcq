#!/usr/bin/env python3
"""CLI entry point for the LangGraph MCQ generation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python generate_mcqs.py` from the mcq_agent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import run_workflow
from src.schemas import Difficulty, MCQBatchOutput
from src.utils import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    setup_logging,
    validate_data_directories,
    write_outputs,
)
from src.vectorstore import VectorStoreManager, ensure_indexes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate chemistry MCQs with a LangGraph agent workflow.",
    )
    parser.add_argument(
        "--topic",
        help="Subject topic for the questions (required unless --index-only).",
    )
    parser.add_argument(
        "--learning_objective",
        help="Measurable learning objective the questions should assess "
        "(required unless --index-only).",
    )
    parser.add_argument(
        "--difficulty",
        choices=[d.value for d in Difficulty],
        default=Difficulty.MEDIUM.value,
        help="Target difficulty level.",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=1,
        help="Number of MCQs to generate.",
    )
    parser.add_argument(
        "--max_revision_rounds",
        type=int,
        default=3,
        help="Maximum evaluate/revise loops per question.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild vector indexes from source documents before generation.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only build vector indexes and exit (no generation).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_DIR / "generated_mcqs.json",
        help="Path for JSON output.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=OUTPUT_DIR / "generated_mcqs.md",
        help="Path for Markdown output.",
    )
    args = parser.parse_args()
    if not args.index_only and (not args.topic or not args.learning_objective):
        parser.error("--topic and --learning_objective are required unless using --index-only")
    return args


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    logger.info("Project root: %s", PROJECT_ROOT)

    try:
        validate_data_directories(logger)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.index_only:
        try:
            manager = VectorStoreManager()
            counts = manager.build_indexes(reset=args.rebuild_index)
            logger.info("Indexing complete: %s", counts)
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.exception("Indexing failed: %s", exc)
            return 1

    try:
        ensure_indexes(rebuild=args.rebuild_index)
        final_state = run_workflow(
            topic=args.topic,
            learning_objective=args.learning_objective,
            difficulty=args.difficulty,
            num_questions=args.num_questions,
            max_revision_rounds=args.max_revision_rounds,
            rebuild_indexes=args.rebuild_index,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow failed: %s", exc)
        return 1

    if final_state.get("error"):
        logger.error("Workflow error: %s", final_state["error"])
        return 1

    questions = final_state.get("completed_questions", [])
    rejected = final_state.get("rejected_questions", [])
    if not questions:
        logger.error(
            "No approved questions were generated (%d rejected).",
            len(rejected),
        )
        return 1

    if len(questions) < args.num_questions:
        logger.warning(
            "Only %d of %d requested questions were approved (%d rejected).",
            len(questions),
            args.num_questions,
            len(rejected),
        )

    batch = MCQBatchOutput(
        topic=args.topic,
        learning_objective=args.learning_objective,
        difficulty=args.difficulty,
        num_questions=len(questions),
        questions=questions,
        rejected_questions=rejected,
    )
    json_path, md_path = write_outputs(batch, args.output_json, args.output_md)
    logger.info(
        "Wrote %d approved question(s) to %s and %s (%d rejected)",
        len(questions),
        json_path,
        md_path,
        len(rejected),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
