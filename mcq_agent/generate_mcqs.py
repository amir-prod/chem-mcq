#!/usr/bin/env python3
"""CLI entry point for the LangGraph MCQ generation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python generate_mcqs.py` from the mcq_agent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import run_workflow
from src.schemas import Difficulty
from src.utils import (
    PROJECT_ROOT,
    persist_workflow_outputs,
    resolve_output_paths,
    setup_logging,
    validate_data_directories,
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
        "--batch_size",
        type=int,
        default=5,
        help="Number of MCQs to blueprint and generate per batch (default: 5).",
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
        "--output_name",
        help=(
            "Base output filename without extension (written under outputs/). "
            "Creates <name>.json, <name>.md, and <name>_rejected.json when needed."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for JSON output (overrides --output_name for JSON).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path for Markdown output (overrides --output_name for Markdown).",
    )
    args = parser.parse_args()
    if not args.index_only and (not args.topic or not args.learning_objective):
        parser.error("--topic and --learning_objective are required unless using --index-only")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    try:
        args.output_json, args.output_md, args.output_rejected_json = resolve_output_paths(
            output_name=args.output_name,
            output_json=args.output_json,
            output_md=args.output_md,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _save_partial(
    *,
    topic: str,
    learning_objective: str,
    difficulty: str,
    completed,
    rejected,
    output_json: Path,
    output_md: Path,
    output_rejected_json: Path,
    logger,
) -> bool:
    """Write whatever questions exist; return True if anything was saved."""
    paths = persist_workflow_outputs(
        topic=topic,
        learning_objective=learning_objective,
        difficulty=difficulty,
        completed_questions=completed or [],
        rejected_questions=rejected or [],
        json_path=output_json,
        md_path=output_md,
        rejected_json_path=output_rejected_json,
    )
    if paths is None:
        return False
    logger.info(
        "Wrote %d approved / %d rejected to %s and %s",
        len(completed or []),
        len(rejected or []),
        paths[0],
        paths[1],
    )
    return True


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

    final_state: dict = {}
    interrupted = False
    try:
        ensure_indexes(rebuild=args.rebuild_index)
        final_state = run_workflow(
            topic=args.topic,
            learning_objective=args.learning_objective,
            difficulty=args.difficulty,
            num_questions=args.num_questions,
            batch_size=args.batch_size,
            max_revision_rounds=args.max_revision_rounds,
            rebuild_indexes=args.rebuild_index,
            output_json=args.output_json,
            output_md=args.output_md,
            output_rejected_json=args.output_rejected_json,
        )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted before workflow returned; checking for checkpoints.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow failed: %s", exc)
        # Incremental checkpoints may already be on disk; still try a final flush.
        questions = final_state.get("completed_questions", [])
        rejected = final_state.get("rejected_questions", [])
        if questions or rejected:
            _save_partial(
                topic=args.topic,
                learning_objective=args.learning_objective,
                difficulty=args.difficulty,
                completed=questions,
                rejected=rejected,
                output_json=args.output_json,
                output_md=args.output_md,
                output_rejected_json=args.output_rejected_json,
                logger=logger,
            )
        return 1

    if final_state.get("error") == "Interrupted by user":
        interrupted = True

    questions = final_state.get("completed_questions", [])
    rejected = final_state.get("rejected_questions", [])

    saved = _save_partial(
        topic=args.topic,
        learning_objective=args.learning_objective,
        difficulty=args.difficulty,
        completed=questions,
        rejected=rejected,
        output_json=args.output_json,
        output_md=args.output_md,
        output_rejected_json=args.output_rejected_json,
        logger=logger,
    )

    if final_state.get("error") and not interrupted:
        logger.error("Workflow error: %s", final_state["error"])

    if not questions:
        logger.error(
            "No approved questions were generated (%d rejected).%s",
            len(rejected),
            " Partial rejected output was saved." if saved else "",
        )
        return 130 if interrupted else 1

    if len(questions) < args.num_questions:
        logger.warning(
            "Only %d of %d requested questions were approved (%d rejected).",
            len(questions),
            args.num_questions,
            len(rejected),
        )

    if interrupted:
        logger.warning(
            "Run interrupted; kept %d approved question(s) on disk.",
            len(questions),
        )
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
