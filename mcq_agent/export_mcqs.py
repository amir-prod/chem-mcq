#!/usr/bin/env python3
"""Convert generated_mcqs.md to DOCX and PDF using Pandoc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.export import (
    ExportError,
    PandocNotFoundError,
    PdfEngineNotFoundError,
    export_markdown,
)
from src.utils import OUTPUT_DIR, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export generated MCQ Markdown to DOCX and PDF via Pandoc.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_DIR / "generated_mcqs.md",
        help="Input Markdown file (default: outputs/generated_mcqs.md).",
    )
    parser.add_argument(
        "--output-docx",
        type=Path,
        default=None,
        help="Output DOCX path (default: same name as input with .docx).",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Output PDF path (default: same name as input with .pdf).",
    )
    parser.add_argument(
        "--docx-only",
        action="store_true",
        help="Only produce the DOCX file.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Only produce the PDF file.",
    )
    parser.add_argument(
        "--pdf-engine",
        default=None,
        help="Pandoc PDF engine (e.g. pdflatex, xelatex, wkhtmltopdf). "
        "Auto-detected when omitted.",
    )
    parser.add_argument(
        "--reference-doc",
        type=Path,
        default=None,
        help="Optional Pandoc reference DOCX for custom Word styling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging()

    if args.docx_only and args.pdf_only:
        logger.error("Use only one of --docx-only or --pdf-only.")
        return 1

    write_docx = not args.pdf_only
    write_pdf = not args.docx_only

    try:
        written = export_markdown(
            args.input,
            output_docx=args.output_docx,
            output_pdf=args.output_pdf,
            write_docx=write_docx,
            write_pdf=write_pdf,
            pdf_engine=args.pdf_engine,
            reference_doc=args.reference_doc,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except PandocNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except PdfEngineNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ExportError as exc:
        logger.error("%s", exc)
        return 1

    for fmt, path in written.items():
        logger.info("Exported %s -> %s", fmt.upper(), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
