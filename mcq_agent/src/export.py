"""Export generated MCQ Markdown to DOCX and PDF via Pandoc."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("mcq_agent.export")

# Pandoc PDF engines tried in order when none is specified.
PDF_ENGINE_CANDIDATES = (
    "pdflatex",
    "xelatex",
    "lualatex",
    "wkhtmltopdf",
    "weasyprint",
    "prince",
    "context",
    "tectonic",
)


class PandocNotFoundError(RuntimeError):
    """Raised when the pandoc executable is not on PATH."""


class PdfEngineNotFoundError(RuntimeError):
    """Raised when no suitable PDF engine is available for Pandoc."""


class ExportError(RuntimeError):
    """Raised when Pandoc conversion fails."""


def find_pandoc() -> str:
    """Return the pandoc executable path or raise PandocNotFoundError."""
    path = shutil.which("pandoc")
    if not path:
        raise PandocNotFoundError(
            "Pandoc is not installed or not on PATH. "
            "Install it from https://pandoc.org/installing.html"
        )
    return path


def find_pdf_engine(preferred: str | None = None) -> str:
    """
    Resolve a Pandoc PDF engine.

    Uses ``preferred`` when set and available; otherwise scans common engines.
    """
    if preferred:
        if shutil.which(preferred):
            return preferred
        raise PdfEngineNotFoundError(
            f"Requested PDF engine not found on PATH: {preferred}"
        )

    for engine in PDF_ENGINE_CANDIDATES:
        if shutil.which(engine):
            logger.info("Using PDF engine: %s", engine)
            return engine

    raise PdfEngineNotFoundError(
        "No PDF engine found for Pandoc. Install one of: "
        f"{', '.join(PDF_ENGINE_CANDIDATES)} "
        "(e.g. TeX Live for pdflatex/xelatex), or pass --pdf-engine explicitly."
    )


def _run_pandoc(args: list[str]) -> None:
    """Run pandoc and raise ExportError on failure."""
    pandoc = find_pandoc()
    command = [pandoc, *args]
    logger.info("Running: %s", " ".join(command))
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or f"exit code {exc.returncode}"
        raise ExportError(f"Pandoc failed: {detail}") from exc


def convert_md_to_docx(
    input_md: Path,
    output_docx: Path,
    *,
    reference_doc: Path | None = None,
) -> Path:
    """Convert a Markdown file to DOCX."""
    if not input_md.is_file():
        raise FileNotFoundError(f"Input Markdown not found: {input_md}")

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(input_md),
        "-f",
        "markdown",
        "-t",
        "docx",
        "-o",
        str(output_docx),
    ]
    if reference_doc:
        if not reference_doc.is_file():
            raise FileNotFoundError(f"Reference DOCX not found: {reference_doc}")
        args.extend(["--reference-doc", str(reference_doc)])

    _run_pandoc(args)
    logger.info("Wrote DOCX: %s", output_docx)
    return output_docx


def convert_md_to_pdf(
    input_md: Path,
    output_pdf: Path,
    *,
    pdf_engine: str | None = None,
) -> Path:
    """Convert a Markdown file to PDF using Pandoc."""
    if not input_md.is_file():
        raise FileNotFoundError(f"Input Markdown not found: {input_md}")

    engine = find_pdf_engine(pdf_engine)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    _run_pandoc(
        [
            str(input_md),
            "-f",
            "markdown",
            "-o",
            str(output_pdf),
            f"--pdf-engine={engine}",
        ]
    )
    logger.info("Wrote PDF: %s", output_pdf)
    return output_pdf


def export_markdown(
    input_md: Path,
    *,
    output_docx: Path | None = None,
    output_pdf: Path | None = None,
    write_docx: bool = True,
    write_pdf: bool = True,
    pdf_engine: str | None = None,
    reference_doc: Path | None = None,
) -> dict[str, Path]:
    """
    Export Markdown to DOCX and/or PDF.

    Returns a dict of format -> output path for each file written.
    """
    input_md = input_md.resolve()
    stem = input_md.with_suffix("")
    output_docx = output_docx or stem.with_suffix(".docx")
    output_pdf = output_pdf or stem.with_suffix(".pdf")

    written: dict[str, Path] = {}

    if write_docx:
        written["docx"] = convert_md_to_docx(
            input_md,
            output_docx,
            reference_doc=reference_doc,
        )

    if write_pdf:
        written["pdf"] = convert_md_to_pdf(
            input_md,
            output_pdf,
            pdf_engine=pdf_engine,
        )

    return written
