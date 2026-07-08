"""Document ingestion pipeline for misconceptions and best-practice materials."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from src.schemas import DocumentMetadata, IngestedDocument
from src.utils import (
    BEST_PRACTICES_DIR,
    BEST_PRACTICES_COLLECTION,
    MISCONCEPTIONS_COLLECTION,
    MISCONCEPTIONS_DIR,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger("mcq_agent.ingestion")


class ImageTextExtractor(ABC):
    """Pluggable interface for OCR / image captioning on PNG files."""

    @abstractmethod
    def extract(self, image_path: Path) -> str:
        """Return text extracted from an image file."""


class PlaceholderImageTextExtractor(ImageTextExtractor):
    """
    Placeholder OCR/captioning backend.

    Replace this class with a real vision model or OCR service when ready.
    """

    def extract(self, image_path: Path) -> str:
        return (
            f"[Image placeholder: {image_path.name}. "
            "Plug in OCR or a vision model via ImageTextExtractor.]"
        )


def _collection_for_root(root: Path) -> str:
    """Map a data root path to its vector collection name."""
    resolved = root.resolve()
    if resolved == BEST_PRACTICES_DIR.resolve():
        return BEST_PRACTICES_COLLECTION
    if resolved == MISCONCEPTIONS_DIR.resolve():
        return MISCONCEPTIONS_COLLECTION
    name = root.name.lower()
    if "misconception" in name or name.startswith("mds_"):
        return MISCONCEPTIONS_COLLECTION
    if "practice" in name or "best" in name:
        return BEST_PRACTICES_COLLECTION
    raise ValueError(f"Cannot infer collection for data root: {root}")


def _extract_json_text(payload: object, prefix: str = "") -> list[str]:
    """Recursively pull human-readable strings from JSON structures."""
    texts: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in {"polygon", "debug_data_path", "block_metadata"}:
                continue
            texts.extend(_extract_json_text(value, child_prefix))
    elif isinstance(payload, list):
        for item in payload:
            texts.extend(_extract_json_text(item, prefix))
    elif isinstance(payload, (str, int, float, bool)):
        text = str(payload).strip()
        if text and len(text) > 1:
            label = f"{prefix}: " if prefix else ""
            texts.append(f"{label}{text}")
    return texts


def load_markdown(path: Path) -> str:
    """Read markdown file content."""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_json(path: Path) -> str:
    """Parse JSON and flatten useful text fields."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Skipping invalid JSON %s: %s", path, exc)
        return ""
    parts = _extract_json_text(data)
    return "\n".join(parts).strip()


def load_image(
    path: Path,
    extractor: ImageTextExtractor | None = None,
) -> str:
    """Load PNG (or other image) content via the configured extractor."""
    backend = extractor or PlaceholderImageTextExtractor()
    return backend.extract(path).strip()


def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield supported files under root recursively."""
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def ingest_file(
    path: Path,
    root: Path,
    collection: str,
    image_extractor: ImageTextExtractor | None = None,
) -> IngestedDocument | None:
    """Load a single file and attach metadata."""
    suffix = path.suffix.lower()
    relative = str(path.relative_to(root))
    source_folder = str(path.parent.relative_to(root))

    try:
        if suffix == ".md":
            content = load_markdown(path)
            doc_type = "md"
        elif suffix == ".json":
            content = load_json(path)
            doc_type = "json"
        elif suffix == ".png":
            content = load_image(path, image_extractor)
            doc_type = "png"
        else:
            logger.debug("Unsupported file skipped: %s", path)
            return None
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    if not content:
        logger.debug("Empty content after ingestion: %s", path)
        return None

    metadata = DocumentMetadata(
        source_folder=source_folder,
        filename=path.name,
        document_type=doc_type,
        collection=collection,
        relative_path=relative,
    )
    return IngestedDocument(content=content, metadata=metadata)


def ingest_directory(
    root: Path,
    collection: str | None = None,
    image_extractor: ImageTextExtractor | None = None,
) -> list[IngestedDocument]:
    """Ingest all supported files from a directory tree."""
    collection_name = collection or _collection_for_root(root)
    documents: list[IngestedDocument] = []

    try:
        files = list(iter_source_files(root))
    except FileNotFoundError:
        logger.error("Directory missing: %s", root)
        return documents

    if not files:
        logger.warning("No supported files found under %s", root)

    for path in files:
        doc = ingest_file(path, root, collection_name, image_extractor)
        if doc:
            documents.append(doc)

    logger.info(
        "Ingested %d documents from %s (%s)",
        len(documents),
        root,
        collection_name,
    )
    return documents


def ingest_all(
    image_extractor: ImageTextExtractor | None = None,
) -> dict[str, list[IngestedDocument]]:
    """Ingest misconception and best-practice corpora."""
    if not MISCONCEPTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Misconceptions directory not found: {MISCONCEPTIONS_DIR}"
        )

    corpora = {
        MISCONCEPTIONS_COLLECTION: ingest_directory(
            MISCONCEPTIONS_DIR,
            MISCONCEPTIONS_COLLECTION,
            image_extractor,
        ),
        BEST_PRACTICES_COLLECTION: ingest_directory(
            BEST_PRACTICES_DIR,
            BEST_PRACTICES_COLLECTION,
            image_extractor,
        ),
    }
    return corpora
