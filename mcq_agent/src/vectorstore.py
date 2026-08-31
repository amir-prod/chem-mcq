"""Vector store management and retrieval for RAG."""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion import ingest_all
from src.schemas import IngestedDocument
from src.utils import (
    BEST_PRACTICES_COLLECTION,
    MISCONCEPTIONS_COLLECTION,
    MISCONCEPTIONS_DIR,
    VECTORSTORE_DIR,
    get_embedding_config,
    get_retrieval_config,
)

logger = logging.getLogger("mcq_agent.vectorstore")


class VectorStoreManager:
    """Build, persist, and query separate Chroma collections."""

    def __init__(
        self,
        persist_directory: str | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.persist_directory = str(persist_directory or VECTORSTORE_DIR)
        self.embeddings = embeddings or self._build_embeddings()
        self._stores: dict[str, Chroma] = {}
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=get_retrieval_config()["chunk_size"],
            chunk_overlap=get_retrieval_config()["chunk_overlap"],
        )

    @staticmethod
    def _build_embeddings() -> OpenAIEmbeddings:
        config = get_embedding_config()
        if not config["api_key"]:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to mcq_agent/.env before indexing."
            )
        kwargs: dict[str, Any] = {
            "model": config["model"],
            "api_key": config["api_key"],
        }
        # Always pass base_url explicitly. Omitting it lets the OpenAI SDK inherit
        # OPENAI_BASE_URL from the environment (often a local chat server).
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        return OpenAIEmbeddings(**kwargs)

    def _get_store(self, collection_name: str) -> Chroma:
        if collection_name not in self._stores:
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._stores[collection_name]

    def _documents_from_ingested(
        self,
        documents: list[IngestedDocument],
    ) -> list[Document]:
        """Convert ingested docs to LangChain Documents and chunk them."""
        lc_docs: list[Document] = []
        for doc in documents:
            lc_docs.append(
                Document(
                    page_content=doc.content,
                    metadata=doc.metadata.model_dump(),
                )
            )
        if not lc_docs:
            return []
        return self._splitter.split_documents(lc_docs)

    def index_collection(
        self,
        collection_name: str,
        documents: list[IngestedDocument],
        *,
        reset: bool = False,
    ) -> int:
        """Index documents into a named collection. Returns chunk count."""
        if reset:
            self.reset_collection(collection_name)

        chunks = self._documents_from_ingested(documents)
        if not chunks:
            logger.warning("No chunks to index for collection %s", collection_name)
            return 0

        store = self._get_store(collection_name)
        store.add_documents(chunks)
        logger.info("Indexed %d chunks into %s", len(chunks), collection_name)
        return len(chunks)

    def reset_collection(self, collection_name: str) -> None:
        """Delete and recreate a collection."""
        try:
            store = self._get_store(collection_name)
            store.delete_collection()
        except Exception as exc:  # noqa: BLE001 - Chroma API varies by version
            logger.debug("Collection reset note (%s): %s", collection_name, exc)
        self._stores.pop(collection_name, None)

    def build_indexes(self, *, reset: bool = False) -> dict[str, int]:
        """Ingest source data and build vector collections."""
        corpora = ingest_all()
        counts = {}
        for collection_name, docs in corpora.items():
            counts[collection_name] = self.index_collection(
                collection_name,
                docs,
                reset=reset,
            )
        return counts

    def get_retriever(self, collection_name: str, k: int):
        """Return a LangChain retriever for the given collection."""
        store = self._get_store(collection_name)
        return store.as_retriever(search_kwargs={"k": k})

    def retrieve(
        self,
        collection_name: str,
        query: str,
        k: int | None = None,
    ) -> list[str]:
        """Retrieve top-k text chunks as plain strings."""
        default_k = (
            get_retrieval_config()["misconceptions_k"]
            if collection_name == MISCONCEPTIONS_COLLECTION
            else get_retrieval_config()["best_practices_k"]
        )
        k = k or default_k
        retriever = self.get_retriever(collection_name, k)
        docs = retriever.invoke(query)
        if not docs:
            logger.warning(
                "Empty retrieval for collection=%s query=%r", collection_name, query
            )
            return []
        return [doc.page_content for doc in docs]

    def collection_count(self, collection_name: str) -> int:
        """Return approximate document count in a collection."""
        try:
            store = self._get_store(collection_name)
            return store._collection.count()  # noqa: SLF001 - pragmatic introspection
        except Exception:
            return 0


def ensure_indexes(
    manager: VectorStoreManager | None = None,
    *,
    rebuild: bool = False,
) -> VectorStoreManager:
    """Ensure vector indexes exist; build them when empty or when rebuild=True."""
    mgr = manager or VectorStoreManager()
    bp_count = mgr.collection_count(BEST_PRACTICES_COLLECTION)
    misc_count = mgr.collection_count(MISCONCEPTIONS_COLLECTION)

    needs_build = rebuild or bp_count == 0 or misc_count == 0
    if (
        MISCONCEPTIONS_DIR.exists()
        and misc_count == 0
        and any(MISCONCEPTIONS_DIR.rglob("*.md"))
    ):
        needs_build = True

    if needs_build:
        logger.info("Building vector indexes (rebuild=%s)...", rebuild)
        mgr.build_indexes(reset=rebuild)
    else:
        logger.info(
            "Using existing indexes (%s=%d, %s=%d chunks)",
            BEST_PRACTICES_COLLECTION,
            bp_count,
            MISCONCEPTIONS_COLLECTION,
            misc_count,
        )
    return mgr
