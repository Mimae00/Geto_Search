"""High-level RAG pipeline tying ingestion, retrieval and generation together."""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import config
from .chunking import chunk_documents
from .llm import LLMSettings, get_llm
from .loaders import Document
from .retriever import HybridRetriever, RetrievedChunk
from .vectorstore import VectorStore


@dataclass
class RAGResult:
    question: str
    search_query: str
    chunks: list[RetrievedChunk]
    answer_stream: Iterator[str]
    timings: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self.retriever = HybridRetriever(self.store)

    def ingest(self, docs: list[Document], chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> tuple[int, int]:
        """Chunk + embed + index documents. Returns (chunks_created, chunks_added)."""
        chunks = chunk_documents(docs, chunk_size, overlap)
        added = self.store.add(chunks)
        return len(chunks), added

    def ask(self, question: str, history: list[dict] | None = None, *, top_k: int = config.TOP_K,
            mode: str = "hybrid", rerank: bool = True, llm_settings: LLMSettings | None = None) -> RAGResult:
        """Answer a question. With no (or incomplete) llm_settings this runs fully offline."""
        llm = get_llm(llm_settings)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        search_query = llm.rewrite_query(question, history or [])
        timings["rewrite_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(search_query, top_k=top_k, mode=mode, rerank=rerank)
        timings["retrieve_s"] = time.perf_counter() - t0
        timings["llm"] = llm.name

        return RAGResult(
            question=question,
            search_query=search_query,
            chunks=chunks,
            answer_stream=llm.stream(question, chunks, history),
            timings=timings,
        )
