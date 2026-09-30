"""Persistent vector store (ChromaDB) + in-memory BM25 keyword index."""
from __future__ import annotations

import re

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from . import config
from .chunking import Chunk
from .embeddings import get_embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(
    "a an the and or of to in on for with by at from is are was were be been it its this that "
    "as which who what when where how why do does did has have had not but if than then so".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class VectorStore:
    def __init__(self, path=config.INDEX_DIR, collection=config.COLLECTION_NAME):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self._name = collection
        self._col = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[str] = []
        self._bm25_meta: list[dict] = []

    # ---------- write ----------
    def add(self, chunks: list[Chunk], batch_size: int = 256) -> int:
        embedder = get_embedder()
        existing = set(self._col.get(ids=[c.id for c in chunks], include=[])["ids"]) if chunks else set()
        new = [c for c in chunks if c.id not in existing]
        for i in range(0, len(new), batch_size):
            batch = new[i : i + batch_size]
            vectors = embedder.embed_passages([c.text for c in batch])
            self._col.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=vectors.tolist(),
                metadatas=[c.metadata for c in batch],
            )
        self._bm25 = None  # invalidate keyword index
        return len(new)

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})
        self._bm25 = None

    # ---------- read ----------
    def count(self) -> int:
        return self._col.count()

    def sources(self) -> list[dict]:
        """Unique documents in the index with their chunk counts."""
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        seen: dict[str, dict] = {}
        for m in metas:
            key = m.get("source", "?")
            entry = seen.setdefault(key, {"title": m.get("title", key), "source": key, "type": m.get("type"), "chunks": 0})
            entry["chunks"] += 1
        return sorted(seen.values(), key=lambda e: e["title"])

    def dense_search(self, query: str, k: int) -> list[dict]:
        if self.count() == 0:
            return []
        qvec = get_embedder().embed_query(query)
        res = self._col.query(query_embeddings=[qvec.tolist()], n_results=min(k, self.count()),
                              include=["documents", "metadatas", "distances"])
        return [
            {"id": i, "text": d, "metadata": m, "score": 1.0 - dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        data = self._col.get(include=["documents", "metadatas"])
        self._bm25_ids, self._bm25_docs, self._bm25_meta = data["ids"], data["documents"], data["metadatas"]
        self._bm25 = BM25Okapi([tokenize(d) for d in self._bm25_docs]) if self._bm25_docs else None

    def keyword_search(self, query: str, k: int) -> list[dict]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"id": self._bm25_ids[i], "text": self._bm25_docs[i], "metadata": self._bm25_meta[i], "score": float(scores[i])}
            for i in top
            if scores[i] > 0
        ]
