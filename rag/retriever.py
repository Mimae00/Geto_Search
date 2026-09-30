"""Hybrid retrieval: dense (semantic) + BM25 (keyword), fused with Reciprocal Rank Fusion,
then optionally re-ranked with a cross-encoder."""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .embeddings import get_reranker
from .vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float           # final score used for ordering
    dense_rank: int | None
    keyword_rank: int | None


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF: score(d) = sum over lists of 1 / (k + rank). Robust to incomparable score scales."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = config.TOP_K, mode: str = "hybrid",
                 rerank: bool = True, candidates: int = config.CANDIDATES) -> list[RetrievedChunk]:
        dense = self.store.dense_search(query, candidates) if mode in {"hybrid", "dense"} else []
        keyword = self.store.keyword_search(query, candidates) if mode in {"hybrid", "keyword"} else []

        pool = {h["id"]: h for h in keyword}
        pool.update({h["id"]: h for h in dense})
        dense_rank = {h["id"]: r for r, h in enumerate(dense, start=1)}
        kw_rank = {h["id"]: r for r, h in enumerate(keyword, start=1)}

        fused = reciprocal_rank_fusion([[h["id"] for h in dense], [h["id"] for h in keyword]])
        ordered = sorted(fused, key=fused.get, reverse=True)

        results = [
            RetrievedChunk(id=i, text=pool[i]["text"], metadata=pool[i]["metadata"], score=fused[i],
                           dense_rank=dense_rank.get(i), keyword_rank=kw_rank.get(i))
            for i in ordered
        ]

        if rerank and results:
            shortlist = results[: max(top_k * 3, 10)]
            scores = get_reranker().score(query, [r.text for r in shortlist])
            for r, s in zip(shortlist, scores):
                r.score = s
            results = sorted(shortlist, key=lambda r: r.score, reverse=True)

        return results[:top_k]
