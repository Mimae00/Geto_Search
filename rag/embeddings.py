"""Local embedding + reranking models via fastembed (ONNX runtime, CPU, free)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from . import config

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(config.MODEL_CACHE_DIR))

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.array(list(self._model.passage_embed(texts, batch_size=batch_size)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)


class Reranker:
    """Cross-encoder that scores (query, passage) pairs jointly - slower but more precise."""

    def __init__(self, model_name: str = RERANK_MODEL):
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(config.MODEL_CACHE_DIR))

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, passages)]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
