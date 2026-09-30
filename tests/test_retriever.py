import pytest

from rag.retriever import HybridRetriever, reciprocal_rank_fusion
from rag.vectorstore import tokenize


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused["a"] == pytest.approx(fused["b"])  # 1st+2nd in both lists
    assert fused["a"] > fused["c"] and fused["a"] > fused["d"]


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == {}


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the Hubble Space-Telescope?") == ["hubble", "space", "telescope"]


class FakeStore:
    """Stands in for VectorStore so retrieval logic is tested without models or a DB."""

    def __init__(self, dense, keyword):
        self._dense, self._keyword = dense, keyword

    @staticmethod
    def _hits(ids):
        return [{"id": i, "text": f"text {i}", "metadata": {"title": i}, "score": 1.0} for i in ids]

    def dense_search(self, query, k):
        return self._hits(self._dense)[:k]

    def keyword_search(self, query, k):
        return self._hits(self._keyword)[:k]


def test_hybrid_merges_both_retrievers():
    r = HybridRetriever(FakeStore(dense=["a", "b", "c"], keyword=["c", "d"]))
    out = r.retrieve("q", top_k=4, mode="hybrid", rerank=False)
    assert [c.id for c in out][0] == "c"  # only doc ranked by both lists
    assert {c.id for c in out} == {"a", "b", "c", "d"}
    c = next(x for x in out if x.id == "c")
    assert (c.dense_rank, c.keyword_rank) == (3, 1)


@pytest.mark.parametrize("mode,expected", [("dense", ["a", "b"]), ("keyword", ["x", "y"])])
def test_single_mode_uses_one_retriever(mode, expected):
    r = HybridRetriever(FakeStore(dense=["a", "b"], keyword=["x", "y"]))
    assert [c.id for c in r.retrieve("q", top_k=5, mode=mode, rerank=False)] == expected


def test_top_k_is_respected():
    r = HybridRetriever(FakeStore(dense=list("abcdefgh"), keyword=list("hgfedcba")))
    assert len(r.retrieve("q", top_k=3, rerank=False)) == 3
