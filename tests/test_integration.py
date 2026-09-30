"""End-to-end: real local models + a throwaway Chroma index. Fully offline after model download."""
import pytest

from rag.loaders import Document
from rag.pipeline import RAGPipeline
from rag.vectorstore import VectorStore

pytestmark = pytest.mark.integration

DOCS = [
    Document("The Hubble Space Telescope was launched into low Earth orbit in 1990. "
             "It observes in ultraviolet, visible and near-infrared light.",
             {"title": "Hubble", "source": "hubble.md"}),
    Document("Ingenuity was a small robotic helicopter that made the first powered flight on Mars "
             "on 19 April 2021. It was carried to Mars by the Perseverance rover.",
             {"title": "Ingenuity", "source": "ingenuity.md"}),
    Document("Yuri Gagarin became the first human in space aboard Vostok 1 on 12 April 1961.",
             {"title": "Gagarin", "source": "gagarin.md"}),
]


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    p = RAGPipeline(VectorStore(path=tmp_path_factory.mktemp("index"), collection="test"))
    total, added = p.ingest(DOCS, chunk_size=400, overlap=50)
    assert total == added == 3
    return p


def test_reingest_is_idempotent(pipe):
    assert pipe.ingest(DOCS, chunk_size=400, overlap=50) == (3, 0)
    assert pipe.store.count() == 3


@pytest.mark.parametrize("mode", ["hybrid", "dense", "keyword"])
def test_retrieves_the_right_document(pipe, mode):
    res = pipe.ask("Who was the first person to fly in space?", mode=mode, top_k=1, rerank=False)
    assert res.chunks[0].metadata["title"] == "Gagarin"


def test_offline_answer_with_rerank_and_citation(pipe):
    res = pipe.ask("When did the helicopter first fly on Mars?", top_k=2, rerank=True)
    assert res.chunks[0].metadata["title"] == "Ingenuity"
    assert res.timings["llm"] == "offline extractive"
    answer = "".join(res.answer_stream)
    assert "19 April 2021" in answer and "[1]" in answer
