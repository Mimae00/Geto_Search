from rag.chunking import chunk_documents, chunk_text
from rag.loaders import Document


def test_short_text_is_single_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    # overlap tail can push a chunk slightly past size, but never by more than the overlap
    assert all(len(c) <= 300 + 50 for c in chunks)


def test_no_content_is_lost():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(80)]
    chunks = chunk_text(" ".join(sentences), size=200, overlap=0)
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_overlap_carries_context_forward():
    text = " ".join(f"token{i}" for i in range(400))
    chunks = chunk_text(text, size=200, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:]):
        first_word = nxt.split()[0]
        assert first_word in prev  # next chunk starts inside the previous one's tail


def test_very_long_word_is_hard_split():
    chunks = chunk_text("x" * 1000, size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks) == "x" * 1000


def test_chunk_documents_ids_and_metadata():
    docs = [Document("Alpha. " * 200, {"title": "A", "source": "a.md"}),
            Document("Beta. " * 200, {"title": "B", "source": "b.md"})]
    chunks = chunk_documents(docs, size=200, overlap=20)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert {c.metadata["title"] for c in chunks} == {"A", "B"}
    assert [c.metadata["chunk"] for c in chunks if c.metadata["title"] == "A"][:3] == [0, 1, 2]
    # deterministic ids -> re-ingesting the same docs is a no-op
    assert ids == [c.id for c in chunk_documents(docs, size=200, overlap=20)]
