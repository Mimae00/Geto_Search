import pytest

from rag import loaders
from rag.loaders import Document, load_directory, load_file, safe_filename, save_document


def test_save_and_reload_roundtrip(tmp_path):
    doc = Document("Body text here.", {"title": "Apollo 11", "source": "https://en.wikipedia.org/wiki/Apollo_11"})
    path = save_document(doc, tmp_path)
    loaded = load_file(path)
    assert loaded.text == "Body text here."
    assert loaded.metadata["title"] == "Apollo 11"
    assert loaded.metadata["source"].startswith("https://")


def test_plain_txt_uses_filename_as_title(tmp_path):
    (tmp_path / "my_notes.txt").write_text("Some notes.", encoding="utf-8")
    doc = load_file(tmp_path / "my_notes.txt")
    assert doc.metadata == {"title": "my notes", "source": "my_notes.txt", "type": "txt"}


def test_unsupported_type_raises(tmp_path):
    (tmp_path / "x.docx").write_text("?", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(tmp_path / "x.docx")


def test_load_directory_skips_empty_and_unknown(tmp_path):
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    assert [d.metadata["source"] for d in load_directory(tmp_path)] == ["a.md"]


def test_safe_filename():
    assert safe_filename("Perseverance (rover)") == "Perseverance_rover"


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


WIKI_PAYLOAD = {"query": {"pages": {"1": {
    "title": "Moon", "fullurl": "https://en.wikipedia.org/wiki/Moon",
    "extract": "The Moon is Earth's satellite.\nSee also\nStuff to drop",
}}}}


def test_fetch_wikipedia_parses_and_strips_boilerplate(monkeypatch):
    monkeypatch.setattr(loaders.requests, "get", lambda *a, **k: FakeResp(200, WIKI_PAYLOAD))
    doc = loaders.fetch_wikipedia("Moon")
    assert doc.text == "The Moon is Earth's satellite."
    assert doc.metadata["source"] == "https://en.wikipedia.org/wiki/Moon"


def test_fetch_wikipedia_retries_on_429(monkeypatch):
    responses = iter([FakeResp(429, headers={"Retry-After": "1"}), FakeResp(200, WIKI_PAYLOAD)])
    sleeps = []
    monkeypatch.setattr(loaders.requests, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(loaders.time, "sleep", sleeps.append)
    assert loaders.fetch_wikipedia("Moon") is not None
    assert sleeps == [1.0]


def test_fetch_wikipedia_missing_page(monkeypatch):
    payload = {"query": {"pages": {"-1": {"title": "Nope", "missing": ""}}}}
    monkeypatch.setattr(loaders.requests, "get", lambda *a, **k: FakeResp(200, payload))
    assert loaders.fetch_wikipedia("Nope") is None
