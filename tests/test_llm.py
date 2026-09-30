import json

import pytest

from rag import llm
from rag.llm import OFFLINE, ChatLLM, ExtractiveAnswerer, LLMSettings, get_llm, parse_sse, split_sentences
from rag.retriever import RetrievedChunk


def chunk(text, title="T"):
    return RetrievedChunk(id=title, text=text, metadata={"title": title}, score=1.0, dense_rank=1, keyword_rank=1)


# ---------------------------------------------------------------- settings / selection
def test_offline_is_the_default():
    assert isinstance(get_llm(), ExtractiveAnswerer)
    assert isinstance(get_llm(LLMSettings()), ExtractiveAnswerer)
    assert LLMSettings().provider == OFFLINE


def test_cloud_provider_needs_a_key():
    s = LLMSettings(provider="Groq", base_url="https://api.groq.com/openai/v1", model="m")
    assert not s.enabled and isinstance(get_llm(s), ExtractiveAnswerer)
    s.api_key = "k"
    assert s.enabled and isinstance(get_llm(s), ChatLLM)


def test_ollama_works_without_key():
    assert LLMSettings(provider="Ollama (local)", base_url="http://localhost:11434/v1", model="llama3.2:3b").enabled


def test_missing_model_disables_llm():
    assert not LLMSettings(provider="Ollama (local)", base_url="http://localhost:11434/v1", model="").enabled


def test_no_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "should-be-ignored")
    monkeypatch.setenv("OPENAI_API_KEY", "should-be-ignored")
    assert LLMSettings().api_key == ""
    assert isinstance(get_llm(), ExtractiveAnswerer)


# ---------------------------------------------------------------- streaming protocol
def sse(*deltas):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}".encode() for d in deltas]
    return [b": keep-alive", b""] + lines + [b"data: [DONE]", b"data: ignored-after-done"]


def test_parse_sse():
    assert "".join(parse_sse(sse("Hel", "lo", " [1]"))) == "Hello [1]"


class FakeResp:
    def __init__(self, status=200, lines=(), payload=None):
        self.status_code, self._lines, self._payload, self.text = status, lines, payload, ""

    def iter_lines(self):
        return iter(self._lines)

    def json(self):
        return self._payload


SETTINGS = LLMSettings(provider="Groq", base_url="https://example.test/v1/", api_key="secret", model="m")


def test_chat_llm_streams_and_sends_auth(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, stream, timeout):
        seen.update(url=url, headers=headers, body=json)
        return FakeResp(lines=sse("Answer ", "[1]"))

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = "".join(ChatLLM(SETTINGS).stream("q?", [chunk("ctx text")]))
    assert out == "Answer [1]"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert "[1] (from: T)\nctx text" in seen["body"]["messages"][-1]["content"]


def test_chat_llm_falls_back_offline_on_error(monkeypatch):
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: FakeResp(401, payload={"error": {"message": "Invalid API Key"}}))
    monkeypatch.setattr(ExtractiveAnswerer, "stream", lambda self, *a, **k: iter(["- offline [1]\n"]))
    out = "".join(ChatLLM(SETTINGS).stream("q?", [chunk("ctx")]))
    assert "Invalid API Key" in out and "- offline [1]" in out


def test_rewrite_query(monkeypatch):
    payload = {"choices": [{"message": {"content": '"Apollo 13 budget"'}}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(payload=payload))
    history = [{"role": "user", "content": "Tell me about Apollo 13"}]
    assert ChatLLM(SETTINGS).rewrite_query("what about its budget?", history) == "Apollo 13 budget"
    assert ChatLLM(SETTINGS).rewrite_query("standalone", []) == "standalone"


def test_rewrite_query_survives_errors(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(llm.requests, "post", boom)
    assert ChatLLM(SETTINGS).rewrite_query("q", [{"role": "user", "content": "x"}]) == "q"


def test_list_models(monkeypatch):
    payload = {"data": [{"id": "models/gemini-x"}, {"id": "b-model"}, {"id": ""}]}
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(payload=payload))
    assert llm.list_models(SETTINGS) == ["b-model", "gemini-x"]


def test_list_models_error(monkeypatch):
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(403, payload={"error": "nope"}))
    with pytest.raises(RuntimeError, match="403"):
        llm.list_models(SETTINGS)


# ---------------------------------------------------------------- extractive helpers
def test_split_sentences_skips_headings_and_fragments():
    text = "Mission overview\nApollo 11 landed on the Moon on July 20, 1969. Too short.\nArmstrong stepped out first onto the lunar surface."
    sentences = [s for _, s in split_sentences([chunk(text)])]
    assert sentences == ["Apollo 11 landed on the Moon on July 20, 1969.",
                         "Armstrong stepped out first onto the lunar surface."]
