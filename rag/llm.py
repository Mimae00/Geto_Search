"""Answer generation - offline first.

- ExtractiveAnswerer (default): 100% offline. Picks the most relevant sentences from the
  retrieved passages using the local embedding model, with citations. No LLM, no key.
- ChatLLM (optional): any OpenAI-compatible chat endpoint the *user* configures in the UI:
  Ollama (local, still offline), Groq, OpenAI, Google Gemini, OpenRouter, or a custom URL.

No API key is read from the environment or stored on disk; keys live only in the
user's browser session.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import requests

from .embeddings import get_embedder
from .retriever import RetrievedChunk

SYSTEM_PROMPT = """You are Geto Search, a precise question-answering assistant.
Answer the user's question using ONLY the numbered context passages provided.
Rules:
- Cite passages inline with their number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, say you could not find it in the indexed documents.
- Be concise: a short paragraph or a few bullet points. Do not invent facts."""

OFFLINE = "Offline (no LLM)"


@dataclass(frozen=True)
class Provider:
    base_url: str
    default_model: str
    needs_key: bool = True
    local: bool = False
    key_url: str = ""


# Default model names are suggestions; they change often, so the UI lets users edit them
# or fetch the live list from the provider's /models endpoint.
PROVIDERS: dict[str, Provider | None] = {
    OFFLINE: None,
    "Ollama (local)": Provider("http://localhost:11434/v1", "llama3.2:3b", needs_key=False, local=True,
                               key_url="https://ollama.com/download"),
    "Groq": Provider("https://api.groq.com/openai/v1", "llama-3.1-8b-instant",
                     key_url="https://console.groq.com/keys"),
    "OpenAI": Provider("https://api.openai.com/v1", "gpt-5.4-mini",
                       key_url="https://platform.openai.com/api-keys"),
    "Google Gemini": Provider("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.6-flash",
                              key_url="https://aistudio.google.com/apikey"),
    "OpenRouter": Provider("https://openrouter.ai/api/v1", "openrouter/auto",
                           key_url="https://openrouter.ai/keys"),
    "Custom (OpenAI-compatible)": Provider("http://localhost:1234/v1", "", needs_key=False),
}


@dataclass
class LLMSettings:
    provider: str = OFFLINE
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    @property
    def enabled(self) -> bool:
        spec = PROVIDERS.get(self.provider)
        if spec is None or not self.base_url or not self.model:
            return False
        return bool(self.api_key) or not spec.needs_key


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (from: {c.metadata.get('title', 'unknown')})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: list[RetrievedChunk], history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:  # short memory for follow-up questions
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {question}"})
    return messages


def parse_sse(lines: Iterator[bytes | str]) -> Iterator[str]:
    """Yield content deltas from an OpenAI-style server-sent-events stream."""
    for raw in lines:
        line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            choices = json.loads(data).get("choices") or []
        except json.JSONDecodeError:
            continue
        if choices:
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta


def _error_detail(resp: requests.Response) -> str:
    try:
        err = resp.json().get("error", resp.text)
        return err.get("message", str(err)) if isinstance(err, dict) else str(err)
    except ValueError:
        return resp.text[:200]


class ChatLLM:
    """Minimal OpenAI-compatible chat client (plain HTTP, no vendor SDK)."""

    def __init__(self, settings: LLMSettings, timeout: int = 120):
        self.settings = settings
        self.name = f"{settings.provider} · {settings.model}"
        self.timeout = timeout
        self._url = settings.base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if settings.api_key:
            self._headers["Authorization"] = f"Bearer {settings.api_key}"

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        resp = requests.post(f"{self._url}/chat/completions", headers=self._headers, json=payload,
                             stream=stream, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {_error_detail(resp)}")
        return resp

    def stream(self, question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None,
               temperature: float = 0.1) -> Iterator[str]:
        payload = {"model": self.settings.model, "messages": build_messages(question, chunks, history),
                   "temperature": temperature, "max_tokens": 700, "stream": True}
        try:
            resp = self._post(payload, stream=True)
            yield from parse_sse(resp.iter_lines())
        except Exception as exc:  # bad key, rate limit, Ollama not running... keep the demo usable
            yield f"_LLM request failed ({exc}). Showing offline extractive answer instead._\n\n"
            yield from ExtractiveAnswerer().stream(question, chunks, show_banner=False)

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up ("what about its budget?") into a standalone search query."""
        if not history:
            return question
        convo = "\n".join(f"{t['role']}: {t['content'][:400]}" for t in history[-4:])
        payload = {
            "model": self.settings.model, "temperature": 0, "max_tokens": 60, "stream": False,
            "messages": [
                {"role": "system", "content": "Rewrite the last user question as a standalone search query. "
                                              "Return only the query, nothing else."},
                {"role": "user", "content": f"Conversation:\n{convo}\n\nLast question: {question}"},
            ],
        }
        try:
            content = self._post(payload, stream=False).json()["choices"][0]["message"]["content"]
        except Exception:
            return question
        return (content or question).strip().strip('"') or question


def list_models(settings: LLMSettings, timeout: int = 15) -> list[str]:
    """Ask the provider which models are available (GET /models). Raises on failure."""
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    resp = requests.get(f"{settings.base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {_error_detail(resp)}")
    ids = [m.get("id", "") for m in resp.json().get("data", [])]
    return sorted(i.removeprefix("models/") for i in ids if i)


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(chunks: list[RetrievedChunk]) -> list[tuple[int, str]]:
    """(passage_number, sentence) candidates, skipping headings and fragments."""
    out: list[tuple[int, str]] = []
    for idx, c in enumerate(chunks, start=1):
        for line in c.text.split("\n"):  # per line so headings don't glue onto sentences
            for s in _SENT_RE.split(line):
                s = s.strip()
                if 40 <= len(s) <= 500 and s[-1] in ".!?\"')":
                    out.append((idx, s))
    return out


class ExtractiveAnswerer:
    name = "offline extractive"

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def stream(self, question: str, chunks: list[RetrievedChunk], history=None, temperature=0.0,
               show_banner: bool = True) -> Iterator[str]:
        candidates = split_sentences(chunks)
        if not candidates:
            yield "I could not find relevant information in the indexed documents."
            return

        emb = get_embedder()
        q = emb.embed_query(question)
        vecs = emb.embed_passages([s for _, s in candidates])
        sims = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
        best = sorted(np.argsort(-sims)[: self.max_sentences].tolist())  # keep source order for readability

        if show_banner:
            yield "_Offline mode - most relevant sentences from the sources:_\n\n"
        for i in best:
            idx, sentence = candidates[i]
            yield f"- {sentence} [{idx}]\n"

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        return question


def get_llm(settings: LLMSettings | None = None):
    if settings is not None and settings.enabled:
        return ChatLLM(settings)
    return ExtractiveAnswerer()
