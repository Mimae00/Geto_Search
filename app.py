"""Geto Search - Streamlit UI for the hybrid RAG Q&A demo.

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag import RAGPipeline
from rag import config
from rag.llm import PROVIDERS, LLMSettings, list_models
from rag.loaders import fetch_wikipedia, load_directory, load_file, save_document

st.set_page_config(page_title="Geto Search - RAG Q&A", page_icon="🔎", layout="wide")

EXAMPLES = [
    "How did Apollo 13's crew survive after the oxygen tank exploded?",
    "What instruments does the James Webb Space Telescope carry?",
    "How far away is Voyager 1 and is it still sending data?",
    "What was Ingenuity's first flight on Mars?",
    "Compare the Hubble and James Webb telescopes.",
]


@st.cache_resource(show_spinner="Loading embedding models and index (first run builds the index, ~3-5 min)...")
def get_pipeline() -> RAGPipeline:
    pipe = RAGPipeline()
    # Fresh clone / cloud deploy: build the index from the bundled corpus automatically.
    if pipe.store.count() == 0 and config.CORPUS_DIR.exists():
        docs = load_directory(config.CORPUS_DIR)
        if docs:
            pipe.ingest(docs)
    return pipe


pipe = get_pipeline()


def render_llm_settings() -> LLMSettings:
    """Optional LLM settings. Keys stay in this browser session's memory only (never written to disk)."""
    with st.expander("⚙️ LLM settings (optional)", expanded=False):
        provider = st.selectbox("Answer engine", list(PROVIDERS), index=0,
                                help="Offline works with no LLM. Ollama runs a local model, also offline. "
                                     "The others are cloud APIs and need your own key.")
        spec = PROVIDERS[provider]
        if spec is None:
            st.caption("Answers are built from the most relevant source sentences, with citations. "
                       "Fully offline.")
            return LLMSettings()

        # Widget keys are per-provider so switching providers keeps each one's values separate.
        base_url = st.text_input("Base URL", value=spec.base_url, key=f"url_{provider}")
        api_key = ""
        if provider.startswith("Custom") or spec.needs_key:
            api_key = st.text_input("API key" + ("" if spec.needs_key else " (if required)"),
                                    type="password", key=f"key_{provider}",
                                    help="Kept in memory for this browser session only. Never saved to disk.")
        if spec.key_url:
            st.caption(f"{'Install' if spec.local else 'Get a key'}: {spec.key_url}")

        # Local servers are cheap to query, so auto-detect installed models once.
        if spec.local and f"models_{provider}" not in st.session_state:
            try:
                st.session_state[f"models_{provider}"] = list_models(LLMSettings(provider, base_url.strip()))
            except Exception:
                st.session_state[f"models_{provider}"] = []
                st.warning(f"{provider} is not reachable at {base_url}. Is it running?")
        models = st.session_state.get(f"models_{provider}")
        if models:
            default = models.index(spec.default_model) if spec.default_model in models else 0
            model = st.selectbox("Model", models, index=default, key=f"model_sel_{provider}")
        else:
            model = st.text_input("Model", value=spec.default_model, key=f"model_{provider}",
                                  placeholder="model id")

        settings = LLMSettings(provider=provider, base_url=base_url.strip(), api_key=api_key.strip(),
                               model=model.strip())
        if st.button("List available models", key=f"list_{provider}", use_container_width=True):
            try:
                st.session_state[f"models_{provider}"] = list_models(settings)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not reach {provider}: {exc}")
        if spec.needs_key and not settings.api_key:
            st.info("Add an API key to enable this provider. Until then answers stay offline.")
        return settings


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔎 Geto Search")
    st.caption("Offline-first hybrid RAG over your documents. Free, local, no API key needed.")

    llm_settings = render_llm_settings()
    if llm_settings.enabled:
        spec = PROVIDERS[llm_settings.provider]
        where = "local, offline" if spec.local else "cloud"
        st.caption(f"Answers: **{llm_settings.provider}** ({where}) · `{llm_settings.model}`")
    else:
        st.caption("Answers: **offline extractive** (no LLM, nothing leaves this machine)")

    st.subheader("Retrieval")
    mode = st.radio("Search mode", ["hybrid", "dense", "keyword"], horizontal=True,
                    help="hybrid = semantic vectors + BM25 fused with Reciprocal Rank Fusion")
    top_k = st.slider("Passages to use (top-k)", 2, 10, config.TOP_K)
    rerank = st.toggle("Cross-encoder re-ranking", value=True)

    st.subheader("Knowledge base")
    sources = pipe.store.sources()
    c1, c2 = st.columns(2)
    c1.metric("Documents", len(sources))
    c2.metric("Chunks", pipe.store.count())

    with st.expander("Add documents"):
        uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
        if uploads and st.button("Index uploaded files", use_container_width=True):
            docs = []
            with tempfile.TemporaryDirectory() as tmp:
                for f in uploads:
                    path = Path(tmp) / f.name
                    path.write_bytes(f.getvalue())
                    doc = load_file(path)
                    doc.metadata["source"] = f.name
                    docs.append(doc)
            with st.spinner("Chunking and embedding..."):
                total, added = pipe.ingest(docs)
            st.success(f"Indexed {added} new chunks from {len(docs)} file(s).")
            st.rerun()

        wiki_title = st.text_input("...or a Wikipedia article title", placeholder="e.g. Neutron star")
        if wiki_title and st.button("Fetch & index", use_container_width=True):
            with st.spinner(f"Fetching '{wiki_title}'..."):
                doc = fetch_wikipedia(wiki_title)
            if doc is None:
                st.error("Article not found.")
            else:
                save_document(doc, config.CORPUS_DIR)
                _, added = pipe.ingest([doc])
                st.success(f"Indexed '{doc.metadata['title']}' ({added} chunks).")
                st.rerun()

    with st.expander(f"Indexed documents ({len(sources)})"):
        for s in sources:
            label = f"[{s['title']}]({s['source']})" if str(s["source"]).startswith("http") else s["title"]
            st.markdown(f"- {label} · {s['chunks']} chunks")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- helpers
def render_sources(chunks: list[dict], meta: dict) -> None:
    with st.expander(f"Sources ({len(chunks)}) · retrieval {meta.get('retrieve_s', 0):.2f}s · "
                     f"query: “{meta.get('search_query', '')}”"):
        for i, c in enumerate(chunks, start=1):
            m = c["metadata"]
            src = m.get("source", "")
            title = f"[{m.get('title')}]({src})" if str(src).startswith("http") else m.get("title")
            ranks = f"dense #{c['dense_rank'] or '-'} · bm25 #{c['keyword_rank'] or '-'} · score {c['score']:.3f}"
            st.markdown(f"**[{i}] {title}** — chunk {m.get('chunk')}  \n`{ranks}`")
            st.caption(c["text"][:700] + ("…" if len(c["text"]) > 700 else ""))


# ---------------------------------------------------------------- main chat
st.header("Ask the knowledge base")
if pipe.store.count() == 0:
    st.warning("The index is empty. Run `python scripts/ingest.py` or add documents from the sidebar.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for col, q in zip(cols, EXAMPLES):
        if col.button(q, use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            render_sources(msg["chunks"], msg["meta"])

question = st.chat_input("Ask a question about the indexed documents...") or st.session_state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                result = pipe.ask(question, history, top_k=top_k, mode=mode, rerank=rerank,
                                  llm_settings=llm_settings)
            answer = st.write_stream(result.answer_stream)
        except Exception as exc:  # surface API errors (bad key, rate limit) instead of crashing
            st.error(f"Something went wrong: {exc}")
            st.session_state.messages.pop()
            st.stop()

        chunks = [c.__dict__ for c in result.chunks]
        meta = {**result.timings, "search_query": result.search_query}
        render_sources(chunks, meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks, "meta": meta})
