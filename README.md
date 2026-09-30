# 🔎 Geto Search: Offline-first Hybrid RAG Q&A

[![CI](https://github.com/<your-username>/geto-search/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/geto-search/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)

A retrieval-augmented question-answering system that runs **entirely on your machine**: no API key, no cloud, no GPU. Ask natural-language questions over a document collection and get answers with inline citations `[1]` that link back to the source passages.

An LLM is **optional**. You can plug in a local model through Ollama, which keeps everything offline, or bring your own key for any OpenAI-compatible provider.

The demo ships with a **24-article space-exploration knowledge base** (~1.3M characters, ~2,000 chunks) pulled from the free Wikipedia API. You can upload your own PDF, TXT and Markdown files, or pull in any Wikipedia article from the UI.

## Features

- **Offline by default**: answers are assembled from the most relevant source sentences, ranked by the local embedding model, with citations
- **Hybrid retrieval**: dense semantic search (BGE embeddings in ChromaDB) plus BM25 keyword search, merged with **Reciprocal Rank Fusion**
- **Cross-encoder re-ranking** (MiniLM) for higher-precision top-k
- **Optional LLM, your choice**: Ollama (local), Groq, OpenAI, Google Gemini, OpenRouter, or any custom OpenAI-compatible URL (LM Studio, vLLM, llama.cpp...). Local models are auto-detected, and "List available models" asks any provider what it offers.
- **Conversational follow-ups** (with an LLM): "who commanded that mission?" is rewritten into a standalone search query
- **Fails safe**: if the LLM is unreachable or the key is wrong, the app shows the error and falls back to the offline answer
- **Transparent retrieval**: every answer shows its sources, dense and BM25 ranks, re-rank scores and latency. You can switch between hybrid, dense and keyword modes live.

## Privacy: how API keys are handled

- The app **never reads keys from environment variables or `.env` files**, and never writes them to disk.
- Keys are typed into the sidebar and live only in that browser session's memory. They're gone when you close the tab.
- In offline mode (and with Ollama), no question or document ever leaves your machine.

## Architecture

```
Ingestion
  Wikipedia API / PDF / TXT / MD
    → loaders.py → chunking.py (recursive, overlapping) → embeddings.py (bge-small, ONNX, CPU)
    → vectorstore.py: ChromaDB (cosine) + BM25 index

Query
  question → (optional LLM query rewrite for follow-ups)
    → dense top-20  ┐
    → BM25 top-20   ┴→ RRF fusion → cross-encoder rerank → top-k passages
    → llm.py:  offline extractive answer   (default)
               or any OpenAI-compatible LLM (Ollama / Groq / OpenAI / Gemini / OpenRouter / custom)
    → Streamlit chat UI with source inspector
```

| Layer | Tool | Runs |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, no PyTorch) | local |
| Re-ranker | `ms-marco-MiniLM-L-6-v2` cross-encoder via fastembed | local |
| Vector DB | ChromaDB (embedded, persistent) | local |
| Keyword search | rank-bm25 | local |
| Answering | Extractive (default) · Ollama · any OpenAI-compatible API | local / optional cloud |
| Data | Wikipedia MediaWiki API (TextExtracts, no key) | one-time download |
| UI | Streamlit | local |

## Run it locally

> **Live demo:** coming soon. In the meantime, the app runs on any laptop in a few minutes.

### Requirements

- Python **3.12 or 3.13** ([download](https://www.python.org/downloads/))
- About 2 GB of free RAM and 1.5 GB of disk space
- Internet access **for the first setup only** (Python packages + ~150 MB of models). Everything after that works offline.
- No GPU, no API key, no Docker needed

### 1. One-time setup

**Windows (PowerShell)**

```powershell
git clone https://github.com/<your-username>/geto-search.git
cd geto-search
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/ingest.py --no-download   # downloads the models and builds the search index (~3-5 min)
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**macOS / Linux**

```bash
git clone https://github.com/<your-username>/geto-search.git
cd geto-search
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py --no-download
```

The demo documents are already in the repo (`data/corpus/`), so `--no-download` skips the Wikipedia download and only builds the index. You can skip this step, and the app will build the index on its first launch instead. Running it ahead of time means the app opens instantly during a demo.

### 2. Start the app

```bash
# activate the venv first (see above), then:
streamlit run app.py
```

It opens at <http://localhost:8501>. Stop it with `Ctrl+C`.

### 3. Demo walkthrough (about 5 minutes)

1. **Offline answer:** click an example question, e.g. *"How did Apollo 13's crew survive after the oxygen tank exploded?"* The answer comes back with `[1]`-style citations and no LLM.
2. **Show the retrieval:** expand **Sources** under the answer to show the dense rank, the BM25 rank, the re-rank score and the latency for each passage.
3. **Compare strategies:** in the sidebar, switch **Search mode** between `hybrid`, `dense` and `keyword`, and turn re-ranking off and on, then ask the same question again.
4. **Bring your own data:** under **Add documents**, upload a PDF, or type a Wikipedia title such as `Neutron star` (this needs internet), then ask about it.
5. **Optional LLM:** open **⚙️ LLM settings**, pick **Ollama (local)** (see below), and ask a follow-up like *"Who commanded that mission?"* to show conversational query rewriting.
6. **Prove it's offline:** turn off Wi-Fi and ask another question. Offline mode and Ollama both keep working.

### Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found, or the wrong version | Windows: `py -3.13 -m venv .venv`. macOS/Linux: `python3.13 -m venv .venv` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| "The index is empty" warning | `python scripts/ingest.py --no-download` |
| Weird results after changing documents | `python scripts/ingest.py --reset --no-download` rebuilds the index from scratch |
| Ollama "not reachable" | Start the Ollama app, or run `ollama serve`, and check `ollama list` shows a model |
| First question is slow | Normal: the models load on the first query (a few seconds). Later queries take about 1 second. |

### Optional: local LLM with Ollama (still offline)

```powershell
# install from https://ollama.com/download, then:
ollama pull llama3.2:3b     # or qwen2.5:3b, llama3.2:1b for low-RAM machines
```

In the app, open **⚙️ LLM settings**, choose **Ollama (local)**, and your installed models appear automatically.

### Optional: cloud LLM

Choose a provider in **⚙️ LLM settings** and paste your own API key. Groq has a free tier with no credit card: <https://console.groq.com/keys>.

## Managing the knowledge base

```powershell
python scripts/ingest.py                                  # download missing demo articles + index
python scripts/ingest.py --topics "Neutron star" "Kuiper belt"
python scripts/ingest.py --no-download                    # index files you dropped into data/corpus/
python scripts/ingest.py --reset                          # rebuild the index from scratch
```

Tuning via environment variables (optional): `EMBED_MODEL`, `CHUNK_SIZE` (900), `CHUNK_OVERLAP` (150), `TOP_K` (5), `CANDIDATES` (20).

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -m "not integration"   # fast unit tests: no models, no network
pytest                        # everything, including end-to-end tests with the real local models
```

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the unit tests on Python 3.12 and 3.13. It then runs the end-to-end suite and a headless Streamlit smoke test that asks a question in offline mode and checks for a cited answer.

## Project layout

```
app.py                 Streamlit UI (chat, sources, LLM settings, uploads)
scripts/ingest.py      Corpus download + indexing CLI
rag/
  config.py            Settings (env-overridable tuning only)
  loaders.py           Wikipedia API (with 429 back-off) + PDF/TXT/MD loaders
  chunking.py          Recursive paragraph/sentence chunker with overlap
  embeddings.py        fastembed embedder + cross-encoder reranker
  vectorstore.py       ChromaDB + BM25
  retriever.py         Hybrid retrieval, RRF, reranking
  llm.py               Offline extractive answerer + OpenAI-compatible chat client
  pipeline.py          End-to-end RAGPipeline
tests/                 pytest suite (unit + integration)
data/corpus/           Demo documents (Wikipedia, CC BY-SA 4.0)
```

## Example questions

- How did Apollo 13's crew survive after the oxygen tank exploded?
- What was Ingenuity's first flight on Mars?
- Compare the Hubble and James Webb telescopes.
- Who was the first human in space, and what spacecraft did he fly?

## License

Code: [MIT](LICENSE). Demo corpus text comes from [Wikipedia](https://en.wikipedia.org/) under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each file keeps its source URL.
