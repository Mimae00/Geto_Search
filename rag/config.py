"""Central configuration. Tuning values can be overridden via environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # raw documents (.txt / .md / .pdf)
INDEX_DIR = DATA_DIR / "index"            # persisted Chroma DB
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", DATA_DIR / "models"))  # downloaded ONNX models

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "geto_docs")

# Embeddings (local, free, ONNX via fastembed - no GPU / torch needed)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATES = int(os.getenv("CANDIDATES", "20"))  # per retriever, before fusion

# LLM: none by default (offline). Optional providers and keys are set by the user in the app UI
# and kept in the browser session only - see rag/llm.py.
