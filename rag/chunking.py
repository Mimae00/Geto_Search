"""Recursive, paragraph-aware text chunking with overlap."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .loaders import Document

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def _split(text: str, size: int, separators: list[str]) -> list[tuple[str, str]]:
    """Split text into (piece, joiner) pairs no longer than `size`, preferring coarse separators.

    `joiner` is the whitespace used when gluing the piece back onto the previous one.
    """
    if len(text) <= size:
        return [(text, "\n")]
    if not separators:
        return [(text[i : i + size], "") for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if sep == ". ":  # keep sentence punctuation
        parts = [p + "." for p in parts[:-1]] + parts[-1:]
    joiner = "\n" if "\n" in sep else " "

    pieces: list[tuple[str, str]] = []
    for part in parts:
        if len(part) > size:
            pieces.extend(_split(part, size, rest))
        elif part.strip():
            pieces.append((part, joiner))
    return pieces


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks: list[str] = []
    current = ""
    for piece, joiner in _split(text, size, _SEPARATORS):
        piece = piece.strip()
        if current and len(current) + len(piece) + 1 > size:
            chunks.append(current)
            # Carry an overlap tail (cut at a word boundary) into the next chunk.
            tail = current[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail} {piece}".strip() if tail else piece
        else:
            current = f"{current}{joiner}{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        for i, text in enumerate(chunk_text(doc.text, size, overlap)):
            digest = hashlib.sha1(f"{doc.metadata.get('source')}|{i}|{text}".encode()).hexdigest()[:16]
            out.append(Chunk(id=digest, text=text, metadata={**doc.metadata, "chunk": i}))
    return out
