"""Document loaders: Wikipedia (free MediaWiki API, no key) and local files."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from pypdf import PdfReader

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks API clients to send a descriptive User-Agent.
USER_AGENT = "GetoSearch-RAG-Demo/1.0 (educational portfolio project) python-requests"


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def safe_filename(title: str) -> str:
    return re.sub(r"[^\w\-]+", "_", title).strip("_")[:120]


def fetch_wikipedia(title: str, timeout: int = 30) -> Document | None:
    """Fetch the full plain-text extract of a Wikipedia article."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "redirects": 1,
        "titles": title,
    }
    # Wikimedia throttles bursts with HTTP 429; back off (honouring Retry-After) and retry.
    for attempt in range(6):
        resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code != 429:
            break
        retry_after = resp.headers.get("Retry-After", "")
        time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** attempt * 2)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract")
        if not text:
            return None
        # Drop trailing boilerplate sections that add noise to retrieval.
        text = re.split(r"\n(See also|References|External links|Notes|Further reading)\n", text)[0]
        return Document(
            text=text.strip(),
            metadata={
                "title": page.get("title", title),
                "source": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}"),
                "type": "wikipedia",
            },
        )
    return None


def save_document(doc: Document, directory: Path) -> Path:
    """Persist a document as Markdown with a small header so it can be re-indexed offline."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_filename(doc.metadata['title'])}.md"
    header = f"<!-- title: {doc.metadata['title']} -->\n<!-- source: {doc.metadata['source']} -->\n\n"
    path.write_text(header + doc.text, encoding="utf-8")
    return path


_HEADER_RE = re.compile(r"<!--\s*(\w+):\s*(.*?)\s*-->")


def load_file(path: Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    meta = {"title": path.stem.replace("_", " "), "source": path.name, "type": suffix.lstrip(".")}

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Pick up optional title/source headers written by save_document().
        for key, value in _HEADER_RE.findall(text[:500]):
            if key in {"title", "source"}:
                meta[key] = value
        text = _HEADER_RE.sub("", text).strip()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return Document(text=text, metadata=meta)


def load_directory(directory: Path) -> list[Document]:
    docs = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            doc = load_file(path)
            if doc.text.strip():
                docs.append(doc)
    return docs
