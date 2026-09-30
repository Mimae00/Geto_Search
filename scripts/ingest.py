"""Build the demo knowledge base.

Usage:
    python scripts/ingest.py                 # download demo corpus (if missing) + index it
    python scripts/ingest.py --reset         # wipe the index first
    python scripts/ingest.py --topics "Black hole" "Neutron star"   # add your own Wikipedia articles
    python scripts/ingest.py --no-download   # only index files already in data/corpus
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RAGPipeline  # noqa: E402
from rag.config import CORPUS_DIR  # noqa: E402
from rag.embeddings import get_reranker  # noqa: E402
from rag.loaders import fetch_wikipedia, load_directory, safe_filename, save_document  # noqa: E402

# Demo corpus: space exploration. Free, CC BY-SA licensed text from Wikipedia.
DEMO_TOPICS = [
    "Space exploration",
    "Apollo program",
    "Apollo 11",
    "Apollo 13",
    "International Space Station",
    "James Webb Space Telescope",
    "Hubble Space Telescope",
    "Voyager 1",
    "Voyager 2",
    "Perseverance (rover)",
    "Curiosity (rover)",
    "Ingenuity (helicopter)",
    "SpaceX Starship",
    "Falcon 9",
    "Artemis program",
    "Sputnik 1",
    "Yuri Gagarin",
    "Space Shuttle program",
    "Chandrayaan-3",
    "Cassini–Huygens",
    "New Horizons",
    "Mars",
    "Moon",
    "Black hole",
]


def download(topics: list[str], directory: Path) -> None:
    for title in topics:
        try:
            doc = fetch_wikipedia(title)
        except Exception as exc:  # network hiccups shouldn't abort the whole run
            print(f"  ! {title}: {exc}")
            continue
        if doc is None:
            print(f"  ! {title}: not found")
            continue
        path = save_document(doc, directory)
        print(f"  + {doc.metadata['title']:<40} {len(doc.text):>7,} chars -> {path.name}")
        time.sleep(1.0)  # be polite to the Wikimedia API


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topics", nargs="*", help="Wikipedia article titles to download")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    parser.add_argument("--no-download", action="store_true", help="skip downloading, index local files only")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        # Default: fetch any demo articles not already on disk (safe to re-run).
        topics = args.topics or [t for t in DEMO_TOPICS if not (CORPUS_DIR / f"{safe_filename(t)}.md").exists()]
        if topics:
            print(f"Downloading {len(topics)} Wikipedia articles...")
            download(topics, CORPUS_DIR)

    docs = load_directory(CORPUS_DIR)
    print(f"\nLoaded {len(docs)} documents from {CORPUS_DIR}")

    pipe = RAGPipeline()
    if args.reset:
        pipe.store.reset()
        print("Index reset.")

    t0 = time.perf_counter()
    total, added = pipe.ingest(docs)
    print(f"Chunks: {total} total, {added} newly embedded in {time.perf_counter() - t0:.1f}s")
    print(f"Index now holds {pipe.store.count()} chunks.")

    # Pre-fetch the re-ranker too, so the app works fully offline from the first question.
    get_reranker()
    print("Models ready - the app can now run offline.")


if __name__ == "__main__":
    main()
