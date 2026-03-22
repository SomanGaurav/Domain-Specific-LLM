"""Chunk the corpus, embed the chunks, and persist them to Chroma.

Usage:
    python -m finllm.rag.ingest --config configs/rag.yaml
"""

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from finllm.config import load_config


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Word-based sliding-window chunking (word count as a cheap token proxy)."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.split()) >= 32:  # drop trailing fragments
            chunks.append(chunk)
    return chunks


def ingest(config: dict) -> None:
    corpus_dir = Path(config["data"]["corpus_dir"])
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No corpus files in {corpus_dir} — run finllm.data.download_corpus first.")

    embedder = SentenceTransformer(config["embedding"]["model"], device=config["embedding"]["device"])
    client = chromadb.PersistentClient(path=config["vectorstore"]["persist_dir"])
    collection = client.get_or_create_collection(config["vectorstore"]["collection"])

    for file in tqdm(files, desc="ingesting"):
        chunks = chunk_text(
            file.read_text(),
            chunk_size=config["chunking"]["chunk_size"],
            overlap=config["chunking"]["chunk_overlap"],
        )
        embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False)
        collection.upsert(
            ids=[f"{file.stem}::{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"source": file.name, "chunk": i} for i in range(len(chunks))],
        )

    print(f"[done] collection '{collection.name}' now holds {collection.count()} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    ingest(load_config(args.config))


if __name__ == "__main__":
    main()
