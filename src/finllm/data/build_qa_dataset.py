"""Build the finance QA dataset from the downloaded corpus.

Strategy (to implement):
  1. Chunk each filing into passages (reuse finllm.rag.ingest.chunk_text so the
     QA dataset and the RAG index see identical passages).
  2. Generate question-answer pairs per passage. Options, in order of quality:
       a. LLM-generated QA (e.g. via a hosted API or a larger local model) with
          the passage as grounding — answers must be extractable/verifiable.
       b. Seed from public finance QA sets (FiQA, FinQA) filtered to topics the
          corpus covers.
  3. Deduplicate, filter degenerate pairs, write JSONL:
       {"question": ..., "answer": ..., "source_file": ..., "passage": ...}

Usage:
    python -m finllm.data.build_qa_dataset --corpus-dir data/raw/sec_filings
"""

import argparse
from pathlib import Path


def generate_qa_pairs(passage: str, source_file: str) -> list[dict]:
    """Generate grounded QA pairs for one passage. TODO: implement (see module docstring)."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw/sec_filings"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/qa_dataset.jsonl"))
    args = parser.parse_args()
    raise NotImplementedError("QA generation not implemented yet — see module docstring.")


if __name__ == "__main__":
    main()
