"""Deterministic train/val/test split of the QA dataset.

Usage:
    python -m finllm.data.make_splits --config configs/base.yaml
"""

import argparse
import json
import random
from pathlib import Path

from finllm.config import load_config


def make_splits(qa_path: Path, out_dir: Path, ratios: dict[str, float], seed: int) -> None:
    rows = [json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)

    n = len(rows)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in split_rows) + "\n")
        print(f"[ok] {path} ({len(split_rows)} examples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    make_splits(
        qa_path=Path(config["data"]["qa_dataset"]),
        out_dir=Path(config["data"]["processed_dir"]),
        ratios=config["data"]["splits"],
        seed=config["data"]["seed"],
    )


if __name__ == "__main__":
    main()
