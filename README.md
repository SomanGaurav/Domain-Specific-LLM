# finllm — PEFT vs RAG for Finance-Domain QA

Does parameter-efficient fine-tuning or retrieval-augmented generation adapt a
small open LLM to a specialized domain better, under realistic resource
constraints? This project compares LoRA, QLoRA, and DoRA against a RAG pipeline
built on the **same base model and the same corpus** (SEC 10-K filings), with
every experiment tracked in MLflow.

## Experiment matrix

| Arm | Adaptation | Runs on |
|---|---|---|
| `base` | none (zero-shot baseline) | local GPU |
| `lora` | fp16 LoRA | Colab/Kaggle T4 (fp16 1.5B won't fit 4 GB) |
| `qlora` | 4-bit NF4 base + LoRA | local GPU |
| `dora` | 4-bit NF4 base + DoRA (only `use_dora` differs from qlora) | local GPU |
| `rag` | Chroma + bge-small retrieval, base model generator | local GPU |

All arms are scored on the same held-out test set: Exact Match, token F1,
ROUGE-L, BERTScore, latency (mean/p50/p95), and peak GPU memory — logged under
one MLflow experiment so runs are directly comparable.

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
**Hardware target:** RTX 3050 Ti Laptop (4 GB VRAM), 7 GB RAM.

## Setup

```bash
uv sync            # installs into .venv (torch+CUDA is a multi-GB download)
source .venv/bin/activate.fish
```

## Workflow

```bash
# 1. Corpus: download 10-K filings from SEC EDGAR
python -m finllm.data.download_corpus --tickers AAPL MSFT JPM GS WMT XOM --num-filings 2

# 2. QA dataset (TODO: generation strategy in build_qa_dataset.py docstring)
python -m finllm.data.build_qa_dataset
python -m finllm.data.make_splits --config configs/base.yaml

# 3. Fine-tune (one script, method chosen by config)
python -m finllm.training.train_peft --config configs/qlora.yaml
python -m finllm.training.train_peft --config configs/dora.yaml

# 4. RAG index
python -m finllm.rag.ingest --config configs/rag.yaml

# 5. Evaluate every arm on the shared test set
python -m finllm.eval.run_eval --config configs/base.yaml  --system base
python -m finllm.eval.run_eval --config configs/qlora.yaml --system peft --adapter checkpoints/qlora/final_adapter
python -m finllm.eval.run_eval --config configs/rag.yaml   --system rag

# Inspect results
mlflow ui
```

## Layout

```
configs/           YAML configs with inheritance (base.yaml + per-method overrides)
src/finllm/
  data/            EDGAR downloader, QA dataset builder, splits
  training/        config-driven LoRA/QLoRA/DoRA fine-tuning (TRL SFTTrainer)
  rag/             chunk+embed ingestion (Chroma), retrieval+generation pipeline
  eval/            metrics, latency/VRAM profiling, unified eval runner
  config.py        YAML loader with `inherits:` support
  tracking.py      MLflow helpers
tests/             smoke tests (config system, pure metrics)
```

## Status

- [x] Scaffold: configs, training/RAG/eval pipelines, MLflow wiring
- [ ] QA dataset generation (`build_qa_dataset.py`)
- [ ] Training runs (qlora, dora local; lora on Colab)
- [ ] Full evaluation + results writeup
