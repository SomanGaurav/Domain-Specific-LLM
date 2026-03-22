"""Evaluate one system (base model, PEFT adapter, or RAG) on the shared test set.

Every run logs the same metric names to MLflow, so runs are directly comparable
in the UI:  quality (EM, token F1, ROUGE-L, BERTScore), latency (mean/p50/p95),
and peak GPU memory.

Usage:
    python -m finllm.eval.run_eval --config configs/base.yaml  --system base
    python -m finllm.eval.run_eval --config configs/qlora.yaml --system peft \
        --adapter checkpoints/qlora/final_adapter
    python -m finllm.eval.run_eval --config configs/rag.yaml   --system rag
"""

import argparse
import json
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from finllm.config import load_config
from finllm.eval.metrics import compute_all
from finllm.eval.profiling import LatencyTracker, gpu_memory_snapshot, reset_gpu_peak
from finllm.tracking import log_config, setup_mlflow


def load_test_set(config: dict) -> list[dict]:
    path = Path(config["data"]["processed_dir"]) / "test.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class HFGenerator:
    """Plain generator for the base model or a PEFT-adapted model (closed-book)."""

    def __init__(self, config: dict, adapter: str | None):
        model_name = config["model"]["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.gen = config["generation"]

    def answer(self, question: str) -> dict:
        messages = [{"role": "user", "content": question}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.gen["max_new_tokens"],
                do_sample=self.gen["temperature"] > 0,
                temperature=self.gen["temperature"] or None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        return {"answer": text.strip()}


def build_system(name: str, config: dict, adapter: str | None):
    if name in ("base", "peft"):
        return HFGenerator(config, adapter if name == "peft" else None)
    if name == "rag":
        from finllm.rag.pipeline import RAGPipeline

        return RAGPipeline(config)
    raise ValueError(f"unknown system: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--system", choices=["base", "peft", "rag"], required=True)
    parser.add_argument("--adapter", default=None, help="adapter dir (system=peft only)")
    parser.add_argument("--limit", type=int, default=None, help="cap test examples (smoke runs)")
    args = parser.parse_args()

    config = load_config(args.config)
    test_set = load_test_set(config)
    if args.limit:
        test_set = test_set[: args.limit]

    setup_mlflow(config)
    run_name = f"eval-{args.system}" + (f"-{config['method']}" if args.system == "peft" else "")
    with mlflow.start_run(run_name=run_name):
        log_config(config)
        mlflow.log_param("eval.system", args.system)
        mlflow.log_param("eval.num_examples", len(test_set))

        system = build_system(args.system, config, args.adapter)
        reset_gpu_peak()
        tracker = LatencyTracker()

        predictions, references, records = [], [], []
        for example in tqdm(test_set, desc=f"eval {args.system}"):
            with tracker.track():
                result = system.answer(example["question"])
            predictions.append(result["answer"])
            references.append(example["answer"])
            records.append({**example, "prediction": result["answer"],
                            "sources": result.get("sources")})

        metrics = compute_all(predictions, references)
        metrics.update(tracker.summary())
        metrics.update(gpu_memory_snapshot())
        mlflow.log_metrics(metrics)

        pred_path = Path("data/processed") / f"predictions_{run_name}.jsonl"
        pred_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
