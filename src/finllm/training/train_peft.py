"""Config-driven PEFT fine-tuning: one script covers LoRA, QLoRA, and DoRA.

The method is fully determined by the config file (quantization + peft blocks),
so runs differ only in the knobs MLflow records.

Usage:
    python -m finllm.training.train_peft --config configs/qlora.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from finllm.config import load_config
from finllm.tracking import log_config, setup_mlflow


def build_model_and_tokenizer(config: dict):
    model_name = config["model"]["base_model"]
    quant = config["quantization"]

    quant_config = None
    if quant["load_in_4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if quant_config is None else None,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quant["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = config["peft"]
    lora_config = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        target_modules=peft_cfg["target_modules"],
        use_dora=peft_cfg["use_dora"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def format_example(example: dict, tokenizer) -> str:
    """Render a QA pair with the model's chat template (closed-book: no passage)."""
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]

    setup_mlflow(config)
    with mlflow.start_run(run_name=config["method"]):
        log_config(config)

        processed = Path(config["data"]["processed_dir"])
        dataset = load_dataset(
            "json",
            data_files={"train": str(processed / "train.jsonl"), "val": str(processed / "val.jsonl")},
        )

        model, tokenizer = build_model_and_tokenizer(config)

        sft_config = SFTConfig(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["per_device_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler"],
            warmup_ratio=train_cfg["warmup_ratio"],
            gradient_checkpointing=train_cfg["gradient_checkpointing"],
            fp16=train_cfg["fp16"],
            bf16=train_cfg["bf16"],
            optim=train_cfg.get("optim", "adamw_torch"),
            logging_steps=train_cfg["logging_steps"],
            eval_steps=train_cfg["eval_steps"],
            eval_strategy="steps",
            save_steps=train_cfg["save_steps"],
            max_length=config["model"]["max_seq_length"],
            report_to="mlflow",
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            formatting_func=lambda ex: format_example(ex, tokenizer),
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer.train()

        if torch.cuda.is_available():
            mlflow.log_metric("train_peak_vram_gb", torch.cuda.max_memory_allocated() / 1e9)

        adapter_dir = Path(train_cfg["output_dir"]) / "final_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        print(f"[done] adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
