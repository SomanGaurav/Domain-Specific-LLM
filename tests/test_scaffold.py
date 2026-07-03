"""Dependency-free smoke tests for the scaffold (config system + pure metrics)."""

from finllm.config import load_config
from finllm.eval.metrics import exact_match, normalize, token_f1


def test_config_inheritance():
    config = load_config("configs/qlora.yaml")
    assert config["method"] == "qlora"
    assert config["quantization"]["load_in_4bit"] is True
    # inherited from base.yaml
    assert config["model"]["base_model"].startswith("Qwen")
    assert config["mlflow"]["experiment_name"] == "finllm-peft-vs-rag"


def test_dora_overrides_qlora():
    config = load_config("configs/dora.yaml")
    assert config["method"] == "dora"
    assert config["peft"]["use_dora"] is True
    # everything else inherited from qlora.yaml
    assert config["quantization"]["load_in_4bit"] is True
    assert config["training"]["output_dir"] == "checkpoints/dora"


def test_normalize():
    assert normalize("The Revenue, was $10M!") == "revenue was 10m"


def test_exact_match_and_f1():
    assert exact_match("The answer is 42", "answer is 42.") == 1.0
    assert token_f1("net revenue grew 10%", "revenue grew") > 0.5
    assert token_f1("completely different", "no overlap here") == 0.0
