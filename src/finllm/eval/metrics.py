"""Quality metrics: Exact Match, token F1, ROUGE-L, BERTScore."""

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/whitespace."""
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)]


def bert_score_f1(predictions: list[str], references: list[str]) -> list[float]:
    from bert_score import score

    _, _, f1 = score(predictions, references, lang="en", rescale_with_baseline=True)
    return f1.tolist()


def compute_all(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Corpus-level means of every metric."""
    n = len(predictions)
    assert n == len(references) and n > 0
    return {
        "exact_match": sum(exact_match(p, r) for p, r in zip(predictions, references)) / n,
        "token_f1": sum(token_f1(p, r) for p, r in zip(predictions, references)) / n,
        "rouge_l": sum(rouge_l(predictions, references)) / n,
        "bert_score_f1": sum(bert_score_f1(predictions, references)) / n,
    }
