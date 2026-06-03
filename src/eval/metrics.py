"""Common metrics: exact match, token-F1, ROUGE-L."""

from __future__ import annotations

import re
import string
from collections import Counter

from rouge_score import rouge_scorer


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pt = _normalize(pred).split()
    gt = _normalize(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pt)
    r = num_same / len(gt)
    return 2 * p * r / (p + r)


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, gold: str) -> float:
    return _ROUGE.score(gold, pred)["rougeL"].fmeasure


def best_over_golds(pred: str, golds: list[str], metric) -> float:
    return max((metric(pred, g) for g in golds), default=0.0) if golds else 0.0
