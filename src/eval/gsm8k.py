"""GSM8K eval — 8-shot CoT, regex extraction of #### <number>."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from datasets import load_dataset


GSM8K_FEWSHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in "
        "April and May?",
        "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
        "of babysitting. How much did she earn?",
        "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 * 50 = $10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only "
        "half of the money she needs. Her parents decided to give her $15 for that "
        "purpose, and her grandparents twice as much as her parents. How much more "
        "money does Betty need to buy the wallet?",
        "Betty has 100/2 = $50.\nGrandparents give 15*2 = $30.\nTotal so far: 50+15+30 = $95.\nShe still needs 100-95 = $5.\n#### 5",
    ),
    (
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages "
        "and today, she read twice as many pages as yesterday. If she wants to read "
        "half of the remaining pages tomorrow, how many pages should she read?",
        "Today she read 12*2 = 24.\nTotal read: 12+24 = 36.\nRemaining: 120-36 = 84.\nTomorrow she reads 84/2 = 42.\n#### 42",
    ),
]


def build_prompt(question: str) -> str:
    parts = []
    for q, a in GSM8K_FEWSHOT:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


_NUMERIC = re.compile(r"-?\d+(?:\.\d+)?")
_FINAL = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def extract_answer(text: str) -> str | None:
    m = _FINAL.search(text)
    if m:
        return m.group(1)
    nums = _NUMERIC.findall(text)
    return nums[-1] if nums else None


def gold_answer(answer_field: str) -> str:
    """GSM8K gold answers end with `#### <number>`."""
    m = _FINAL.search(answer_field)
    if m:
        return m.group(1)
    nums = _NUMERIC.findall(answer_field)
    return nums[-1] if nums else ""


def is_correct(pred: str, gold: str) -> bool:
    p = extract_answer(pred)
    if p is None:
        return False
    try:
        return abs(float(p) - float(gold)) < 1e-4
    except ValueError:
        return False


@dataclass
class GSM8KExample:
    question: str
    answer: str  # gold numeric answer string


def load_gsm8k(split: str = "test", limit: int | None = None) -> List[GSM8KExample]:
    """Load GSM8K. Prefers local /work/data/gsm8k/{split}.jsonl (from openai's
    raw GH release); falls back to HF datasets if not present.
    """
    import json
    import os
    local = f"/root/bayes-tmp/project/data/gsm8k/{split}.jsonl"
    if os.path.exists(local):
        out: List[GSM8KExample] = []
        with open(local) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                row = json.loads(line)
                out.append(GSM8KExample(
                    question=row["question"], answer=gold_answer(row["answer"]),
                ))
        return out

    ds = load_dataset("gsm8k", "main", split=split)
    out: List[GSM8KExample] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(GSM8KExample(question=row["question"], answer=gold_answer(row["answer"])))
    return out
