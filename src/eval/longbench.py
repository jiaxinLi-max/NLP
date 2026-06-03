"""LongBench eval — subset of tasks with the official prompts and metrics.

Tasks in our subset:
    narrativeqa  -> token-F1
    qasper       -> token-F1
    gov_report   -> ROUGE-L
    hotpotqa     -> token-F1

Inputs longer than `max_length` get middle-truncated, matching the official
LongBench `pred.py` recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from datasets import load_dataset

from .metrics import best_over_golds, rouge_l, token_f1


PROMPTS: Dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\nNow, answer the question based on the story "
        "asap and in as few words as possible.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\n"
        "Article: {context}\n\n Answer the question based on the above "
        "article as concisely as you can, using a single phrase or sentence if "
        "possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no "
        "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide "
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page "
        "summary of the report.\n\nSummary:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words.\n"
        "\nQuestion: {input}\nAnswer:"
    ),
}

MAX_NEW: Dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "gov_report": 512,
    "hotpotqa": 32,
}

METRIC: Dict[str, Callable[[str, list[str]], float]] = {
    "narrativeqa": lambda p, g: best_over_golds(p, g, token_f1),
    "qasper": lambda p, g: best_over_golds(p, g, token_f1),
    "gov_report": lambda p, g: best_over_golds(p, g, rouge_l),
    "hotpotqa": lambda p, g: best_over_golds(p, g, token_f1),
}


@dataclass
class LongBenchExample:
    task: str
    prompt: str
    answers: List[str]
    max_new: int


def middle_truncate(tokenizer, text: str, max_length: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_length:
        return text
    half = max_length // 2
    kept = ids[:half] + ids[-half:]
    return tokenizer.decode(kept, skip_special_tokens=True)


def load_longbench(
    task: str,
    tokenizer,
    max_length: int = 3500,
    limit: int | None = None,
) -> List[LongBenchExample]:
    """Load and pre-render prompts (with middle-truncation) for a single task.

    Prefers local /work/data/longbench/data/{task}.jsonl (extracted from the
    official ModelScope mirror's data.zip); falls back to HF datasets.

    `max_length` is in tokens of the prompt INPUT (context+question), leaving
    headroom for max_new and any chat template wrappers.
    """
    if task not in PROMPTS:
        raise ValueError(f"unsupported task: {task}; pick from {list(PROMPTS)}")

    import json
    import os
    local = f"/work/data/longbench/data/{task}.jsonl"
    if os.path.exists(local):
        rows = []
        with open(local, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        ds = rows
    else:
        ds = load_dataset("THUDM/LongBench", task, split="test")

    examples: List[LongBenchExample] = []
    template = PROMPTS[task]
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        ctx = row["context"]
        ctx = middle_truncate(tokenizer, ctx, max_length)
        prompt = template.format(context=ctx, input=row.get("input", ""))
        examples.append(
            LongBenchExample(
                task=task,
                prompt=prompt,
                answers=list(row.get("answers", [])),
                max_new=MAX_NEW[task],
            )
        )
    return examples


def score(task: str, pred: str, golds: List[str]) -> float:
    return METRIC[task](pred, golds)
