"""Evaluation helpers for prompt-only biomedical QA baselines."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from biomed.reward import compute_score


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return "".join(text.split())


def valid_answer(dataset: str, answer: str) -> bool:
    if dataset == "pubmedqa_labeled":
        return re.fullmatch(r"\s*(yes|no|maybe)\s*", str(answer), flags=re.I) is not None
    return re.fullmatch(r"\s*[A-Za-z]\s*", str(answer)) is not None


def clean_answer(dataset: str, answer: str) -> str:
    answer = str(answer).strip()
    if dataset == "pubmedqa_labeled":
        match = re.fullmatch(r"\s*(yes|no|maybe)\s*", answer, flags=re.I)
        return match.group(1).lower() if match else answer
    match = re.fullmatch(r"\s*([A-Za-z])\s*", answer)
    return match.group(1).upper() if match else answer


def _extract_from_tag(dataset: str, text: str) -> str:
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S))
    for match in reversed(matches):
        content = match.group(1).strip()
        if valid_answer(dataset, content):
            return clean_answer(dataset, content)
        token = _extract_valid_token(dataset, content)
        if token:
            return token
    return ""


def _extract_valid_token(dataset: str, text: str) -> str:
    if dataset == "pubmedqa_labeled":
        matches = list(re.finditer(r"\b(yes|no|maybe)\b", text, flags=re.I))
        return matches[-1].group(1).lower() if matches else ""
    matches = list(re.finditer(r"\b([A-E])\b", text, flags=re.I))
    return matches[-1].group(1).upper() if matches else ""


def extract_final_answer(dataset: str, text: str) -> str:
    """Extract a final-answer token without changing the answer semantics."""
    text = str(text or "").strip()
    if not text:
        return ""

    tagged = _extract_from_tag(dataset, text)
    if tagged:
        return tagged

    if valid_answer(dataset, text):
        return clean_answer(dataset, text)

    if dataset == "pubmedqa_labeled":
        choices = r"(yes|no|maybe)"
        patterns = [
            rf"(?:final\s+answer|answer|correct\s+answer)\s*(?:is\s*:|is|:)?\s*{choices}\b",
            rf"\btherefore\s*,?\s*(?:the\s+answer\s+is\s*)?{choices}\b",
        ]
    else:
        choices = r"([A-E])"
        patterns = [
            rf"(?:final\s+answer|answer|correct\s+answer|correct\s+option|option)\s*(?:is\s*:|is|:)?\s*\(?{choices}\)?\b",
            rf"\btherefore\s*,?\s*(?:the\s+answer\s+is\s*)?\(?{choices}\)?\b",
        ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.I))
        if matches:
            return clean_answer(dataset, matches[-1].group(1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        if valid_answer(dataset, line):
            return clean_answer(dataset, line)

    return ""


def score_extracted_answer(data_source: str, dataset: str, answer: str, targets: list[str]) -> dict[str, Any]:
    if not valid_answer(dataset, answer):
        return {"score": 0.0, "format": 0.0, "accuracy": 0.0, "answer": answer}
    scored_output = f"<answer>{clean_answer(dataset, answer)}</answer>"
    metrics = compute_score(data_source, scored_output, {"target": targets})
    metrics["answer"] = clean_answer(dataset, answer)
    metrics["scored_output"] = scored_output
    return metrics


def score_baseline_output(data_source: str, dataset: str, output: str, targets: list[str]) -> dict[str, Any]:
    answer = extract_final_answer(dataset, output)
    return score_extracted_answer(data_source, dataset, answer, targets)


def majority_vote(dataset: str, outputs: list[str]) -> tuple[str, str, list[str]]:
    answers = [extract_final_answer(dataset, output) for output in outputs]
    valid = [answer for answer in answers if valid_answer(dataset, answer)]
    if valid:
        counts = Counter(normalize_answer(answer) for answer in valid)
        winner_norm, _ = counts.most_common(1)[0]
        for answer in valid:
            if normalize_answer(answer) == winner_norm:
                answer = clean_answer(dataset, answer)
                return answer, f"<answer>{answer}</answer>", answers
    first = clean_answer(dataset, answers[0]) if answers else ""
    return first, f"<answer>{first}</answer>" if first else "", answers
