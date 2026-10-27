"""两种推理：一次聊天生成，或按题顺序读取字母对数概率。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from jev_compare.data import Sample
from jev_compare.parse import canonicalize, extract_json_object
from jev_compare.prompt import LETTERS, render_qwen_prompt, render_suffix


class Client(Protocol):
    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str: ...

    def slot_logprobs(self, prompt: str, letters: list[str]) -> dict[str, float]: ...


@dataclass
class Prediction:
    answers: dict[str, str | None]
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    question_errors: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    raw: str | None = None


def predict_generate(client: Client, sample: Sample, max_tokens: int) -> Prediction:
    raw = client.chat([dict(message) for message in sample.messages], max_tokens)
    parsed, error = extract_json_object(raw)
    if error or parsed is None:
        return Prediction({}, error=error, raw=raw)
    answers: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        label = canonicalize(value)
        if label is None:
            answers[key] = None
            errors[key] = "字段值不能转换成标签"
        else:
            answers[key] = label
    return Prediction(answers, question_errors=errors, raw=raw)


def predict_jev(client: Client, sample: Sample) -> Prediction:
    answers: dict[str, str | None] = {}
    probabilities: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}
    for qid, question in sample.questions.items():
        suffix, labels = render_suffix(question)
        prompt = render_qwen_prompt(sample.state, suffix)
        letters = list(LETTERS[: len(labels)])
        try:
            found = client.slot_logprobs(prompt, letters)
            label, probs = _label_from_logprobs(letters, labels, found)
        except Exception as exc:
            answers[qid] = None
            errors[qid] = _short(exc)
            continue
        answers[qid] = label
        probabilities[qid] = probs
    error = "; ".join(f"{qid}: {message}" for qid, message in errors.items()) or None
    return Prediction(answers, probabilities, errors, error)


def _label_from_logprobs(
    letters: list[str], labels: tuple[str, ...], found: dict[str, float]
) -> tuple[str, dict[str, float]]:
    values = [found[letter] for letter in letters]
    probabilities = _renormalize(values)
    best = max(range(len(values)), key=values.__getitem__)
    return labels[best], {label: probability for label, probability in zip(labels, probabilities, strict=True)}


def _renormalize(logprobs: list[float]) -> list[float]:
    peak = max(logprobs)
    weights = [math.exp(value - peak) for value in logprobs]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _short(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[:500]
