"""按任务、模型、推理方法汇总硬标签准确率和样本耗时。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class SampleResult:
    sample_id: str
    task: str
    model_id: str
    method: str
    latency_ms: float
    gold: dict[str, str]
    answers: dict[str, str | None]
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    question_errors: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    raw: str | None = None

    @property
    def counts(self) -> tuple[int, int, int]:
        correct = invalid = wrong = 0
        for key, expected in self.gold.items():
            if key in self.question_errors or self.answers.get(key) is None:
                invalid += 1
            elif self.answers[key] == expected:
                correct += 1
            else:
                wrong += 1
        return correct, invalid, wrong

    @property
    def all_correct(self) -> bool:
        correct, invalid, wrong = self.counts
        return wrong == 0 and invalid == 0 and correct == len(self.gold)


@dataclass(frozen=True)
class SummaryRow:
    task: str
    model_id: str
    method: str
    samples: int
    questions: int
    question_accuracy: float
    invalid_rate: float
    wrong_rate: float
    sample_accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "model": self.model_id,
            "method": self.method,
            "samples": self.samples,
            "questions": self.questions,
            "question_accuracy": self.question_accuracy,
            "invalid_rate": self.invalid_rate,
            "wrong_rate": self.wrong_rate,
            "sample_accuracy": self.sample_accuracy,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
        }


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("没有样本，无法计算分位")
    if not 0 < percent <= 100:
        raise ValueError("分位必须在 (0, 100] 内")
    ordered = sorted(values)
    rank = math.ceil(percent / 100 * len(ordered)) - 1
    return ordered[min(max(rank, 0), len(ordered) - 1)]


def summarize(results: Sequence[SampleResult]) -> list[SummaryRow]:
    groups: dict[tuple[str, str, str], list[SampleResult]] = {}
    for result in results:
        groups.setdefault((result.task, result.model_id, result.method), []).append(result)
    rows: list[SummaryRow] = []
    for task, model_id, method in sorted(groups):
        bucket = groups[(task, model_id, method)]
        correct = invalid = wrong = 0
        for result in bucket:
            one_correct, one_invalid, one_wrong = result.counts
            correct += one_correct
            invalid += one_invalid
            wrong += one_wrong
        questions = correct + invalid + wrong
        if questions == 0:
            accuracy = invalid_rate = wrong_rate = 0.0
        else:
            accuracy = correct / questions
            invalid_rate = invalid / questions
            wrong_rate = wrong / questions
        latencies = [result.latency_ms for result in bucket]
        rows.append(
            SummaryRow(
                task,
                model_id,
                method,
                len(bucket),
                questions,
                accuracy,
                invalid_rate,
                wrong_rate,
                sum(result.all_correct for result in bucket) / len(bucket),
                percentile(latencies, 50),
                percentile(latencies, 95),
            )
        )
    return rows


def render_markdown(rows: Sequence[SummaryRow], concurrency: int) -> str:
    lines = [
        f"并发 {concurrency}。准确率是硬标签与金标一致的题目比例。无效率是没有给出可用标签的题目比例。耗时是一条样本全部请求返回的墙钟时间。",
        "",
        "| 任务 | 模型 | 方法 | 样本 | 题目 | 准确率 | 无效率 | 错误率 | 样本全对 | p50 毫秒 | p95 毫秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {task} | {model} | {method} | {samples} | {questions} | {accuracy:.3f} | {invalid:.3f} | {wrong:.3f} | {sample:.3f} | {p50:.1f} | {p95:.1f} |".format(
                task=row.task,
                model=row.model_id,
                method=row.method,
                samples=row.samples,
                questions=row.questions,
                accuracy=row.question_accuracy,
                invalid=row.invalid_rate,
                wrong=row.wrong_rate,
                sample=row.sample_accuracy,
                p50=row.latency_p50_ms,
                p95=row.latency_p95_ms,
            )
        )
    return "\n".join(lines) + "\n"
