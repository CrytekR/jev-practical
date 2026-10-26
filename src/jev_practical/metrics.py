"""概率质量的常用度量：可靠性、Brier 与负对数似然。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> float:
    """等宽分箱的顶层置信度 ECE。空箱不参与。"""

    if len(confidences) != len(correct):
        raise ValueError("置信度与标签数量必须相同")
    if not confidences:
        raise ValueError("样本不能为空")
    if n_bins < 1:
        raise ValueError("n_bins 必须为正")
    bins = reliability_diagram(confidences, correct, n_bins=n_bins)
    total = len(confidences)
    return sum(item.count / total * abs(item.accuracy - item.mean_confidence) for item in bins)


def reliability_diagram(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    if len(confidences) != len(correct):
        raise ValueError("置信度与标签数量必须相同")
    if n_bins < 1:
        raise ValueError("n_bins 必须为正")
    counts = [0] * n_bins
    confidence_sums = [0.0] * n_bins
    correct_sums = [0.0] * n_bins
    for confidence, is_correct in zip(confidences, correct, strict=True):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("置信度必须在 [0, 1]")
        index = n_bins - 1 if confidence >= 1.0 else min(n_bins - 1, int(confidence * n_bins))
        counts[index] += 1
        confidence_sums[index] += confidence
        correct_sums[index] += 1.0 if is_correct else 0.0
    width = 1.0 / n_bins
    diagram: list[ReliabilityBin] = []
    for index, count in enumerate(counts):
        if count == 0:
            continue
        diagram.append(
            ReliabilityBin(
                lower=index * width,
                upper=(index + 1) * width,
                count=count,
                mean_confidence=confidence_sums[index] / count,
                accuracy=correct_sums[index] / count,
            )
        )
    return diagram


def brier_score(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """多类 Brier：各类平方误差先求和，再对样本取平均。"""

    _check_rows(probabilities, labels)
    total = 0.0
    for row, label in zip(probabilities, labels, strict=True):
        total += sum((probability - (1.0 if index == label else 0.0)) ** 2 for index, probability in enumerate(row))
    return total / len(labels)


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    eps: float = 1e-12,
) -> float:
    _check_rows(probabilities, labels)
    if eps <= 0:
        raise ValueError("eps 必须为正")
    total = 0.0
    for row, label in zip(probabilities, labels, strict=True):
        total -= math.log(max(eps, row[label]))
    return total / len(labels)


def _check_rows(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("概率行与标签数量必须相同且非空")
    width = len(probabilities[0])
    if width < 2:
        raise ValueError("至少需要两个类别")
    for row, label in zip(probabilities, labels, strict=True):
        if len(row) != width:
            raise ValueError("每一行的类别数必须相同")
        if not 0 <= label < width:
            raise ValueError("标签超出类别范围")
