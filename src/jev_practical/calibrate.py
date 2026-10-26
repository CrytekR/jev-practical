"""在标注校准集上调整分布形状。温度缩放不改变 argmax，Platt 用于是非概率。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from jev_practical.metrics import negative_log_likelihood
from jev_practical.scoring import softmax


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float


@dataclass(frozen=True)
class PlattFit:
    slope: float
    intercept: float
    nll_before: float
    nll_after: float


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    grid: Sequence[float] | None = None,
) -> TemperatureFit:
    """在给定网格上最小化负对数似然，选出温度。"""

    if len(logits) != len(labels) or not labels:
        raise ValueError("logits 与标签数量必须相同且非空")
    candidates = list(grid) if grid is not None else _temperature_grid()
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("温度网格必须为正数")
    before = _nll_at(logits, labels, 1.0)
    best_temperature = candidates[0]
    best_nll = math.inf
    for temperature in candidates:
        score = _nll_at(logits, labels, temperature)
        if score < best_nll:
            best_nll = score
            best_temperature = temperature
    return TemperatureFit(temperature=best_temperature, nll_before=before, nll_after=best_nll)


def apply_temperature(logits: Sequence[Sequence[float]], temperature: float) -> list[list[float]]:
    return [softmax(row, temperature) for row in logits]


def fit_platt(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    slopes: Sequence[float] | None = None,
    intercepts: Sequence[float] | None = None,
) -> PlattFit:
    """用 q = sigmoid(a * logit(p) + b) 拟合是非概率。a = 0 时退化为常数概率。"""

    if len(probabilities) != len(labels) or not labels:
        raise ValueError("概率与标签数量必须相同且非空")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Platt 标签必须是 0 或 1")
    logits = [_clip_logit(probability) for probability in probabilities]
    slope_grid = list(slopes) if slopes is not None else [index / 10 for index in range(0, 41)]
    intercept_grid = list(intercepts) if intercepts is not None else [index / 10 for index in range(-30, 31)]
    before = _binary_nll([_sigmoid(value) for value in logits], labels)
    best = PlattFit(slope=1.0, intercept=0.0, nll_before=before, nll_after=before)
    for slope in slope_grid:
        for intercept in intercept_grid:
            fitted = [_sigmoid(slope * logit + intercept) for logit in logits]
            score = _binary_nll(fitted, labels)
            if score < best.nll_after:
                best = PlattFit(slope=slope, intercept=intercept, nll_before=before, nll_after=score)
    return best


def apply_platt(probabilities: Sequence[float], fit: PlattFit) -> list[float]:
    return [_sigmoid(fit.slope * _clip_logit(probability) + fit.intercept) for probability in probabilities]


def _nll_at(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    return negative_log_likelihood(apply_temperature(logits, temperature), labels)


def _temperature_grid() -> list[float]:
    low = 0.05
    high = 5.0
    steps = 80
    return [low * (high / low) ** (index / (steps - 1)) for index in range(steps)]


def _clip_logit(probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability 必须在 [0, 1]")
    clipped = min(1.0 - 1e-6, max(1e-6, probability))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_nll(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    total = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        chosen = probability if label == 1 else 1.0 - probability
        total -= math.log(max(1e-12, chosen))
    return total / len(labels)
