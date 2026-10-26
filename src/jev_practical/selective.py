"""把概率变成动作：成本阈值、选择性接受，以及带回退的期望成本。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    selective_risk: float | None
    accepted: int
    errors: int
    total: int


@dataclass(frozen=True)
class CascadeCost:
    cascade: float
    always_fallback: float
    max_fallback_rate: float


def bayes_threshold(false_positive_cost: float, false_negative_cost: float) -> float:
    """二元动作里，执行的期望成本低于放弃时，所需的最小 p(yes)。

    正确动作的成本记为 0。阈值是 C_fp / (C_fp + C_fn)，由两类错误的代价决定。
    """

    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("成本不能为负")
    total = false_positive_cost + false_negative_cost
    if total <= 0:
        raise ValueError("两类错误成本不能同时为 0")
    return false_positive_cost / total


def selective_point(
    confidences: Sequence[float],
    correct: Sequence[bool],
    threshold: float,
) -> SelectivePoint:
    """覆盖率是被自动接受的比例。选择性风险是接受集里的错误率；无人被接受时为空。"""

    if len(confidences) != len(correct) or not confidences:
        raise ValueError("置信度与标签数量必须相同且非空")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须在 [0, 1]")
    accepted = 0
    errors = 0
    for confidence, is_correct in zip(confidences, correct, strict=True):
        if confidence >= threshold:
            accepted += 1
            if not is_correct:
                errors += 1
    risk = None if accepted == 0 else errors / accepted
    return SelectivePoint(
        threshold=threshold,
        coverage=accepted / len(confidences),
        selective_risk=risk,
        accepted=accepted,
        errors=errors,
        total=len(confidences),
    )


def cascade_cost(
    *,
    decision_cost: float,
    fallback_cost: float,
    fallback_rate: float,
    extra_cost: float = 0.0,
) -> CascadeCost:
    """C = 额外处理 + 决策调用 + 回退率 * 回退成本，并给出仍不比全程回退更贵的最大回退率。"""

    for name, value in (
        ("decision_cost", decision_cost),
        ("fallback_cost", fallback_cost),
        ("extra_cost", extra_cost),
    ):
        if value < 0:
            raise ValueError(f"{name} 不能为负")
    if not 0.0 <= fallback_rate <= 1.0:
        raise ValueError("fallback_rate 必须在 [0, 1]")
    cascade = extra_cost + decision_cost + fallback_rate * fallback_cost
    if fallback_cost <= 0:
        max_rate = 1.0
    else:
        max_rate = 1.0 - (extra_cost + decision_cost) / fallback_cost
        max_rate = min(1.0, max(0.0, max_rate))
    return CascadeCost(cascade=cascade, always_fallback=fallback_cost, max_fallback_rate=max_rate)
