"""Score 的期望下标会把不同形状的分布收成同一个数。这里把形状和任务损失分开算。"""

from __future__ import annotations

from collections.abc import Sequence


def expected_value(probabilities: Sequence[float], values: Sequence[float]) -> float:
    if len(probabilities) != len(values) or not probabilities:
        raise ValueError("概率与取值长度必须相同且非空")
    _check_distribution(probabilities)
    return sum(probability * value for probability, value in zip(probabilities, values, strict=True))


def expected_index(probabilities: Sequence[float]) -> float:
    return expected_value(probabilities, list(range(len(probabilities))))


def mode_gap(probabilities: Sequence[float]) -> int:
    """概率最高的两档相距多少级。相邻犹豫是 1，两端对峙接近等级数减 1。"""

    _check_distribution(probabilities)
    ranked = sorted(enumerate(probabilities), key=lambda item: (item[1], item[0]), reverse=True)
    if len(ranked) < 2 or ranked[1][1] == 0:
        return 0
    return abs(ranked[0][0] - ranked[1][0])


def _check_distribution(probabilities: Sequence[float]) -> None:
    if any(probability < 0 for probability in probabilities):
        raise ValueError("概率不能为负")
    total = sum(probabilities)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("概率必须构成分布")
