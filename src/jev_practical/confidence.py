"""从分布形状读出集中度，并单独保留第一名与第二名的差距。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def shape_confidence(probabilities: Mapping[str, float] | Sequence[float]) -> float:
    """公开文档演示里的集中度近似：(n * peak - 1) / (n - 1)，截断到 [0, 1]。

    它描述分布有多尖，不是「这次判断正确」的概率。只看峰值，所以亚军远近不会改变它。
    """

    values = _as_values(probabilities)
    if len(values) < 2:
        raise ValueError("至少需要两个候选才能计算集中度")
    if any(value < 0 for value in values):
        raise ValueError("概率不能为负")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("概率质量必须为正")
    peak = max(values) / total
    count = len(values)
    raw = (count * peak - 1.0) / (count - 1.0)
    return min(1.0, max(0.0, raw))


def margin(probabilities: Mapping[str, float] | Sequence[float]) -> float:
    """第一名概率减去第二名概率。峰值相同而亚军不同时，集中度公式给不出这个差别。"""

    values = sorted(_as_values(probabilities), reverse=True)
    if len(values) < 2:
        raise ValueError("至少需要两个候选才能计算分差")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("概率质量必须为正")
    return (values[0] - values[1]) / total


def noul_decisiveness(probability: float) -> float:
    """是非概率离 0.5 有多远，映到 [0, 1]。Noul 接口本身不返回 confidence。"""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability 必须在 [0, 1]")
    return abs(probability - 0.5) * 2.0


def _as_values(probabilities: Mapping[str, float] | Sequence[float]) -> list[float]:
    if isinstance(probabilities, Mapping):
        return [float(value) for value in probabilities.values()]
    return [float(value) for value in probabilities]
