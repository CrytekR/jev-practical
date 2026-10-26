"""把已知候选的对数分数收成分布。

候选在调用时给定，所以读的是调用方提供的文本的概率，而不是生成一段新文本。
温度只改变分布形状，正温度不改变 argmax。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not logits:
        raise ValueError("logits 不能为空")
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def reduce_token_logprobs(
    token_logprobs: Sequence[float],
    how: Literal["sum", "mean"] = "sum",
) -> float:
    """把一个候选的逐 token 对数概率收成一个分数。

    sum 与 mean 会改变长短文本之间的比较结果，校准必须绑定同一种归约。
    """

    if not token_logprobs:
        raise ValueError("token 对数概率不能为空")
    total = math.fsum(token_logprobs)
    if how == "sum":
        return total
    if how == "mean":
        return total / len(token_logprobs)
    raise ValueError(f"未知归约方式: {how}")


def argmax_index(values: Sequence[float]) -> int:
    """同分时保留更靠前的候选，避免用标签文本做隐式决胜。"""

    if not values:
        raise ValueError("values 不能为空")
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index
