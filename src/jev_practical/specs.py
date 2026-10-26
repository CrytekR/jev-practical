"""与 Jev 类接口对齐的问题规格。

一次请求携带同一份 state 和若干互不读取对方答案的问题。
Choice 在无序候选中选一个，Score 在有序等级上给出分布，Noul 给出命题为真的概率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ChoiceSpec:
    id: str
    criteria: dict[str, str]
    instructions: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("问题 id 不能为空")


@dataclass(frozen=True)
class ScoreSpec:
    """levels 从低到高，顺序就是期望分数使用的位置 0, 1, 2, ..."""

    id: str
    levels: tuple[tuple[str, str], ...]
    instructions: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("问题 id 不能为空")
        labels = [label for label, _ in self.levels]
        if len(labels) != len(set(labels)):
            raise ValueError(f"{self.id} 的等级名重复")


@dataclass(frozen=True)
class NoulSpec:
    id: str
    proposition: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("问题 id 不能为空")


QuestionSpec = Union[ChoiceSpec, ScoreSpec, NoulSpec]


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    margin: float


@dataclass(frozen=True)
class ScoreAnswer:
    """score 是等级下标的期望。expected_loss 仅在调用方提供每级损失时填写。"""

    score: float
    probabilities: dict[str, float]
    confidence: float
    margin: float
    mode_gap: int
    expected_loss: float | None = None


@dataclass(frozen=True)
class NoulAnswer:
    probability: float
    decisiveness: float
