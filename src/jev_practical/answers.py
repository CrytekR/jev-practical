"""由对数分数组装三类答案。损失向量只影响 Score 的期望损失，不参与选档。"""

from __future__ import annotations

from collections.abc import Sequence

from jev_practical.confidence import margin, noul_decisiveness, shape_confidence
from jev_practical.score_shape import expected_index, expected_value, mode_gap
from jev_practical.scoring import argmax_index, softmax
from jev_practical.specs import ChoiceAnswer, ChoiceSpec, NoulAnswer, ScoreAnswer, ScoreSpec


def answer_choice(spec: ChoiceSpec, logits: Sequence[float], *, temperature: float = 1.0) -> ChoiceAnswer:
    labels = list(spec.criteria)
    if len(labels) != len(logits):
        raise ValueError(f"{spec.id} 的分数数量与选项数量不一致")
    probabilities = softmax(logits, temperature)
    mapping = dict(zip(labels, probabilities, strict=True))
    return ChoiceAnswer(
        choice=labels[argmax_index(probabilities)],
        probabilities=mapping,
        confidence=shape_confidence(probabilities),
        margin=margin(probabilities),
    )


def answer_score(
    spec: ScoreSpec,
    logits: Sequence[float],
    *,
    temperature: float = 1.0,
    losses: Sequence[float] | None = None,
) -> ScoreAnswer:
    labels = [label for label, _ in spec.levels]
    if len(labels) != len(logits):
        raise ValueError(f"{spec.id} 的分数数量与等级数量不一致")
    probabilities = softmax(logits, temperature)
    mapping = dict(zip(labels, probabilities, strict=True))
    expected_loss = None if losses is None else expected_value(probabilities, losses)
    return ScoreAnswer(
        score=expected_index(probabilities),
        probabilities=mapping,
        confidence=shape_confidence(probabilities),
        margin=margin(probabilities),
        mode_gap=mode_gap(probabilities),
        expected_loss=expected_loss,
    )


def answer_noul(logit_yes: float, logit_no: float = 0.0, *, temperature: float = 1.0) -> NoulAnswer:
    yes, _no = softmax((logit_yes, logit_no), temperature)
    return NoulAnswer(probability=yes, decisiveness=noul_decisiveness(yes))
