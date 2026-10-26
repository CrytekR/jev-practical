"""Jev 类决策模型的场景、方法与针对性优化研究工具。

这里的实现是部署侧可检验的方法：读出、归一化、校准、选择性执行和成本阈值。
本地词法读出器只用来复现这些步骤，不代表某一家模型的权重。
"""

from jev_practical.batching import attention_pairs, readout_tokens
from jev_practical.calibrate import TemperatureFit, fit_platt, fit_temperature
from jev_practical.confidence import margin, noul_decisiveness, shape_confidence
from jev_practical.design import Finding, lint_questions
from jev_practical.metrics import brier_score, expected_calibration_error, negative_log_likelihood
from jev_practical.scoring import reduce_token_logprobs, softmax
from jev_practical.selective import bayes_threshold, cascade_cost, selective_point

__all__ = [
    "Finding",
    "TemperatureFit",
    "attention_pairs",
    "bayes_threshold",
    "brier_score",
    "cascade_cost",
    "expected_calibration_error",
    "fit_platt",
    "fit_temperature",
    "lint_questions",
    "margin",
    "negative_log_likelihood",
    "noul_decisiveness",
    "readout_tokens",
    "reduce_token_logprobs",
    "selective_point",
    "shape_confidence",
    "softmax",
]
