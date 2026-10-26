import pytest

from jev_practical.calibrate import apply_platt, apply_temperature, fit_platt, fit_temperature
from jev_practical.metrics import brier_score, expected_calibration_error, negative_log_likelihood
from jev_practical.scoring import argmax_index, softmax
from jev_practical.selective import bayes_threshold, cascade_cost, selective_point


def test_temperature_fit_flattens_overconfident_logits_without_moving_the_label() -> None:
    logits = []
    labels = []
    for index in range(24):
        label = index % 3
        row = [0.0, 0.0, 0.0]
        row[label] = 5.0
        logits.append(row)
        labels.append(label)
    for index in range(12):
        label = index % 3
        row = [0.0, 0.0, 0.0]
        row[(label + 1) % 3] = 5.0
        logits.append(row)
        labels.append(label)
    fit = fit_temperature(logits, labels)
    before = apply_temperature(logits, 1.0)
    after = apply_temperature(logits, fit.temperature)
    assert fit.temperature > 1
    assert fit.nll_after < fit.nll_before
    assert _accuracy(before, labels) == pytest.approx(_accuracy(after, labels))
    assert _ece(after, labels) < _ece(before, labels) / 2


def test_platt_pulls_a_constant_overconfident_probability_toward_the_base_rate() -> None:
    probabilities = [0.95] * 50
    labels = [1] * 30 + [0] * 20
    fit = fit_platt(probabilities, labels)
    fitted = apply_platt(probabilities, fit)
    assert fit.nll_after < fit.nll_before
    assert abs(sum(fitted) / len(fitted) - 0.6) < abs(0.95 - 0.6)


def test_brier_and_nll_prefer_the_true_distribution() -> None:
    labels = [0, 1, 2]
    sharp = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    flat = [[1 / 3, 1 / 3, 1 / 3]] * 3
    assert brier_score(sharp, labels) < brier_score(flat, labels)
    assert negative_log_likelihood(sharp, labels) < negative_log_likelihood(flat, labels)


def test_bayes_threshold_follows_the_two_error_costs() -> None:
    assert bayes_threshold(2, 8) == pytest.approx(0.2)
    assert bayes_threshold(1, 1) == pytest.approx(0.5)


def test_selective_risk_is_undefined_when_nothing_is_accepted() -> None:
    point = selective_point([0.4, 0.6], [True, False], 0.9)
    assert point.accepted == 0
    assert point.selective_risk is None
    assert point.coverage == 0


def test_cascade_is_cheaper_only_while_fallback_stays_below_the_break_even_rate() -> None:
    cheap = cascade_cost(decision_cost=0.05, fallback_cost=1.0, fallback_rate=0.2)
    expensive = cascade_cost(decision_cost=0.05, fallback_cost=1.0, fallback_rate=0.99)
    assert cheap.max_fallback_rate == pytest.approx(0.95)
    assert cheap.cascade < cheap.always_fallback
    assert expensive.cascade > expensive.always_fallback


def _accuracy(rows: list[list[float]], labels: list[int]) -> float:
    hits = sum(argmax_index(row) == label for row, label in zip(rows, labels, strict=True))
    return hits / len(labels)


def _ece(rows: list[list[float]], labels: list[int]) -> float:
    confidences = [max(row) for row in rows]
    correct = [argmax_index(row) == label for row, label in zip(rows, labels, strict=True)]
    return expected_calibration_error(confidences, correct)


def test_softmax_confidence_uses_the_normalized_peak() -> None:
    row = softmax([5.0, 0.0, 0.0])
    assert max(row) == pytest.approx(row[0])
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
