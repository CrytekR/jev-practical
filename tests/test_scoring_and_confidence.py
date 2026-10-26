import math

import pytest

from jev_practical.confidence import margin, noul_decisiveness, shape_confidence
from jev_practical.scoring import argmax_index, reduce_token_logprobs, softmax


def test_softmax_is_stable_and_sums_to_one() -> None:
    probabilities = softmax([1000.0, 1000.0, 999.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx(probabilities[1])
    assert probabilities[0] > probabilities[2]


def test_positive_temperature_preserves_argmax() -> None:
    logits = [0.2, 1.4, -0.3, 1.1]
    for temperature in (0.2, 1.0, 4.0):
        assert argmax_index(softmax(logits, temperature)) == 1


def test_ties_keep_the_earlier_option() -> None:
    assert argmax_index([0.2, 0.2, 0.1]) == 0


def test_mean_and_sum_can_rank_options_differently() -> None:
    short = reduce_token_logprobs((-0.1,))
    long_sum = reduce_token_logprobs((-0.08, -0.08, -0.08), "sum")
    long_mean = reduce_token_logprobs((-0.08, -0.08, -0.08), "mean")
    assert short > long_sum
    assert long_mean > short


def test_shape_confidence_matches_the_documented_demo() -> None:
    assert shape_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert shape_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert shape_confidence([0.9, 0.06, 0.04]) == pytest.approx((3 * 0.9 - 1) / 2)


def test_same_peak_can_hide_a_different_runner_up() -> None:
    near_tie = {"returns": 0.60, "billing": 0.35, "shipping": 0.05}
    clear_second = {"returns": 0.60, "billing": 0.20, "shipping": 0.20}
    assert shape_confidence(near_tie) == pytest.approx(shape_confidence(clear_second))
    assert margin(near_tie) == pytest.approx(0.25)
    assert margin(clear_second) == pytest.approx(0.40)


def test_noul_decisiveness_treats_both_extremes_as_decided() -> None:
    assert noul_decisiveness(0.5) == pytest.approx(0.0)
    assert noul_decisiveness(0.02) == pytest.approx(noul_decisiveness(0.98))
    assert noul_decisiveness(0.0) == pytest.approx(1.0)


def test_invalid_scores_are_rejected() -> None:
    with pytest.raises(ValueError):
        softmax([0.0], temperature=0)
    with pytest.raises(ValueError):
        shape_confidence([1.0])
    with pytest.raises(ValueError):
        noul_decisiveness(1.2)
    assert math.isfinite(softmax([-1000.0, -1001.0])[0])
