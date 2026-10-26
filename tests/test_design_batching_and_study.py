import pytest

from jev_practical.answers import answer_noul, answer_score
from jev_practical.batching import attention_pairs, readout_tokens, saved_pairs
from jev_practical.design import Severity, lint_questions
from jev_practical.readout import lexical_choice
from jev_practical.score_shape import expected_index, expected_value, mode_gap
from jev_practical.specs import ChoiceSpec, NoulSpec, ScoreSpec
from jev_practical.study import render, run_study


def test_identical_score_expectations_keep_different_losses_and_shapes() -> None:
    adjacent = (0.0, 0.5, 0.5, 0.0)
    bimodal = (0.5, 0.0, 0.0, 0.5)
    losses = (0.0, 1.0, 10.0, 100.0)
    assert expected_index(adjacent) == pytest.approx(expected_index(bimodal))
    assert mode_gap(adjacent) == 1
    assert mode_gap(bimodal) == 3
    assert expected_value(bimodal, losses) > expected_value(adjacent, losses) * 5


def test_shared_prefix_saves_only_the_repeated_state() -> None:
    shared = attention_pairs(state_tokens=128, branch_tokens=24, questions=3, shared_prefix=True)
    separate = attention_pairs(state_tokens=128, branch_tokens=24, questions=3, shared_prefix=False)
    assert separate - shared == saved_pairs(state_tokens=128, questions=3)
    letter = readout_tokens(state_tokens=256, questions=5, options_per_question=77, option_tokens=4, mode="letter")
    full = readout_tokens(state_tokens=256, questions=5, options_per_question=77, option_tokens=4, mode="full")
    assert letter < full


def test_question_lint_flags_uncovered_and_dependent_questions() -> None:
    findings = lint_questions(
        [
            ChoiceSpec(id="team", instructions="是否紧急且涉及付款", criteria={"billing": "", "technical": "故障"}),
            NoulSpec(id="retry", proposition="根据上一题的结果决定能不能重试"),
            NoulSpec(id="empty", proposition=""),
            ScoreSpec(id="priority", levels=(("low", "低"),)),
        ]
    )
    codes = {item.code for item in findings}
    assert "missing_catchall" in codes
    assert "compound_question" in codes
    assert "dependent_question" in codes
    assert "empty_proposition" in codes
    assert "score_level_count" in codes
    assert any(item.severity is Severity.ERROR for item in findings)
    repaired = lint_questions(
        [
            ChoiceSpec(
                id="team",
                instructions="这条消息应该交给哪个团队？",
                criteria={"billing": "扣款、发票、退款", "other": "以上都不是"},
            )
        ]
    )
    assert repaired == []


def test_lexical_readout_handles_clear_routes_and_misses_negation() -> None:
    spec = ChoiceSpec(
        id="team",
        criteria={
            "billing": "扣款、发票、退款、账单金额",
            "technical": "接口超时、报错、连接失败、页面故障",
            "account": "登录、密码、绑定手机、个人信息",
            "other": "以上都不是，包括与业务无关的内容",
        },
    )
    assert lexical_choice(spec, "请把发票重开一下，金额不对").choice == "billing"
    assert lexical_choice(spec, "明天上海会下雨吗").choice == "other"
    assert lexical_choice(spec, "请不要退款").choice == "billing"


def test_noul_probability_is_the_yes_mass_not_a_confidence_field() -> None:
    certain_no = answer_noul(logit_yes=-4.0, logit_no=4.0)
    assert certain_no.probability < 0.02
    assert certain_no.decisiveness > 0.9


def test_study_report_shows_each_optimization_moving_the_measured_quantity() -> None:
    report = run_study()
    assert report.scenario_count == 6
    assert report.calibration.temperature > 1
    assert report.calibration.accuracy == pytest.approx(2 / 3)
    assert report.calibration.ece_after < report.calibration.ece_before / 2
    assert report.selective_risk < report.overall_error_rate
    assert report.routing.clear_accuracy == pytest.approx(1.0)
    assert report.routing.negation_prediction != report.routing.negation_label
    assert report.shared_index == pytest.approx(report.bimodal_index, abs=0.05)
    assert report.bimodal_loss > report.adjacent_loss * 5
    assert report.bayes == pytest.approx(0.2)
    assert report.letter_tokens < report.full_tokens
    assert report.shared_pairs < report.separate_pairs
    assert report.sum_choice == "short"
    assert report.mean_choice == "long"
    assert report.design_errors_before >= 1
    assert report.design_warnings_before >= 1
    assert report.design_findings_after == 0
    assert report.cascade_cost < report.always_fallback_cost
    text = render(report)
    assert "工单分流" in text
    assert "温度" in text


def test_score_answer_exposes_loss_without_changing_the_index() -> None:
    spec = ScoreSpec(
        id="severity",
        levels=(("none", "没有"), ("low", "轻微"), ("high", "拦截"), ("critical", "损失")),
    )
    left = answer_score(spec, [0.0, 4.0, 4.0, 0.0], losses=(0.0, 1.0, 10.0, 100.0))
    right = answer_score(spec, [4.0, 0.0, 0.0, 4.0], losses=(0.0, 1.0, 10.0, 100.0))
    assert left.score == pytest.approx(right.score, abs=0.05)
    assert right.expected_loss is not None and left.expected_loss is not None
    assert right.expected_loss > left.expected_loss
    assert right.mode_gap > left.mode_gap
