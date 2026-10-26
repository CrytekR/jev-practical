"""把场景、读出方法和针对性优化放进同一次可复现实验。"""

from __future__ import annotations

from dataclasses import dataclass

from jev_practical.answers import answer_score
from jev_practical.batching import attention_pairs, readout_tokens, saved_pairs
from jev_practical.calibrate import apply_temperature, fit_temperature
from jev_practical.design import lint_questions
from jev_practical.metrics import expected_calibration_error
from jev_practical.readout import lexical_choice
from jev_practical.scenarios import POOR_FITS, SCENARIOS
from jev_practical.scoring import argmax_index, reduce_token_logprobs
from jev_practical.selective import bayes_threshold, cascade_cost, selective_point
from jev_practical.specs import ChoiceSpec, NoulSpec, ScoreSpec


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    accuracy: float
    ece_before: float
    ece_after: float


@dataclass(frozen=True)
class RoutingResult:
    clear_accuracy: float
    negation_prediction: str
    negation_label: str


@dataclass(frozen=True)
class StudyReport:
    scenario_count: int
    calibration: CalibrationResult
    selective_coverage: float
    selective_risk: float
    overall_error_rate: float
    routing: RoutingResult
    adjacent_loss: float
    bimodal_loss: float
    shared_index: float
    bimodal_index: float
    bayes: float
    letter_tokens: int
    full_tokens: int
    shared_pairs: int
    separate_pairs: int
    design_errors_before: int
    design_warnings_before: int
    design_findings_after: int
    cascade_cost: float
    always_fallback_cost: float
    max_fallback_rate: float
    fallback_rate: float
    sum_choice: str
    mean_choice: str


def run_study() -> StudyReport:
    calibration = _calibration()
    confidences, correct = _selective_sample()
    point = selective_point(confidences, correct, 0.9)
    routing = _routing()
    adjacent = answer_score(_severity(), [0.0, 4.0, 4.0, 0.0], losses=(0.0, 1.0, 10.0, 100.0))
    bimodal = answer_score(_severity(), [4.0, 0.0, 0.0, 4.0], losses=(0.0, 1.0, 10.0, 100.0))
    broken, repaired = _contracts()
    broken_findings = lint_questions(broken)
    fallback_rate = 1.0 - point.coverage
    cost = cascade_cost(decision_cost=0.05, fallback_cost=1.0, fallback_rate=fallback_rate)
    short = reduce_token_logprobs((-0.1,))
    long_tokens = (-0.08, -0.08, -0.08)
    labels = ("short", "long")
    sum_choice = labels[argmax_index((short, reduce_token_logprobs(long_tokens, "sum")))]
    mean_choice = labels[argmax_index((short, reduce_token_logprobs(long_tokens, "mean")))]
    return StudyReport(
        scenario_count=len(SCENARIOS),
        calibration=calibration,
        selective_coverage=point.coverage,
        selective_risk=0.0 if point.selective_risk is None else point.selective_risk,
        overall_error_rate=sum(not item for item in correct) / len(correct),
        routing=routing,
        adjacent_loss=0.0 if adjacent.expected_loss is None else adjacent.expected_loss,
        bimodal_loss=0.0 if bimodal.expected_loss is None else bimodal.expected_loss,
        shared_index=adjacent.score,
        bimodal_index=bimodal.score,
        bayes=bayes_threshold(2.0, 8.0),
        letter_tokens=readout_tokens(
            state_tokens=256, questions=5, options_per_question=77, option_tokens=4, mode="letter"
        ),
        full_tokens=readout_tokens(
            state_tokens=256, questions=5, options_per_question=77, option_tokens=4, mode="full"
        ),
        shared_pairs=attention_pairs(state_tokens=128, branch_tokens=24, questions=3, shared_prefix=True),
        separate_pairs=attention_pairs(state_tokens=128, branch_tokens=24, questions=3, shared_prefix=False),
        design_errors_before=sum(item.severity.value == "error" for item in broken_findings),
        design_warnings_before=sum(item.severity.value == "warning" for item in broken_findings),
        design_findings_after=len(lint_questions(repaired)),
        cascade_cost=cost.cascade,
        always_fallback_cost=cost.always_fallback,
        max_fallback_rate=cost.max_fallback_rate,
        fallback_rate=fallback_rate,
        sum_choice=sum_choice,
        mean_choice=mean_choice,
    )


def render(report: StudyReport) -> str:
    lines = [
        "Jev 类决策模型研究实验",
        "",
        f"场景清单 {report.scenario_count} 个。放不进这类接口的工作：{'；'.join(POOR_FITS)}。",
        "",
        "校准",
        (
            f"过置信合成集上准确率 {report.calibration.accuracy:.3f}。"
            f"温度 {report.calibration.temperature:.3f} 把测试 ECE 从 "
            f"{report.calibration.ece_before:.3f} 调到 {report.calibration.ece_after:.3f}。"
            "argmax 保持不变。"
        ),
        "",
        "选择性执行",
        (
            f"置信度不低于 0.9 时覆盖率 {report.selective_coverage:.3f}，"
            f"接受集错误率 {report.selective_risk:.3f}；全量错误率 {report.overall_error_rate:.3f}。"
        ),
        "",
        "分流读出",
        (
            f"边界清楚的工单准确率 {report.routing.clear_accuracy:.3f}。"
            f"「请不要退款」的金标是 {report.routing.negation_label}，"
            f"词重叠读出给出 {report.routing.negation_prediction}。"
            "描述里的表面词会把否定句吸进对应选项。"
        ),
        "",
        "分数与损失",
        (
            f"相邻犹豫和两端对峙的期望等级同为 {report.shared_index:.3f}。"
            f"损失取 0、1、10、100 时，期望损失分别是 {report.adjacent_loss:.3f} 和 {report.bimodal_loss:.3f}。"
        ),
        "",
        "成本、读出与契约",
        f"误报成本 2、漏报成本 8 时，执行动作的概率阈值是 {report.bayes:.3f}。",
        (
            f"5 道 77 选项的题，单代号读出 {report.letter_tokens} 个计数 token，"
            f"全文本读出 {report.full_tokens} 个。三道题共享 128 token 前缀时，"
            f"可见位置对从 {report.separate_pairs} 降到 {report.shared_pairs}，"
            f"差额 {saved_pairs(state_tokens=128, questions=3)}。"
        ),
        (
            f"长短选项的对数概率用 sum 选出 {report.sum_choice}，用 mean 选出 {report.mean_choice}。"
        ),
        (
            f"有缺口的契约产生 {report.design_errors_before} 个错误、"
            f"{report.design_warnings_before} 个警告；补上描述、兜底和原子问题后剩余 "
            f"{report.design_findings_after} 条。"
        ),
        (
            f"选择性样本的回退率 {report.fallback_rate:.3f}，决策成本 0.05、回退成本 1 时，"
            f"级联期望成本 {report.cascade_cost:.3f}，全程回退成本 {report.always_fallback_cost:.3f}。"
            f"回退率不超过 {report.max_fallback_rate:.3f} 时级联更省。"
        ),
        "",
        "场景",
    ]
    for scenario in SCENARIOS:
        joined = "、".join(scenario.optimizations)
        lines.append(f"- {scenario.title}：{scenario.questions}。优化：{joined}。仍由程序处理：{scenario.hold_in_code}。")
    return "\n".join(lines)


def main() -> None:
    print(render(run_study()))


def _calibration() -> CalibrationResult:
    logits, labels = _overconfident_rows()
    train_logits = logits[::2]
    train_labels = labels[::2]
    test_logits = logits[1::2]
    test_labels = labels[1::2]
    fit = fit_temperature(train_logits, train_labels)
    before = apply_temperature(test_logits, 1.0)
    after = apply_temperature(test_logits, fit.temperature)
    return CalibrationResult(
        temperature=fit.temperature,
        accuracy=_accuracy(before, test_labels),
        ece_before=_ece(before, test_labels),
        ece_after=_ece(after, test_labels),
    )


def _overconfident_rows() -> tuple[list[list[float]], list[int]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for index in range(24):
        label = index % 3
        row = [0.0, 0.0, 0.0]
        row[label] = 5.0
        rows.append(row)
        labels.append(label)
    for index in range(12):
        label = index % 3
        row = [0.0, 0.0, 0.0]
        row[(label + 1) % 3] = 5.0
        rows.append(row)
        labels.append(label)
    return rows, labels


def _selective_sample() -> tuple[list[float], list[bool]]:
    confidences = [0.96] * 32 + [0.55] * 18
    correct = [True] * 30 + [False] * 2 + [True] * 8 + [False] * 10
    return confidences, correct


def _routing() -> RoutingResult:
    spec = ChoiceSpec(
        id="team",
        instructions="这条消息应该交给哪个团队？",
        criteria={
            "billing": "扣款、发票、退款、账单金额",
            "technical": "接口超时、报错、连接失败、页面故障",
            "account": "登录、密码、绑定手机、个人信息",
            "other": "以上都不是，包括与业务无关的内容",
        },
    )
    clear = (
        ("请把发票重开一下，金额不对", "billing"),
        ("我要申请退款，同一笔扣了两次", "billing"),
        ("接口一直超时，返回连接失败", "technical"),
        ("页面报错，刷新也没用", "technical"),
        ("登录不了，提示密码错误", "account"),
        ("我想修改绑定手机和个人信息", "account"),
        ("明天上海会下雨吗", "other"),
    )
    hits = sum(lexical_choice(spec, text).choice == label for text, label in clear)
    negation = lexical_choice(spec, "请不要退款")
    return RoutingResult(
        clear_accuracy=hits / len(clear),
        negation_prediction=negation.choice,
        negation_label="other",
    )


def _severity() -> ScoreSpec:
    return ScoreSpec(
        id="severity",
        instructions="违规严重程度",
        levels=(
            ("none", "没有违规内容"),
            ("low", "轻微，可自动放过并记录"),
            ("high", "明显违规，需要拦截"),
            ("critical", "造成安全或资金损失"),
        ),
    )


def _contracts() -> tuple[list[ChoiceSpec | NoulSpec], list[ChoiceSpec | NoulSpec]]:
    broken: list[ChoiceSpec | NoulSpec] = [
        ChoiceSpec(
            id="team",
            instructions="是否紧急且涉及付款",
            criteria={"billing": "", "technical": "故障、报错、接入"},
        ),
        NoulSpec(id="retry", proposition="根据上一题的结果决定能不能重试"),
        NoulSpec(id="empty", proposition=""),
    ]
    repaired: list[ChoiceSpec | NoulSpec] = [
        ChoiceSpec(
            id="team",
            instructions="这条消息应该交给哪个团队？",
            criteria={
                "billing": "扣款、发票、退款",
                "technical": "故障、报错、接入",
                "other": "以上都不是",
            },
        ),
        NoulSpec(id="urgent", proposition="消息是否要求今天内处理"),
        NoulSpec(id="payment", proposition="消息是否在讨论一笔已经发生的付款"),
    ]
    return broken, repaired


def _accuracy(rows: list[list[float]], labels: list[int]) -> float:
    hits = sum(argmax_index(row) == label for row, label in zip(rows, labels, strict=True))
    return hits / len(labels)


def _ece(rows: list[list[float]], labels: list[int]) -> float:
    confidences = [max(row) for row in rows]
    correct = [argmax_index(row) == label for row, label in zip(rows, labels, strict=True)]
    return expected_calibration_error(confidences, correct)


if __name__ == "__main__":
    main()
