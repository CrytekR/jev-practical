"""上线前检查问题契约。模型只会在给定候选里作答，缺选项时置信度不会自动变低。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from jev_practical.specs import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    question_id: str
    code: str
    severity: Severity
    message: str


_CATCHALL = (
    "other",
    "none",
    "unknown",
    "undetermined",
    "其他",
    "以上都不是",
    "无法确定",
    "不确定",
    "不清楚",
    "无关",
)
_COMPOUND = ("且", "并且", "同时", " and ", " AND ")
_DEPENDENT = ("上一题", "上一个问题", "前一题", "根据前面", "previous answer", "prior answer")


def lint_questions(questions: tuple[QuestionSpec, ...] | list[QuestionSpec]) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            findings.append(Finding(question.id, "duplicate_id", Severity.ERROR, "同一请求里问题 id 重复"))
        seen.add(question.id)
        if isinstance(question, ChoiceSpec):
            findings.extend(_lint_choice(question))
        elif isinstance(question, ScoreSpec):
            findings.extend(_lint_score(question))
        elif isinstance(question, NoulSpec):
            findings.extend(_lint_noul(question))
        else:
            raise TypeError(f"未知问题类型: {type(question)!r}")
    return findings


def _lint_choice(question: ChoiceSpec) -> list[Finding]:
    findings: list[Finding] = []
    count = len(question.criteria)
    if count < 2:
        findings.append(Finding(question.id, "too_few_options", Severity.ERROR, "choice 至少需要 2 个选项"))
    if count > 255:
        findings.append(Finding(question.id, "too_many_options", Severity.ERROR, "choice 最多 255 个选项"))
    texts = [f"{key} {description}" for key, description in question.criteria.items()]
    if not any(_contains_any(text.casefold(), _CATCHALL) for text in texts):
        findings.append(
            Finding(
                question.id,
                "missing_catchall",
                Severity.WARNING,
                "选项没有兜底项。输入落在候选之外时，模型仍会选出一项，而且置信度可以很高",
            )
        )
    for key, description in question.criteria.items():
        if not description.strip() or description.strip() == key:
            findings.append(
                Finding(question.id, "missing_description", Severity.WARNING, f"选项 {key} 缺少界定范围的描述")
            )
    findings.extend(_lint_wording(question.id, question.instructions))
    return findings


def _lint_score(question: ScoreSpec) -> list[Finding]:
    findings: list[Finding] = []
    count = len(question.levels)
    if count < 2 or count > 10:
        findings.append(Finding(question.id, "score_level_count", Severity.ERROR, "score 需要 2 到 10 个有序等级"))
    for label, description in question.levels:
        if not description.strip() or description.strip() == label:
            findings.append(
                Finding(
                    question.id,
                    "missing_description",
                    Severity.WARNING,
                    f"等级 {label} 需要写成可区分的情形，而不是只重复等级名",
                )
            )
    findings.extend(_lint_wording(question.id, question.instructions))
    return findings


def _lint_noul(question: NoulSpec) -> list[Finding]:
    findings: list[Finding] = []
    if not question.proposition.strip():
        findings.append(Finding(question.id, "empty_proposition", Severity.ERROR, "noul 需要一条有真假标准的命题"))
    findings.extend(_lint_wording(question.id, question.proposition))
    return findings


def _lint_wording(question_id: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if any(marker in text for marker in _COMPOUND):
        findings.append(
            Finding(
                question_id,
                "compound_question",
                Severity.WARNING,
                "一题里写了多件可分开判断的事。拆成同一请求中的多个原子问题",
            )
        )
    folded = text.casefold()
    if any(marker.casefold() in folded for marker in _DEPENDENT):
        findings.append(
            Finding(
                question_id,
                "dependent_question",
                Severity.WARNING,
                "同一请求里的问题读不到彼此的答案。依赖上一步结果时要分两轮，或由程序组合边缘判断",
            )
        )
    return findings


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker.casefold() in text for marker in markers)
