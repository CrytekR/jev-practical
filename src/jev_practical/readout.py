"""词重叠读出器：研究用的可替换后端，用来观察契约和表面词如何推动选择。

它统计状态与「选项名 + 描述」的词重叠，不是语言模型的 token 概率。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

from jev_practical.answers import answer_choice
from jev_practical.specs import ChoiceAnswer, ChoiceSpec

_ASCII = re.compile(r"[a-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> Counter[str]:
    folded = text.casefold()
    counts: Counter[str] = Counter(_ASCII.findall(folded))
    characters = _CJK.findall(folded)
    counts.update(characters)
    counts.update(left + right for left, right in zip(characters, characters[1:], strict=False))
    return counts


def overlap_logits(state: str, options: Mapping[str, str] | Sequence[tuple[str, str]]) -> list[float]:
    state_counts = tokenize(state)
    pairs = list(options.items()) if isinstance(options, Mapping) else list(options)
    logits: list[float] = []
    for label, description in pairs:
        option_counts = tokenize(f"{label} {description}")
        matched = sum(state_counts[token] for token in option_counts)
        logits.append(float(matched))
    return logits


def lexical_choice(spec: ChoiceSpec, state: str, *, temperature: float = 1.0) -> ChoiceAnswer:
    return answer_choice(spec, overlap_logits(state, spec.criteria), temperature=temperature)
