"""Jev 读出用的 qev-labels-v2 后缀，以及 Qwen 关思考后的聊天包装。

包装与 mammoth-1 的聊天模板在 enable_thinking=False 时一致：系统、用户各一段，
助手回合以空的 think 块结束，停在答案位置。发给 vLLM 的 /completions 时不再套模板。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NOUL_TRUE = "the statement is true / the answer is yes"
NOUL_FALSE = "the statement is false / the answer is no"


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def state_block(state: Any) -> str:
    return f"<state>\n{render_state(state)}\n</state>\n\n"


def render_suffix(question: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """返回后缀文本，以及与字母 A、B、C… 同序的业务标签。"""

    kind = question.get("type")
    instructions = str(question.get("instructions") or "").strip()
    if kind == "noul":
        labels = ("yes", "no")
        header = "Question:"
        instructions = instructions or "Is the statement true, or is the answer to the question yes?"
        lines = [f"A. yes: {NOUL_TRUE}", f"B. no: {NOUL_FALSE}"]
    elif kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice 的 criteria 至少要有两个选项")
        labels = tuple(str(name) for name in criteria)
        descriptions = tuple(str(criteria[name]) for name in criteria)
        header = "Question:"
        instructions = instructions or "Which option applies to the state?"
        lines = [f"{LETTERS[i]}. {name}: {text}" for i, (name, text) in enumerate(zip(labels, descriptions, strict=True))]
    elif kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score 的 criteria 需要 2 到 10 个等级")
        labels = tuple(str(i) for i in range(len(criteria)))
        header = "Rate the state:"
        instructions = instructions or "Which level describes the state?"
        lines = [f"{LETTERS[i]}. level {i}: {level}" for i, level in enumerate(criteria)]
    else:
        raise ValueError(f"未知题型: {kind}")
    if len(labels) > len(LETTERS):
        raise ValueError("单字母槽位最多 26 个选项")
    if len(set(labels)) != len(labels):
        raise ValueError("选项标签有重复")
    letters = ", ".join(LETTERS[: len(labels)])
    suffix = f"{header} {instructions}\n\nOptions:\n" + "\n".join(lines) + f"\n\nAnswer with one letter: {letters}."
    return suffix, labels


def render_qwen_prompt(state: Any, suffix: str) -> str:
    system = SYSTEM_PROMPT.strip()
    user = (state_block(state) + suffix).strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
