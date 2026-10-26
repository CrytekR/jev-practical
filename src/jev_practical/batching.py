"""同一份 state 上并行提问时，共享前缀省下的是重复读取，不是问题分支本身。

计数是单头因果注意力里的可见位置对，用来比较方案，不用来预测服务延迟。
"""

from __future__ import annotations

from typing import Literal


def causal_pairs(length: int) -> int:
    if length < 0:
        raise ValueError("length 不能为负")
    return length * (length + 1) // 2


def attention_pairs(
    *,
    state_tokens: int,
    branch_tokens: int,
    questions: int,
    shared_prefix: bool,
) -> int:
    if state_tokens < 0 or branch_tokens < 0:
        raise ValueError("token 数不能为负")
    if questions < 1:
        raise ValueError("questions 必须为正")
    branch = state_tokens * branch_tokens + causal_pairs(branch_tokens)
    state = causal_pairs(state_tokens)
    if shared_prefix:
        return state + questions * branch
    return questions * (state + branch)


def saved_pairs(*, state_tokens: int, questions: int) -> int:
    """共享前缀相对逐题重读 state，省下的可见位置对。"""

    if questions < 1:
        raise ValueError("questions 必须为正")
    return (questions - 1) * causal_pairs(state_tokens)


def readout_tokens(
    *,
    state_tokens: int,
    questions: int,
    options_per_question: int,
    option_tokens: int,
    mode: Literal["full", "letter"],
) -> int:
    """全文本读出按每个选项的 token 数计；单代号读出每个选项只计 1。

    state 只计一次。题干说明的 token 未计入，这是读出成本的下界比较。
    """

    if min(state_tokens, questions, options_per_question, option_tokens) < 0:
        raise ValueError("token 数不能为负")
    if questions < 1 or options_per_question < 1:
        raise ValueError("问题数和选项数必须为正")
    per_option = 1 if mode == "letter" else option_tokens
    if mode not in ("full", "letter"):
        raise ValueError(f"未知读出方式: {mode}")
    return state_tokens + questions * options_per_question * per_option
