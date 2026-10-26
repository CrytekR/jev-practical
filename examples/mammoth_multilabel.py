"""用 transformers 看 baselabs/mammoth-1 的一次请求、多道题决策。

mammoth-1 不调用 generate。一次请求的过程是：

1. 系统提示和 <state> 组成共享前缀，只做一次 prefill。
2. 每道题有自己的后缀：题干、A/B/C 选项、回答行。后缀之间互不可见。
3. 带上前缀的 KV cache，只再前向这个后缀。最后一个位置的词表 logits 就是「下一个 token」的分布。
4. 只取出选项字母对应的 logits，在这些字母上做 softmax。没有解码，也没有 JSON 生成。

这和 OneJev 在提交 81ce62f 上的 DecisionEngine 相同，mammoth-1 的 setup.sh 把 serve.patch 打在那个版本上。
模型卡里的 readout 是 slot_lm：读的是语言模型头在字母上的 logits，不是额外的分类头。

官方服务还有一处权重相关的差异：config.json 里 full attention 层是 block_bidirectional。
前缀是一块，每道题的后缀是另一块；块内双向，前缀看不见任何后缀，线性注意力层仍然因果。
未打 serve.patch 的 transformers 前向仍是因果掩码。下面的调用顺序、读出位置和 fp32 语言模型头与引擎一致，
概率要和官方服务逐位相同，需要那份掩码补丁。27B 权重约 55GB，默认只打印提示词，不加载模型。

    python examples/mammoth_multilabel.py
    python examples/mammoth_multilabel.py --run --model baselabs/mammoth-1
"""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NOUL_TRUE = "the statement is true / the answer is yes"
NOUL_FALSE = "the statement is false / the answer is no"
PREFIX_MARK = "QEV_PREFIX_MARK"

# 模型卡里的三道题。两道是非题就是多标签：每道题单独为真或为假，不合成一个复合标签。
TICKET = "I was charged twice this month, please refund one of them."
QUESTIONS: dict[str, dict[str, Any]] = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "duplicate": {"type": "noul", "instructions": "Does the customer say the charge happened more than once?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
}


@dataclass(frozen=True)
class RenderedQuestion:
    qid: str
    kind: str
    labels: tuple[str, ...]
    descriptions: tuple[str, ...]
    suffix: str


def state_block(state: str) -> str:
    return f"<state>\n{state}\n</state>\n\n"


def render_question(qid: str, question: Mapping[str, Any]) -> RenderedQuestion:
    """把一道题写成后缀。字母只是槽位，业务标签在 labels 里，按同一顺序对应。"""

    kind = question["type"]
    instructions = str(question.get("instructions") or "").strip()
    if kind == "noul":
        labels = ("yes", "no")
        descriptions = (NOUL_TRUE, NOUL_FALSE)
        header = "Question:"
        instructions = instructions or "Is the statement true, or is the answer to the question yes?"
        lines = [f"A. yes: {NOUL_TRUE}", f"B. no: {NOUL_FALSE}"]
    elif kind == "choice":
        criteria = question["criteria"]
        labels = tuple(criteria)
        descriptions = tuple(str(criteria[name]) for name in labels)
        header = "Question:"
        instructions = instructions or "Which option applies to the state?"
        lines = [f"{LETTERS[i]}. {name}: {text}" for i, (name, text) in enumerate(zip(labels, descriptions, strict=True))]
    elif kind == "score":
        levels = tuple(str(level) for level in question["criteria"])
        labels = tuple(str(i) for i in range(len(levels)))
        descriptions = levels
        header = "Rate the state:"
        instructions = instructions or "Which level describes the state?"
        lines = [f"{LETTERS[i]}. level {i}: {text}" for i, text in enumerate(levels)]
    else:
        raise ValueError(f"未知题型: {kind}")
    if len(labels) > len(LETTERS):
        raise ValueError("这个示例只演示 26 个以内的单字母槽位")
    letters = ", ".join(LETTERS[: len(labels)])
    suffix = f"{header} {instructions}\n\nOptions:\n" + "\n".join(lines) + f"\n\nAnswer with one letter: {letters}."
    return RenderedQuestion(qid, kind, labels, descriptions, suffix)


def messages(state: str, suffix: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_block(state) + suffix},
    ]


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def choice_confidence(probabilities: Sequence[float]) -> float:
    """均匀分布映到 0，全部质量在一项上映到 1。与 (n * 峰值 - 1) / (n - 1) 相同。"""

    count = len(probabilities)
    if count < 2:
        return 1.0
    uniform = 1.0 / count
    return (max(probabilities) - uniform) / (1.0 - uniform)


def score_confidence(probabilities: Sequence[float]) -> float:
    """1 减去相对众数的概率加权绝对偏差，再用均匀分布的偏差归一。"""

    count = len(probabilities)
    if count < 2:
        return 1.0
    mode = max(range(count), key=probabilities.__getitem__)
    distance = sum(probability * abs(index - mode) for index, probability in enumerate(probabilities))
    center = (count - 1) / 2
    uniform_mad = sum(abs(index - center) for index in range(count)) / count
    return max(0.0, 1.0 - distance / uniform_mad)


def assemble(question: RenderedQuestion, slot_logits: Sequence[float]) -> dict[str, Any]:
    """字母上的 softmax 就是这道题的分布。noul 的概率是 yes 槽，不是词表里的绝对概率。"""

    if len(slot_logits) != len(question.labels):
        raise ValueError("槽位 logits 数量与选项数量不一致")
    probabilities = softmax(slot_logits)
    if question.kind == "noul":
        return {"type": "noul", "noul": probabilities[0]}
    if question.kind == "choice":
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": question.labels[best],
            "probabilities": dict(zip(question.labels, probabilities, strict=True)),
            "confidence": choice_confidence(probabilities),
        }
    expected = sum(index * probability for index, probability in enumerate(probabilities))
    return {
        "type": "score",
        "score": expected,
        "probabilities": {label: probability for label, probability in zip(question.labels, probabilities, strict=True)},
        "confidence": score_confidence(probabilities),
    }


def render_prompt(tokenizer: Any, state: str, suffix: str) -> str:
    return tokenizer.apply_chat_template(
        messages(state, suffix),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def shared_prefix_ids(tokenizer: Any, state: str, rendered: Sequence[RenderedQuestion]) -> tuple[list[int], list[list[int]]]:
    """先把每条分支整段编码，再取不超过 state 块结尾的最长公共 token 前缀。"""

    full = [tokenizer.encode(render_prompt(tokenizer, state, item.suffix), add_special_tokens=False) for item in rendered]
    marked = render_prompt(tokenizer, state, PREFIX_MARK)
    cut = marked.index(PREFIX_MARK)
    prefix = tokenizer.encode(marked[:cut], add_special_tokens=False)
    width = len(prefix)
    while width > 0 and not all(ids[:width] == prefix[:width] for ids in full):
        width -= 1
    prefix = prefix[:width]
    suffixes = [ids[width:] for ids in full]
    if any(len(suffix) == 0 for suffix in suffixes):
        raise ValueError("有一道题的后缀是空的，state 边界没有切对")
    return prefix, suffixes


def slot_token_ids(tokenizer: Any, count: int) -> list[int]:
    """每个字母必须来回都是一个 token。读出的是这个 token 的 logit，不是把单词 'billing' 再编码一遍。"""

    ids: list[int] = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"字母 {letter} 不是单个可往返的 token")
        ids.append(encoded[0])
    return ids


def _read_slots(vocab_logits: Any, slot_ids: Sequence[int]) -> tuple[list[float], float]:
    import torch

    values = vocab_logits.float()
    selected = values[list(slot_ids)]
    mass = float((torch.logsumexp(selected, -1) - torch.logsumexp(values, -1)).exp())
    return selected.tolist(), mass


def decide(model: Any, tokenizer: Any, state: str, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """共享前缀一次，随后每道题一次后缀前向。model 是已经 eval 的因果语言模型。"""

    import torch

    rendered = [render_question(qid, question) for qid, question in questions.items()]
    prefix_ids, suffix_ids = shared_prefix_ids(tokenizer, state, rendered)
    device = next(model.parameters()).device
    answers: dict[str, Any] = {}
    with torch.inference_mode():
        prefix = model(
            input_ids=torch.tensor([prefix_ids], device=device),
            attention_mask=torch.ones((1, len(prefix_ids)), dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = prefix.past_key_values
        del prefix
        for item, suffix in zip(rendered, suffix_ids, strict=True):
            branch_cache = copy.deepcopy(cache)
            out = model(
                input_ids=torch.tensor([suffix], device=device),
                attention_mask=torch.ones((1, len(prefix_ids) + len(suffix)), dtype=torch.long, device=device),
                past_key_values=branch_cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            # logits[0, -1] 是提示词最后一个 token 处对「下一个 token」的分布，答案字母就读这里。
            slot_logits, mass = _read_slots(out.logits[0, -1], slot_token_ids(tokenizer, len(item.labels)))
            answer = assemble(item, slot_logits)
            answer["allowed_token_mass"] = mass
            answers[item.qid] = answer
            del out, branch_cache
    return {
        "answers": answers,
        "prefix_tokens": len(prefix_ids),
        "suffix_tokens": {item.qid: len(suffix) for item, suffix in zip(rendered, suffix_ids, strict=True)},
    }


def explain(state: str, questions: Mapping[str, Mapping[str, Any]]) -> str:
    rendered = [render_question(qid, question) for qid, question in questions.items()]
    shared = state_block(state)
    lines = [
        "一次请求、四道题。不调用 generate，也不把四道题收成一个复合标签。",
        "1. 系统提示和 state 块编码成共享前缀，只做一次 prefill，留下 KV cache。",
        "2. 每道题把自己的后缀接到这份 cache 上，再做一次前向。后缀之间互不可见。",
        "3. 读最后一个位置的词表 logits，只留下这道题用到的字母。",
        "4. 在这些字母上 softmax。choice 取最早的最大项，score 取等级下标的期望，noul 取 yes 的概率。",
        "",
        "共享前缀只含系统提示和下面这份 state。前缀不包含任何题干。",
        "",
        shared.rstrip(),
        "",
        "每道题单独接一段后缀。一次后缀前向同时给这段里的全部字母打分。",
    ]
    for item in rendered:
        lines.extend(["", f"## {item.qid} ({item.kind})", item.suffix])
    lines.extend(
        [
            "",
            "聊天模板还会在助手回合放一个空的 think 块（enable_thinking=False），然后停在答案位置。",
            "那个位置不采样。softmax 的分母只包括这道题用到的字母。",
        ]
    )
    return "\n".join(lines)


def load_text_model(model_id: str, device: str) -> tuple[Any, Any]:
    """与 DecisionEngine 一样：mammoth-1 的文本决策走 Qwen3_5 文本配置，不走带视觉塔的生成类。"""

    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        config = config.get_text_config()
        model_cls = transformers.Qwen3_5ForCausalLM
    else:
        model_cls = transformers.AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_id, config=config, dtype=torch.bfloat16, device_map={"": device}, low_cpu_mem_usage=True
    )
    model.eval()
    # 官方引擎把语言模型头升到 fp32。bf16 的大 logits 会按 0.125 量化，两项容易被打成平局。
    _upcast_head(model)
    return model, tokenizer


def _upcast_head(model: Any) -> None:
    import torch

    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        return
    head.weight = torch.nn.Parameter(head.weight.detach().clone().to(torch.float32), requires_grad=False)
    if getattr(head, "bias", None) is not None:
        head.bias = torch.nn.Parameter(head.bias.detach().clone().to(torch.float32), requires_grad=False)
    head.register_forward_pre_hook(lambda module, args: (args[0].to(torch.float32),) + tuple(args[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="mammoth-1 的共享前缀多题读出示例")
    parser.add_argument("--run", action="store_true", help="加载权重并做前向。默认只打印提示词")
    parser.add_argument("--model", default="baselabs/mammoth-1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(explain(TICKET, QUESTIONS))
    if not args.run:
        return
    model, tokenizer = load_text_model(args.model, args.device)
    result = decide(model, tokenizer, TICKET, QUESTIONS)
    print()
    print(result)


if __name__ == "__main__":
    main()
