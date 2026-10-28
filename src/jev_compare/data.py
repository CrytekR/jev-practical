"""读取 JSON 数组或 JSONL。每条样本带任务名、金标，以及两种推理各自的输入。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_compare.parse import canonicalize
from jev_compare.prompt import render_suffix


@dataclass(frozen=True)
class Sample:
    id: str
    task: str
    gold: dict[str, str]
    messages: tuple[dict[str, str], ...]
    state: Any
    questions: dict[str, dict[str, Any]]
    labels: dict[str, tuple[str, ...]]


def load_samples(path: Path) -> tuple[Sample, ...]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"数据文件是空的: {path}")
    rows = _rows(path, text)
    if not rows:
        raise ValueError(f"数据文件里没有样本: {path}")
    seen: set[str] = set()
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        samples.append(_sample(row, index, seen))
    return tuple(samples)


def _rows(path: Path, text: str) -> list[Any]:
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"数据不是合法 JSON: {path}: {exc}") from exc
        if not isinstance(value, list):
            raise ValueError(f"数据文件的顶层必须是数组: {path}")
        return value
    rows: list[Any] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
    return rows


def _sample(row: Any, index: int, seen: set[str]) -> Sample:
    where = f"样本[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"{where} 必须是对象")
    for key in ("id", "task", "gold", "generate", "jev"):
        if key not in row:
            raise ValueError(f"{where} 缺少 {key}")
    sample_id = row["id"]
    if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
        raise ValueError(f"{where}.id 必须是字符串或整数")
    sample_id = str(sample_id).strip()
    if not sample_id:
        raise ValueError(f"{where}.id 不能为空")
    if sample_id in seen:
        raise ValueError(f"样本 id 重复: {sample_id}")
    seen.add(sample_id)
    task = row["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"样本 {sample_id} 的 task 必须是非空字符串")
    gold = _gold(row["gold"], sample_id)
    messages = _messages(row["generate"], sample_id)
    state, questions, labels = _jev(row["jev"], sample_id)
    return Sample(sample_id, task.strip(), gold, messages, state, questions, labels)


def _gold(value: Any, sample_id: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"样本 {sample_id} 的 gold 必须是非空对象")
    gold: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"样本 {sample_id} 的 gold 含有空键")
        label = canonicalize(item)
        if label is None or label == "":
            raise ValueError(f"样本 {sample_id} 的 gold.{key} 不能转换成标签")
        gold[key] = label
    return gold


def _messages(value: Any, sample_id: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, dict) or "messages" not in value:
        raise ValueError(f"样本 {sample_id} 的 generate.messages 缺失")
    messages = value["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"样本 {sample_id} 的 generate.messages 必须是非空数组")
    rendered: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"样本 {sample_id} 的 messages[{index}] 必须是对象")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"样本 {sample_id} 的 messages[{index}].role 必须是非空字符串")
        if not isinstance(content, str):
            raise ValueError(f"样本 {sample_id} 的 messages[{index}].content 必须是字符串")
        rendered.append({"role": role.strip(), "content": content})
    return tuple(rendered)


def _jev(value: Any, sample_id: str) -> tuple[Any, dict[str, dict[str, Any]], dict[str, tuple[str, ...]]]:
    if not isinstance(value, dict) or "state" not in value or "questions" not in value:
        raise ValueError(f"样本 {sample_id} 的 jev 必须包含 state 和 questions")
    state = value["state"]
    if state is None or (isinstance(state, str) and not state.strip()):
        raise ValueError(f"样本 {sample_id} 的 jev.state 不能为空")
    if not isinstance(state, (str, dict, list)):
        raise ValueError(f"样本 {sample_id} 的 jev.state 必须是字符串、对象或数组")
    questions = value["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"样本 {sample_id} 的 jev.questions 必须是非空对象")
    labels: dict[str, tuple[str, ...]] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"样本 {sample_id} 含有空的问题 id")
        if not isinstance(question, dict):
            raise ValueError(f"样本 {sample_id} 的问题 {qid} 必须是对象")
        try:
            _suffix, slot_labels = render_suffix(question)
        except ValueError as exc:
            raise ValueError(f"样本 {sample_id} 的问题 {qid}: {exc}") from exc
        normalized[qid] = question
        labels[qid] = slot_labels
    return state, normalized, labels
