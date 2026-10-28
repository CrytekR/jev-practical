"""把生成文本里的一个 JSON 对象收成标签字符串。"""

from __future__ import annotations

import json
import math
from typing import Any


def canonicalize(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
        return format(value, "g")
    if isinstance(value, str):
        return value.strip()
    return None


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = _strip_fence(text.strip())
    start = stripped.find("{")
    if start < 0:
        return None, "回复里没有 JSON 对象"
    try:
        value, _end = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        return None, f"JSON 无法解析: {exc.msg}"
    if not isinstance(value, dict):
        return None, "回复里的 JSON 不是对象"
    return value, None


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()
