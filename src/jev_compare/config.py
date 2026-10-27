"""实验配置：数据路径、本地 vLLM 模型、每个模型要跑的方法、并发数。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METHODS = ("generate", "jev")
_CONFIG_KEYS = {"data", "concurrency", "models", "output"}
_MODEL_KEYS = {"id", "base_url", "model", "api_key", "methods", "timeout_s", "max_tokens"}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    base_url: str
    model: str
    api_key: str
    methods: tuple[str, ...]
    timeout_s: float
    max_tokens: int


@dataclass(frozen=True)
class Experiment:
    data_path: Path
    concurrency: int
    models: tuple[ModelSpec, ...]
    output_dir: Path | None
    source: Path


def load_experiment(path: Path) -> Experiment:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"找不到配置文件: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置不是合法 JSON: {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("配置的顶层必须是对象")
    unknown = set(raw) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"配置含有未识别的字段: {', '.join(sorted(unknown))}")
    if "data" not in raw or "concurrency" not in raw or "models" not in raw:
        raise ValueError("配置必须包含 data、concurrency、models")

    data_path = _resolve_data(source, raw["data"])
    concurrency = raw["concurrency"]
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency 必须是大于等于 1 的整数")
    models_raw = raw["models"]
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("models 必须是非空数组")
    seen: set[str] = set()
    models: list[ModelSpec] = []
    for index, item in enumerate(models_raw):
        models.append(_model(item, index, seen))
    output = raw.get("output")
    output_dir = None if output is None else _resolve_output(source, output)
    return Experiment(data_path, concurrency, tuple(models), output_dir, source)


def _resolve_data(config_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("data 必须是数据文件路径")
    return _existing_path(config_path, value.strip(), "数据文件")


def _resolve_output(config_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("output 必须是目录路径")
    raw = Path(value.strip()).expanduser()
    if raw.is_absolute():
        return raw
    return Path.cwd() / raw


def _existing_path(config_path: Path, value: str, label: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [config_path.parent / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    looked = ", ".join(str(item) for item in candidates)
    raise ValueError(f"找不到{label}: {value}（已查找 {looked}）")


def _model(item: Any, index: int, seen: set[str]) -> ModelSpec:
    if not isinstance(item, dict):
        raise ValueError(f"models[{index}] 必须是对象")
    unknown = set(item) - _MODEL_KEYS
    if unknown:
        raise ValueError(f"models[{index}] 含有未识别的字段: {', '.join(sorted(unknown))}")
    for key in ("id", "base_url", "model", "methods"):
        if key not in item:
            raise ValueError(f"models[{index}] 缺少 {key}")
    model_id = item["id"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"models[{index}].id 必须是非空字符串")
    model_id = model_id.strip()
    if model_id in seen:
        raise ValueError(f"模型 id 重复: {model_id}")
    seen.add(model_id)
    base_url = item["base_url"]
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValueError(f"模型 {model_id} 的 base_url 必须是 http(s) 地址")
    served = item["model"]
    if not isinstance(served, str) or not served.strip():
        raise ValueError(f"模型 {model_id} 的 model 必须是 vLLM 上的模型名")
    methods = item["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError(f"模型 {model_id} 的 methods 必须是非空数组")
    if any(not isinstance(method, str) for method in methods):
        raise ValueError(f"模型 {model_id} 的 methods 只能包含字符串")
    if len(set(methods)) != len(methods):
        raise ValueError(f"模型 {model_id} 的 methods 有重复")
    unknown_methods = [method for method in methods if method not in METHODS]
    if unknown_methods:
        raise ValueError(f"模型 {model_id} 含有未知推理方法: {', '.join(unknown_methods)}")
    api_key = item.get("api_key", "EMPTY")
    if not isinstance(api_key, str):
        raise ValueError(f"模型 {model_id} 的 api_key 必须是字符串")
    timeout_s = item.get("timeout_s", 120)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError(f"模型 {model_id} 的 timeout_s 必须是正数")
    max_tokens = item.get("max_tokens", 256)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"模型 {model_id} 的 max_tokens 必须是大于等于 1 的整数")
    return ModelSpec(
        model_id,
        base_url.rstrip("/"),
        served.strip(),
        api_key,
        tuple(methods),
        float(timeout_s),
        max_tokens,
    )
