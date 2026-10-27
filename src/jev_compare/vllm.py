"""本地 vLLM OpenAI 兼容接口。生成走 chat，Jev 走 completions 的指定 token 对数概率。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from jev_compare.prompt import LETTERS


class VllmClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._single_tokens: dict[str, int] = {}

    def warm_letters(self) -> None:
        for letter in LETTERS:
            self.token_id(letter)

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        body = self._post(f"{self.base_url}/chat/completions", payload)
        return _message_text(body)

    def slot_logprobs(self, prompt: str, letters: list[str]) -> dict[str, float]:
        prompt_ids = self._tokenize(prompt)
        if not prompt_ids:
            raise RuntimeError("提示词被 tokenize 成空序列")
        slot_ids = [self.token_id(letter) for letter in letters]
        payload = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": len(letters),
            "logprob_token_ids": slot_ids,
        }
        body = self._post(f"{self.base_url}/completions", payload)
        return _slot_logprobs(body, letters)

    def token_id(self, letter: str) -> int:
        cached = self._single_tokens.get(letter)
        if cached is not None:
            return cached
        ids = self._tokenize(letter)
        if len(ids) != 1:
            raise RuntimeError(f"字母 {letter} 在服务端不是单个 token，得到 {ids}")
        self._single_tokens[letter] = ids[0]
        return ids[0]

    def _tokenize(self, text: str) -> list[int]:
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        body = self._post(
            f"{root}/tokenize",
            {"model": self.model, "prompt": text, "add_special_tokens": False},
        )
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in tokens):
            raise RuntimeError(f"tokenize 响应缺少整数 tokens: {body}")
        return list(tokens)

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 {url}: {exc.reason}") from exc
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{url} 返回的不是 JSON") from exc
        if not isinstance(body, dict):
            raise RuntimeError(f"{url} 返回的 JSON 不是对象")
        return body


def _message_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("chat 响应里没有 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("chat 响应里没有 message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise RuntimeError("chat 响应里没有文本内容")


def _slot_logprobs(body: dict[str, Any], letters: list[str]) -> dict[str, float]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("completions 响应里没有 choices")
    found = _collect_logprobs(choices[0].get("logprobs"))
    resolved: dict[str, float] = {}
    missing: list[str] = []
    for letter in letters:
        if letter in found:
            resolved[letter] = found[letter]
        elif f" {letter}" in found:
            resolved[letter] = found[f" {letter}"]
        else:
            missing.append(letter)
    if missing:
        raise RuntimeError(f"响应里没有字母 {', '.join(missing)} 的对数概率")
    return resolved


def _collect_logprobs(logprobs: Any) -> dict[str, float]:
    found: dict[str, float] = {}
    if not isinstance(logprobs, dict):
        return found
    top = logprobs.get("top_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], dict):
        for key, value in top[0].items():
            number = _logprob_value(value)
            if number is not None:
                found[str(key)] = number
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(tokens, list) and isinstance(token_logprobs, list) and tokens and token_logprobs:
        number = _logprob_value(token_logprobs[0])
        if number is not None and isinstance(tokens[0], str):
            found.setdefault(tokens[0], number)
    content = logprobs.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        for entry in content[0].get("top_logprobs") or []:
            if isinstance(entry, dict) and isinstance(entry.get("token"), str):
                number = _logprob_value(entry.get("logprob"))
                if number is not None:
                    found[entry["token"]] = number
    return found


def _logprob_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return _logprob_value(value.get("logprob"))
    return None
