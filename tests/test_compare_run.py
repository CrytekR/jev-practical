import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jev_compare.config import ModelSpec
from jev_compare.data import Sample, load_samples
from jev_compare.prompt import render_suffix
from jev_compare.run import run_experiment


class FakeClient:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.depth: dict[str, int] = {}
        self.overlapped = False

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        self._enter("generate")
        time.sleep(0.05)
        self._leave("generate")
        user = messages[-1]["content"]
        if "twice" in user:
            return '{"refund": "yes", "team": "billing", "urgency": 1}'
        return "无法作答"

    def slot_logprobs(self, prompt: str, letters: list[str]) -> dict[str, float]:
        state = prompt.split("<state>\n", 1)[1].split("\n</state>", 1)[0]
        with self.lock:
            depth = self.depth.get(state, 0) + 1
            self.depth[state] = depth
            if depth > 1:
                self.overlapped = True
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.04)
        with self.lock:
            self.depth[state] -= 1
            self.active -= 1
        return {letter: 0.0 if index == 0 else -1.0 for index, letter in enumerate(letters)}

    def _enter(self, _kind: str) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _leave(self, _kind: str) -> None:
        with self.lock:
            self.active -= 1


_ROOT = Path(__file__).parents[1]


def _sample(sample_id: str, text: str) -> dict[str, object]:
    source = json.loads((_ROOT / "data" / "support.jsonl").read_text(encoding="utf-8"))
    source["id"] = sample_id
    source["jev"]["state"] = text
    source["generate"]["messages"][1]["content"] = text
    return source


def test_concurrency_and_sequential_questions(tmp_path: Path) -> None:
    rows = [
        _sample("a", "alpha was charged twice"),
        _sample("b", "beta was charged twice"),
        _sample("c", "gamma lost a password"),
        _sample("d", "delta lost a password"),
    ]
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "data": "items.jsonl",
                "concurrency": 2,
                "output": str(tmp_path / "out"),
                "models": [
                    {
                        "id": "local",
                        "base_url": "http://127.0.0.1:9/v1",
                        "model": "fake",
                        "methods": ["generate", "jev"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from jev_compare.config import load_experiment

    experiment = load_experiment(config)
    samples = load_samples(experiment.data_path)
    client = FakeClient()
    summary = run_experiment(experiment, samples, tmp_path / "out", lambda _model: client)
    assert client.max_active == 2
    assert client.overlapped is False
    by_method = {row["method"]: row for row in summary["rows"]}
    assert by_method["generate"]["question_accuracy"] == 0.5
    assert by_method["generate"]["invalid_rate"] == 0.5
    assert by_method["jev"]["question_accuracy"] == pytest.approx(2 / 3)
    assert by_method["jev"]["wrong_rate"] == pytest.approx(1 / 3)
    assert by_method["jev"]["sample_accuracy"] == 0
    predictions = (tmp_path / "out" / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(predictions) == 8


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.server.captured.append((self.path, body))  # type: ignore[attr-defined]
        if self.path == "/tokenize":
            prompt = body["prompt"]
            if body.get("add_special_tokens") is not False:
                self._json(400, {"error": "special tokens"})
                return
            tokens = [ord(prompt)] if isinstance(prompt, str) and len(prompt) == 1 else [1, 2, 3]
            self._json(200, {"tokens": tokens, "count": len(tokens)})
            return
        if self.path == "/v1/completions":
            letters = ["A", "B", "C"][: len(body["logprob_token_ids"])]
            top = {letter: 0.0 if index == 0 else -2.0 for index, letter in enumerate(letters)}
            self._json(200, {"choices": [{"text": "A", "logprobs": {"top_logprobs": [top], "tokens": ["A"], "token_logprobs": [0.0]}}]})
            return
        if self.path == "/v1/chat/completions":
            content = '{"refund":"yes","team":"billing","urgency":"0"}'
            self._json(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})
            return
        self._json(404, {"error": self.path})

    def _json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def test_vllm_wire_format(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.captured = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        source = json.loads((_ROOT / "data" / "support.jsonl").read_text(encoding="utf-8"))
        source["gold"]["urgency"] = "0"
        data = tmp_path / "items.json"
        data.write_text(json.dumps([source]), encoding="utf-8")
        config = tmp_path / "experiment.json"
        config.write_text(
            json.dumps(
                {
                    "data": "items.json",
                    "concurrency": 1,
                    "models": [
                        {
                            "id": "served",
                            "base_url": f"http://127.0.0.1:{port}/v1",
                            "model": "baselabs/mammoth-1",
                            "api_key": "EMPTY",
                            "methods": ["generate", "jev"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        from jev_compare.config import load_experiment

        experiment = load_experiment(config)
        summary = run_experiment(experiment, load_samples(experiment.data_path), tmp_path / "out")
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()

    rows = {row["method"]: row for row in summary["rows"]}
    assert rows["generate"]["question_accuracy"] == 1
    assert rows["jev"]["question_accuracy"] == 1
    captured = server.captured  # type: ignore[attr-defined]
    completions = [body for path, body in captured if path == "/v1/completions"]
    assert len(completions) == 3
    assert completions[0]["prompt"] == [1, 2, 3]
    assert completions[0]["max_tokens"] == 1
    assert completions[0]["temperature"] == 0
    assert completions[0]["logprob_token_ids"]
    tokenizes = [body for path, body in captured if path == "/tokenize"]
    assert any(body.get("add_special_tokens") is False and body.get("prompt") == "A" for body in tokenizes)
    chats = [body for path, body in captured if path == "/v1/chat/completions"]
    assert chats[0]["messages"][1]["content"].startswith("I was charged twice")
    assert chats[0]["temperature"] == 0
    suffix, labels = render_suffix(source["jev"]["questions"]["team"])
    assert labels[0] == "billing"
    assert "A. billing:" in suffix


def test_model_spec_is_passed_to_factory(tmp_path: Path) -> None:
    seen: list[str] = []

    def factory(model: ModelSpec) -> FakeClient:
        seen.append(model.model)
        return FakeClient()

    sample = load_samples(_ROOT / "data" / "support.jsonl")[0]
    assert isinstance(sample, Sample)
    from jev_compare.config import Experiment

    experiment = Experiment(
        _ROOT / "data" / "support.jsonl",
        1,
        (ModelSpec("m", "http://127.0.0.1:9/v1", "served-name", "EMPTY", ("generate",), 5, 32),),
        None,
        Path("experiment.json"),
    )
    run_experiment(experiment, [sample], tmp_path / "out", factory)
    assert seen == ["served-name"]
