import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_compare.config import load_experiment
from jev_compare.data import load_samples
from jev_compare.metrics import SampleResult, percentile, summarize
from jev_compare.parse import canonicalize, extract_json_object
from jev_compare.prompt import render_qwen_prompt, render_suffix

_EXAMPLE = Path(__file__).parents[1] / "examples" / "mammoth_multilabel.py"
_SPEC = importlib.util.spec_from_file_location("mammoth_multilabel", _EXAMPLE)
assert _SPEC is not None and _SPEC.loader is not None
mammoth = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mammoth
_SPEC.loader.exec_module(mammoth)


def test_suffix_matches_the_mammoth_example() -> None:
    for qid, question in mammoth.QUESTIONS.items():
        suffix, labels = render_suffix(question)
        rendered = mammoth.render_question(qid, question)
        assert suffix == rendered.suffix
        assert labels == rendered.labels


def test_qwen_prompt_stops_after_an_empty_think_block() -> None:
    suffix, _labels = render_suffix(mammoth.QUESTIONS["refund"])
    prompt = render_qwen_prompt(mammoth.TICKET, suffix)
    assert prompt.count("<state>") == 1
    assert "charged twice" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Answer with one letter: A, B." in prompt


def test_extracts_fenced_json_and_canonical_labels() -> None:
    parsed, error = extract_json_object('说明\n```json\n{"team": "billing", "urgency": 1, "refund": true}\n```')
    assert error is None
    assert parsed is not None
    assert canonicalize(parsed["team"]) == "billing"
    assert canonicalize(parsed["urgency"]) == "1"
    assert canonicalize(parsed["refund"]) == "true"
    missing, reason = extract_json_object("没有对象")
    assert missing is None
    assert reason == "回复里没有 JSON 对象"


def test_config_and_sample_round_trip(tmp_path: Path) -> None:
    data = tmp_path / "items.json"
    data.write_text(Path(__file__).parents[1].joinpath("data/support.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "data": "items.json",
                "concurrency": 4,
                "models": [
                    {
                        "id": "mammoth",
                        "base_url": "http://127.0.0.1:8001/v1",
                        "model": "baselabs/mammoth-1",
                        "methods": ["jev", "generate"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    experiment = load_experiment(config)
    assert experiment.concurrency == 4
    assert experiment.models[0].api_key == "EMPTY"
    assert experiment.models[0].methods == ("jev", "generate")
    samples = load_samples(experiment.data_path)
    assert samples[0].task == "support-routing"
    assert samples[0].gold["urgency"] == "1"
    assert samples[0].labels["team"] == ("billing", "technical", "fraud")
    assert samples[0].labels["urgency"] == ("0", "1", "2")


def test_rejects_unknown_method(tmp_path: Path) -> None:
    data = tmp_path / "items.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "data": "items.jsonl",
                "concurrency": 1,
                "models": [
                    {
                        "id": "qwen",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "model": "qwen",
                        "methods": ["sample"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未知推理方法"):
        load_experiment(config)


def test_vllm_script_uses_the_official_026_image() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run-vllm.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run", "baselabs/mammoth-1", "8001"],
        check=True,
        capture_output=True,
        text=True,
    )
    command = result.stdout
    assert "vllm/vllm-openai:v0.26.0" in command
    assert "--enable-prefix-caching" in command
    assert "--max-logprobs" in command and "26" in command
    assert "--api-key" in command and "EMPTY" in command
    assert "8001:8000" in command
    assert "baselabs/mammoth-1" in command
    refused = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert refused.returncode == 2


def test_summary_counts_wrong_invalid_and_percentiles() -> None:
    results = [
        SampleResult("a", "route", "m", "generate", 10, {"team": "billing"}, {"team": "billing"}),
        SampleResult("b", "route", "m", "generate", 20, {"team": "billing"}, {"team": "technical"}),
        SampleResult("c", "route", "m", "generate", 30, {"team": "billing"}, {}, question_errors={"team": "没有"}),
        SampleResult("d", "route", "m", "generate", 40, {"team": "billing"}, {"team": "billing"}),
    ]
    row = summarize(results)[0]
    assert row.question_accuracy == pytest.approx(0.5)
    assert row.invalid_rate == pytest.approx(0.25)
    assert row.wrong_rate == pytest.approx(0.25)
    assert row.sample_accuracy == pytest.approx(0.5)
    assert percentile([10, 20, 30, 40], 50) == 20
    assert row.latency_p50_ms == 20
    assert row.latency_p95_ms == 40
