"""按配置把每个模型的每种方法跑完，写出逐条记录和汇总。"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jev_compare.config import Experiment, ModelSpec, load_experiment
from jev_compare.data import Sample, load_samples
from jev_compare.methods import Client, predict_generate, predict_jev
from jev_compare.metrics import SampleResult, render_markdown, summarize
from jev_compare.vllm import VllmClient

ClientFactory = Callable[[ModelSpec], Client]


def default_client(model: ModelSpec) -> VllmClient:
    return VllmClient(model.base_url, model.model, model.api_key, model.timeout_s)


def run_experiment(
    experiment: Experiment,
    samples: Sequence[Sample],
    output_dir: Path,
    client_factory: ClientFactory = default_client,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SampleResult] = []
    for model in experiment.models:
        client = client_factory(model)
        for method in model.methods:
            results.extend(_run_method(experiment, model, method, samples, client))
    rows = summarize(results)
    summary = {
        "data": str(experiment.data_path),
        "concurrency": experiment.concurrency,
        "rows": [row.as_dict() for row in rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(rows, experiment.concurrency)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(_prediction(result), ensure_ascii=False) + "\n")
    return summary


def _run_method(
    experiment: Experiment,
    model: ModelSpec,
    method: str,
    samples: Sequence[Sample],
    client: Client,
) -> list[SampleResult]:
    if method == "jev" and isinstance(client, VllmClient):
        try:
            client.warm_letters()
        except Exception as exc:
            message = _short(exc)
            return [_failed(sample, model.id, method, message) for sample in samples]
    ordered: list[SampleResult | None] = [None] * len(samples)

    def job(index: int) -> None:
        ordered[index] = _one(model, method, samples[index], client)

    workers = min(experiment.concurrency, len(samples)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(job, range(len(samples))))
    return [item for item in ordered if item is not None]


def _one(model: ModelSpec, method: str, sample: Sample, client: Client) -> SampleResult:
    started = time.perf_counter()
    try:
        if method == "generate":
            prediction = predict_generate(client, sample, model.max_tokens)
        elif method == "jev":
            prediction = predict_jev(client, sample)
        else:
            raise ValueError(f"未知推理方法: {method}")
        answers = prediction.answers
        probabilities = prediction.probabilities
        question_errors = prediction.question_errors
        error = prediction.error
        raw = prediction.raw
    except Exception as exc:
        answers = {}
        probabilities = {}
        question_errors = {}
        error = _short(exc)
        raw = None
    latency_ms = (time.perf_counter() - started) * 1000
    return SampleResult(
        sample.id,
        sample.task,
        model.id,
        method,
        latency_ms,
        sample.gold,
        answers,
        probabilities,
        question_errors,
        error,
        raw,
    )


def _failed(sample: Sample, model_id: str, method: str, message: str) -> SampleResult:
    return SampleResult(sample.id, sample.task, model_id, method, 0.0, sample.gold, {}, {}, {}, message, None)


def _prediction(result: SampleResult) -> dict[str, object]:
    correct, invalid, wrong = result.counts
    return {
        "id": result.sample_id,
        "task": result.task,
        "model": result.model_id,
        "method": result.method,
        "latency_ms": result.latency_ms,
        "gold": result.gold,
        "answers": result.answers,
        "probabilities": result.probabilities,
        "question_errors": result.question_errors,
        "error": result.error,
        "correct": correct,
        "invalid": invalid,
        "wrong": wrong,
    }


def _short(exc: Exception) -> str:
    text = str(exc).strip() or "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return text[:500]


def _default_output(experiment: Experiment) -> Path:
    if experiment.output_dir is not None:
        return experiment.output_dir
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path("runs") / f"{experiment.source.stem}-{stamp}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="用本地 vLLM 对比文本生成和 Jev 读出")
    parser.add_argument("--config", required=True, type=Path, help="实验 JSON 配置")
    args = parser.parse_args(argv)
    try:
        experiment = load_experiment(args.config)
        samples = load_samples(experiment.data_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = _default_output(experiment)
    run_experiment(experiment, samples, output_dir)
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"结果目录: {output_dir.resolve()}")
