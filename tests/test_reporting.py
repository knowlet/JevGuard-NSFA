"""Regression tests for the benchmark reporting and report-comparison path.

Every test here runs with pytest alone: the Jev client and the Hugging Face row
iterator are monkeypatched at module level, so no network call, API key, dataset
download, or GPU is required.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from jevguard_nsfa import benchmark_jev
from jevguard_nsfa.compare import main_from_args as compare_main_from_args
from jevguard_nsfa.compare import render_markdown
from jevguard_nsfa.dataset import BenchmarkRow
from jevguard_nsfa.metrics import evaluate_guard_results, latency_summary, percentile
from jevguard_nsfa.models import Decision, GuardResult, Side

UNSAFE_DOMAIN = "prompt_injection_and_jailbreak"
EMPTY_SAMPLE: dict[str, None] = {"mean": None, "p50": None, "p95": None, "p99": None, "min": None, "max": None}


# --------------------------------------------------------------------------- #
# Stubs: replace the Jev client and the dataset iterator only.
# --------------------------------------------------------------------------- #


class _StubResponse:
    def __init__(self, scores: dict[str, float], *, model: str = "jev-stub", input_tokens: int | None = 11) -> None:
        self.nouls = {name: SimpleNamespace(noul=value) for name, value in scores.items()}
        self.model = model
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=None)


class _StubClient:
    """Async context manager with the one method AsyncJevGuard calls."""

    failing_texts: set[str] = set()

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "_StubClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aclose(self) -> None:
        return None

    async def system_one(self, *, state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> _StubResponse:
        text = state["untrusted_text"]
        if text in type(self).failing_texts:
            raise RuntimeError("stubbed provider failure")
        scores = {name: (0.9 if name == UNSAFE_DOMAIN else 0.05) for name in questions}
        return _StubResponse(scores)


def _stub_rows() -> list[BenchmarkRow]:
    return [
        BenchmarkRow(
            id="q-1",
            text="row-1",
            label=1,
            side=Side.QUERY,
            domains=(UNSAFE_DOMAIN,),
            lang="en",
        ),
        BenchmarkRow(id="q-2", text="row-2", label=0, side=Side.QUERY, domains=(), lang="en"),
        BenchmarkRow(id="q-3", text="row-3", label=1, side=Side.QUERY, domains=("resource_abuse",), lang="en"),
    ]


def _install_stub(monkeypatch: pytest.MonkeyPatch, rows: list[BenchmarkRow], failing_texts: set[str] | None = None) -> None:
    monkeypatch.setattr(benchmark_jev, "AsyncTypeSafeClient", _StubClient)
    monkeypatch.setattr(_StubClient, "failing_texts", set(failing_texts or set()))
    monkeypatch.setattr(benchmark_jev, "iter_huggingface_rows", lambda **kwargs: iter(rows))


def _jev_args(**overrides: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    benchmark_jev.add_arguments(parser)
    args = parser.parse_args([])
    args.output = None
    args.rpm = 60_000.0  # keeps the request limiter out of the way
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _run_jev(monkeypatch: pytest.MonkeyPatch, failing_texts: set[str] | None = None) -> dict[str, Any]:
    rows = _stub_rows()
    _install_stub(monkeypatch, rows, failing_texts)
    return asyncio.run(benchmark_jev.run_benchmark(_jev_args(limit=len(rows))))


def _digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# ISSUE 6: undefined metrics must be None/null, never bare NaN.
# --------------------------------------------------------------------------- #


def test_empty_latency_sample_reports_none_for_every_field() -> None:
    assert percentile([], 0.50) is None
    assert latency_summary([]) == EMPTY_SAMPLE


def test_positive_domain_accuracy_is_none_without_positive_rows() -> None:
    rows = [BenchmarkRow(id="safe-1", text="row", label=0, side=Side.QUERY, domains=(), lang="en")]
    results = [
        GuardResult(
            side=Side.QUERY,
            scores={UNSAFE_DOMAIN: 0.05},
            unsafe=False,
            decision=Decision.ALLOW,
            predicted_domain=UNSAFE_DOMAIN,
            max_risk=0.05,
            latency_ms=3.0,
            model="stub",
        )
    ]
    quality = evaluate_guard_results(rows, results)
    assert quality["positive_domain_accuracy"] is None


def test_defined_metrics_are_unchanged() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.50) == 2.0
    summary = latency_summary([10.0, 20.0])
    assert summary["mean"] == 15.0
    assert summary["min"] == 10.0
    assert summary["max"] == 20.0


def test_all_failing_jev_run_serializes_as_strict_json(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _stub_rows()
    report = _run_jev(monkeypatch, failing_texts={row.text for row in rows})

    serialized = json.dumps(report, allow_nan=False)  # raises ValueError if any NaN/Infinity remains
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert report["samples"]["successful"] == 0
    assert report["quality"] is None
    assert report["latency_ms"] == EMPTY_SAMPLE
    assert json.loads(serialized)["latency_ms"]["p95"] is None


def test_main_from_args_writes_strict_json_for_all_failing_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _stub_rows()
    _install_stub(monkeypatch, rows, failing_texts={row.text for row in rows})
    output = tmp_path / "jevguard.json"

    exit_code = benchmark_jev.main_from_args(_jev_args(limit=len(rows), output=output))

    assert exit_code == 0
    artifact = output.read_text(encoding="utf-8")
    assert "NaN" not in artifact
    assert "Infinity" not in artifact
    assert '"quality": null' in artifact
    json.dumps(json.loads(artifact), allow_nan=False)
    assert "NaN" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# ISSUE 5: report metadata that makes two runs alignable.
# --------------------------------------------------------------------------- #


def test_jev_report_records_selection_fingerprint_and_id_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _stub_rows()
    report = _run_jev(monkeypatch, failing_texts={"row-3"})

    assert report["dataset"]["fingerprint"] == _digest([f"{row.id}|{row.label}|{row.side.value}|{row.lang}" for row in rows])
    assert report["samples"]["attempted_ids_sha256"] == _digest([row.id for row in rows])
    assert report["samples"]["successful_ids_sha256"] == _digest([row.id for row in rows if row.id != "q-3"])
    assert report["latency_scope"] == "request"


def test_partial_failures_change_the_scored_sample_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    complete = _run_jev(monkeypatch)
    partial = _run_jev(monkeypatch, failing_texts={"row-3"})

    assert complete["samples"]["successful"] == 3
    assert complete["samples"]["successful_ids_sha256"] == complete["samples"]["attempted_ids_sha256"]
    assert partial["samples"]["successful"] == 2
    assert partial["samples"]["successful_ids_sha256"] != partial["samples"]["attempted_ids_sha256"]
    # Same attempted selection, so only the scored-subset digest moves.
    assert partial["dataset"]["fingerprint"] == complete["dataset"]["fingerprint"]
    assert partial["samples"]["attempted_ids_sha256"] == complete["samples"]["attempted_ids_sha256"]
    assert partial["quality"]["binary"]["f1"] != complete["quality"]["binary"]["f1"]


# --------------------------------------------------------------------------- #
# ISSUE 5: the comparator must verify alignment before emitting a delta.
# --------------------------------------------------------------------------- #


def _report(engine: str, **overrides: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "engine": engine,
        "mode": "managed-api-online" if engine == "jevguard-nsfa" else "local-realtime-classification",
        "model": "model-x",
        "dataset": {
            "name": "inclusionAI/NSFA_Benchmarks",
            "split": "train",
            "benchmark": "query",
            "languages": ["en"],
            "id_contains": None,
            "seed": 42,
            "fingerprint": "f" * 64,
        },
        "parameters": {"threshold": 0.5},
        "samples": {
            "attempted": 100,
            "successful": 100,
            "failed": 0,
            "attempted_ids_sha256": "a" * 64,
            "successful_ids_sha256": "a" * 64,
        },
        "latency_scope": "request",
        "quality": {"binary": {"f1": 0.9, "precision": 0.91, "recall": 0.89, "accuracy": 0.93, "brier": 0.08}},
        "latency_ms": {"p50": 120.0, "p95": 300.0, "p99": 400.0},
        "throughput": {"successful_requests_per_second": 5.0},
        "usage": {"cost_per_1000_successful_requests_usd": 0.01},
    }
    report.update(overrides)
    return report


def _aligned_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    jev = _report("jevguard-nsfa")
    singguard = _report(
        "singguard-nsfa",
        parameters={"threshold": 0.5, "batch_size": 1},
        quality={"binary": {"f1": 0.4, "precision": 0.4, "recall": 0.4, "accuracy": 0.6, "brier": 0.2}},
        latency_ms={"p50": 10.0, "p95": 20.0, "p99": 30.0},
    )
    return jev, singguard


def _row(markdown: str, label: str) -> list[str]:
    line = next(line for line in markdown.splitlines() if line.startswith(f"| {label} |"))
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def test_compare_reports_quality_and_latency_deltas_for_aligned_runs() -> None:
    """Identical data, both runs on ``latency_scope == "request"``, SingGuard batch size 1."""
    jev, singguard = _aligned_reports()
    markdown = render_markdown(jev, singguard)

    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "0.5000"]
    assert _row(markdown, "Latency p50 (ms)") == ["Latency p50 (ms)", "120.00", "10.00", "110.00"]
    assert "## Alignment" in markdown
    assert "All alignment checks are ok: quality deltas and latency deltas are reported." in markdown


def test_compare_keeps_per_engine_values_but_withholds_deltas_without_metadata() -> None:
    # Mirrors the minimal reports in tests/test_compare.py: no dataset/samples/parameters.
    markdown = render_markdown(
        {"model": "jev-test", "quality": {"binary": {"f1": 0.9}}},
        {"model": "sing-test", "quality": {"binary": {"f1": 0.8}}},
    )

    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.8000", "n/a"]
    assert "| dataset.fingerprint | missing | missing | unknown |" in markdown
    assert "quality deltas are withheld" in markdown


def test_compare_withholds_deltas_for_mismatched_runs() -> None:
    jev, singguard = _aligned_reports()
    singguard = copy.deepcopy(singguard)
    singguard["dataset"].update({"benchmark": "response", "languages": ["zh"], "seed": 7, "fingerprint": "b" * 64})
    singguard["parameters"]["threshold"] = 0.35
    singguard["quality"]["binary"]["f1"] = 0.1

    markdown = render_markdown(jev, singguard)

    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.1000", "n/a"]
    assert "| dataset.benchmark | query | response | mismatch |" in markdown
    assert "| dataset.seed | 42 | 7 | mismatch |" in markdown
    assert "| dataset.fingerprint |" in markdown and "| mismatch |" in markdown
    assert "not comparable" in markdown


def test_compare_withholds_deltas_when_scored_samples_differ() -> None:
    jev, singguard = _aligned_reports()
    singguard = copy.deepcopy(singguard)
    singguard["samples"]["successful"] = 99
    singguard["samples"]["successful_ids_sha256"] = "c" * 64

    markdown = render_markdown(jev, singguard)

    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    assert "| samples.successful_ids_sha256 |" in markdown
    assert f"| {'c' * 64} | mismatch |" in markdown


def test_compare_withholds_deltas_when_alignment_metadata_is_unknown() -> None:
    jev, singguard = _aligned_reports()
    jev = copy.deepcopy(jev)
    del jev["dataset"]["fingerprint"]

    markdown = render_markdown(jev, singguard)

    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    assert "| dataset.fingerprint | missing |" in markdown
    assert "unknown" in markdown


def test_compare_batched_singguard_keeps_quality_deltas_and_withholds_latency_deltas() -> None:
    """A throughput run with a large batch classifies the same samples, so quality stays comparable."""
    jev, singguard = _aligned_reports()
    singguard = copy.deepcopy(singguard)
    singguard["parameters"]["batch_size"] = 256

    markdown = render_markdown(jev, singguard)

    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "0.5000"]
    assert _row(markdown, "Latency p50 (ms)") == ["Latency p50 (ms)", "120.00", "10.00", "n/a"]
    assert "| parameters.batch_size (SingGuard) | n/a (not batched) | 256 | mismatch |" in markdown
    assert "Data alignment is ok: quality deltas are reported." in markdown
    latency_warning = next(line for line in markdown.splitlines() if "latency deltas are withheld" in line)
    assert latency_warning.startswith("WARNING: latency deltas are withheld")
    assert "SingGuard parameters.batch_size is 256" in latency_warning
    assert "these runs are not comparable" not in markdown


def test_compare_missing_latency_scope_only_withholds_latency_deltas() -> None:
    jev, singguard = _aligned_reports()
    jev = copy.deepcopy(jev)
    singguard = copy.deepcopy(singguard)
    del jev["latency_scope"]
    del singguard["latency_scope"]

    markdown = render_markdown(jev, singguard)

    assert "| latency_scope | missing | missing | unknown |" in markdown
    assert _row(markdown, "Latency p50 (ms)") == ["Latency p50 (ms)", "120.00", "10.00", "n/a"]
    # Quality deltas are still governed only by the data checks.
    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "0.5000"]
    latency_warning = next(line for line in markdown.splitlines() if "latency deltas are withheld" in line)
    assert "latency_scope is missing on JevGuard-NSFA and SingGuard-NSFA" in latency_warning


def test_compare_data_mismatch_withholds_quality_deltas_with_data_reason() -> None:
    jev, singguard = _aligned_reports()
    singguard = copy.deepcopy(singguard)
    singguard["dataset"]["fingerprint"] = "b" * 64

    markdown = render_markdown(jev, singguard)

    assert f"| {'b' * 64} | mismatch |" in markdown
    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    quality_warning = next(line for line in markdown.splitlines() if "quality deltas are withheld" in line)
    assert quality_warning.startswith("WARNING: these runs are not comparable")
    assert "dataset.fingerprint (mismatch)" in quality_warning


def test_compare_main_warns_before_markdown_on_mismatch(
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jev, singguard = _aligned_reports()
    singguard = copy.deepcopy(singguard)
    singguard["dataset"]["benchmark"] = "response"
    jev_path = tmp_path / "jev.json"
    sing_path = tmp_path / "singguard.json"
    jev_path.write_text(json.dumps(jev), encoding="utf-8")
    sing_path.write_text(json.dumps(singguard), encoding="utf-8")

    args = argparse.Namespace(jev=jev_path, singguard=sing_path, output=None)
    exit_code = compare_main_from_args(args)
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert stdout.splitlines()[0].startswith("WARNING: benchmark runs are not comparable")
    assert "dataset.benchmark" in stdout.splitlines()[0]
    assert stdout.splitlines()[1] == "# JevGuard-NSFA vs SingGuard-NSFA"


def test_compare_main_warns_about_latency_without_claiming_incomparable_runs(
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jev, singguard = _aligned_reports()
    singguard = copy.deepcopy(singguard)
    singguard["parameters"]["batch_size"] = 256
    jev_path = tmp_path / "jev.json"
    sing_path = tmp_path / "singguard.json"
    jev_path.write_text(json.dumps(jev), encoding="utf-8")
    sing_path.write_text(json.dumps(singguard), encoding="utf-8")

    args = argparse.Namespace(jev=jev_path, singguard=sing_path, output=None)
    exit_code = compare_main_from_args(args)
    stdout = capsys.readouterr().out

    assert exit_code == 0
    first_line = stdout.splitlines()[0]
    assert first_line.startswith("WARNING: latency deltas are withheld")
    assert "parameters.batch_size (SingGuard)" in first_line
    assert "not comparable" not in first_line
    assert stdout.splitlines()[1] == "# JevGuard-NSFA vs SingGuard-NSFA"


def test_compare_main_does_not_warn_for_aligned_runs(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    jev, singguard = _aligned_reports()
    jev_path = tmp_path / "jev.json"
    sing_path = tmp_path / "singguard.json"
    output = tmp_path / "comparison.md"
    jev_path.write_text(json.dumps(jev), encoding="utf-8")
    sing_path.write_text(json.dumps(singguard), encoding="utf-8")

    args = argparse.Namespace(jev=jev_path, singguard=sing_path, output=output)
    exit_code = compare_main_from_args(args)
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert stdout.startswith("# JevGuard-NSFA vs SingGuard-NSFA")
    # print() appends one newline to the markdown that is also written to the file.
    assert output.read_text(encoding="utf-8") + "\n" == stdout


def test_compare_treats_json_null_as_a_real_null_not_missing_metadata() -> None:
    jev, singguard = _aligned_reports()
    singguard = copy.deepcopy(singguard)
    singguard["dataset"]["id_contains"] = "AgentHarm"

    markdown = render_markdown(jev, singguard)

    assert "| dataset.id_contains | null | AgentHarm | mismatch |" in markdown
    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]


def test_compare_renders_json_null_metric_values_as_na() -> None:
    jev, singguard = _aligned_reports()
    jev = copy.deepcopy(jev)
    jev["latency_ms"] = {"p50": None, "p95": None, "p99": None}

    markdown = render_markdown(jev, singguard)

    assert _row(markdown, "Latency p50 (ms)") == ["Latency p50 (ms)", "n/a", "10.00", "n/a"]
