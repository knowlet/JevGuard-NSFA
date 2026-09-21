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
import sys
import types
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from jevguard_nsfa import benchmark_jev, benchmark_singguard, cli
from jevguard_nsfa.compare import main_from_args as compare_main_from_args
from jevguard_nsfa.compare import render_markdown
from jevguard_nsfa.dataset import BenchmarkRow, iter_huggingface_rows
from jevguard_nsfa.metrics import evaluate_guard_results, latency_summary, percentile
from jevguard_nsfa.models import Decision, GuardResult, Side
from jevguard_nsfa.taxonomy import domains_for

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
# Stubs for the runner-managed retry loop and the SDK model resolution.
# --------------------------------------------------------------------------- #


def _guard_result(side: Side, *, score: float = 0.9) -> GuardResult:
    # Every domain of the side is scored: metrics.evaluate_guard_results rejects a partial score
    # set, because an absent domain is an unmeasured one rather than a 0.0 probability.
    scores = {domain.id: 0.05 for domain in domains_for(side)}
    scores[UNSAFE_DOMAIN] = score
    unsafe = score > 0.5
    return GuardResult(
        side=side,
        scores=scores,
        unsafe=unsafe,
        decision=Decision.BLOCK if unsafe else Decision.ALLOW,
        predicted_domain=UNSAFE_DOMAIN,
        max_risk=score,
        latency_ms=1.0,
        model="guard-stub-model",
        input_tokens=5,
    )


class _GuardRecorder:
    """Call log of the stub guard installed by :func:`_install_guard_stub`."""

    def __init__(self) -> None:
        self.screen_calls: list[str] = []
        self.fallback_calls: list[str] = []


def _install_guard_stub(monkeypatch: pytest.MonkeyPatch, errors: list[BaseException | None]) -> _GuardRecorder:
    """Install a guard stub that raises ``errors[n]`` on screening attempt ``n``.

    Once the script is exhausted every attempt succeeds. ``screen_or_fallback`` is recorded and
    fails the test loudly: an operational Jev failure must never fall through to System Two.
    """
    recorder = _GuardRecorder()
    remaining = list(errors)

    class StubGuard:
        def __init__(self, *, policy: Any = None, client: Any = None) -> None:
            self.policy = policy
            self.client = client

        async def screen(self, text: str, side: Side, *, timeout: float | None = None) -> GuardResult:
            recorder.screen_calls.append(text)
            error = remaining.pop(0) if remaining else None
            if error is not None:
                raise error
            return _guard_result(side)

        async def screen_or_fallback(self, *args: Any, **kwargs: Any) -> Any:
            recorder.fallback_calls.append(str(args[0] if args else ""))
            raise AssertionError("the System-Two fallback must not run for an operational failure")

    monkeypatch.setattr(benchmark_jev, "AsyncJevGuard", StubGuard)
    return recorder


def _install_counting_limiter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the limiter with a non-sleeping counter; returns its shared counter."""
    counter = [0]

    class CountingLimiter:
        def __init__(self, rpm: float) -> None:
            self.rpm = rpm

        async def acquire(self) -> None:
            counter[0] += 1

    monkeypatch.setattr(benchmark_jev, "RequestStartLimiter", CountingLimiter)
    return counter


def _install_sleep_recorder(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the inter-attempt sleep with a recorder so no test ever waits."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(benchmark_jev, "_sleep", fake_sleep)
    return waits


class _StubTransport(httpx2.AsyncBaseTransport):
    """Serve every HTTP attempt locally so a real SDK client never touches the network."""

    def __init__(
        self,
        *,
        failures: int = 0,
        status: int = 500,
        headers: dict[str, str] | None = None,
        model: str = "jev-stub-model",
    ) -> None:
        self.failures = failures
        self.status = status
        self.headers = headers or {}
        self.model = model
        self.attempts = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.attempts += 1
        if self.attempts <= self.failures:
            return httpx2.Response(self.status, headers=self.headers, json={"error": "stub failure"})
        requested = json.loads(request.content)["questions"]
        return httpx2.Response(
            200,
            json={
                "model": self.model,
                "usage": {"input_tokens": 3, "output_tokens": 1},
                "answers": {name: {"type": "noul", "noul": 0.1} for name in requested},
            },
        )


class _RecordingClientFactory:
    """Build the real SDK client over a stub transport and record its constructor arguments."""

    def __init__(self, transport: _StubTransport) -> None:
        self.transport = transport
        self.kwargs: dict[str, Any] = {}
        self.clients: list[AsyncTypeSafeClient] = []

    def __call__(self, **kwargs: Any) -> AsyncTypeSafeClient:
        self.kwargs = kwargs
        client = AsyncTypeSafeClient(transport=self.transport, **kwargs)
        self.clients.append(client)
        return client


def _install_real_client_factory(
    monkeypatch: pytest.MonkeyPatch,
    transport: _StubTransport,
    rows: list[BenchmarkRow],
) -> _RecordingClientFactory:
    """Route the runner through the real SDK client and a one-row stub selection."""
    factory = _RecordingClientFactory(transport)
    monkeypatch.setattr(benchmark_jev, "AsyncTypeSafeClient", factory)
    monkeypatch.setattr(benchmark_jev, "iter_huggingface_rows", lambda **kwargs: iter(rows))
    return factory


def _server_error(status: int = 500, headers: dict[str, str] | None = None) -> TypeSafeInternalServerError:
    return TypeSafeInternalServerError(status, {"error": "stubbed overload"}, httpx2.Headers(headers or {}))


def _run_jev_with_guard_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    errors: list[BaseException | None] | None = None,
    rows: list[BenchmarkRow] | None = None,
    **overrides: Any,
) -> tuple[dict[str, Any], _GuardRecorder, list[int], list[float]]:
    """Run the Jev benchmark against a scripted guard stub, a counting limiter and a fake sleep."""
    selected = rows if rows is not None else _stub_rows()
    recorder = _install_guard_stub(monkeypatch, errors or [])
    monkeypatch.setattr(benchmark_jev, "AsyncTypeSafeClient", _StubClient)
    monkeypatch.setattr(benchmark_jev, "iter_huggingface_rows", lambda **kwargs: iter(selected))
    acquires = _install_counting_limiter(monkeypatch)
    waits = _install_sleep_recorder(monkeypatch)
    report = asyncio.run(benchmark_jev.run_benchmark(_jev_args(limit=len(selected), **overrides)))
    return report, recorder, acquires, waits


# --------------------------------------------------------------------------- #
# ISSUE 6: undefined metrics must be None/null, never bare NaN.
# --------------------------------------------------------------------------- #


def test_empty_latency_sample_reports_none_for_every_field() -> None:
    assert percentile([], 0.50) is None
    assert latency_summary([]) == EMPTY_SAMPLE


def test_positive_domain_accuracy_is_none_without_positive_rows() -> None:
    rows = [BenchmarkRow(id="safe-1", text="row", label=0, side=Side.QUERY, domains=(), lang="en")]
    results = [_guard_result(Side.QUERY, score=0.05)]
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

    assert report["dataset"]["fingerprint"] == benchmark_jev._rows_fingerprint(rows)
    assert report["samples"]["attempted_ids_sha256"] == _digest([row.id for row in rows])
    assert report["samples"]["successful_sha256"] == benchmark_jev._rows_fingerprint(
        [row for row in rows if row.id != "q-3"]
    )
    # The id-only digest stays a diagnostic: the scored subset is identified by content.
    assert "successful_ids_sha256" not in report["samples"]
    assert report["dataset"]["revision"] is None
    # head_manifest is a SingGuard-only key: the managed API has no local head set.
    assert "head_manifest" not in report
    assert report["latency_scope"] == "request"


def test_partial_failures_change_the_scored_sample_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    complete = _run_jev(monkeypatch)
    partial = _run_jev(monkeypatch, failing_texts={"row-3"})

    assert complete["samples"]["successful"] == 3
    assert complete["samples"]["successful_sha256"] == complete["dataset"]["fingerprint"]
    assert partial["samples"]["successful"] == 2
    assert partial["samples"]["successful_sha256"] != partial["dataset"]["fingerprint"]
    # Same attempted selection, so only the scored-subset fingerprint moves.
    assert partial["dataset"]["fingerprint"] == complete["dataset"]["fingerprint"]
    assert partial["samples"]["attempted_ids_sha256"] == complete["samples"]["attempted_ids_sha256"]
    assert partial["quality"]["binary"]["f1"] != complete["quality"]["binary"]["f1"]


def test_dataset_revision_is_forwarded_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _stub_rows()
    captured: dict[str, Any] = {}

    def fake_rows(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return iter(rows)

    _install_stub(monkeypatch, rows)
    monkeypatch.setattr(benchmark_jev, "iter_huggingface_rows", fake_rows)

    pinned = asyncio.run(
        benchmark_jev.run_benchmark(_jev_args(limit=len(rows), dataset_revision="v1.2.3"))
    )
    assert captured["revision"] == "v1.2.3"
    assert pinned["dataset"]["revision"] == "v1.2.3"

    default = asyncio.run(benchmark_jev.run_benchmark(_jev_args(limit=len(rows))))
    assert captured["revision"] is None
    assert default["dataset"]["revision"] is None


def test_dataset_loader_forwards_a_revision_only_when_it_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    fake_datasets = types.ModuleType("datasets")

    def load_dataset(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    fake_datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    assert list(iter_huggingface_rows(revision="abc123")) == []
    assert calls[0]["revision"] == "abc123"

    assert list(iter_huggingface_rows()) == []
    assert "revision" not in calls[1]  # the hub default must not be spelled as a fake revision


def test_benchmark_parquet_uri_carries_the_requested_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fake_datasets = types.ModuleType("datasets")

    def load_dataset(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append((args, kwargs))
        return []

    fake_datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    list(iter_huggingface_rows(benchmark="query", revision="abc123"))
    assert calls[0][0] == ("parquet",)
    assert calls[0][1]["data_files"]["train"] == (
        "hf://datasets/inclusionAI/NSFA_Benchmarks@abc123/NSFA_Query_Multilingual.parquet"
    )
    assert calls[0][1]["split"] == "train"
    assert "revision" not in calls[0][1]


def test_negative_retries_are_rejected() -> None:
    with pytest.raises(ValueError, match="--retries must be non-negative"):
        benchmark_jev.main_from_args(_jev_args(retries=-1, output=None))


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
            # The revision actually requested; null means the hub default was used.
            "revision": None,
            "fingerprint": "f" * 64,
        },
        "parameters": {"threshold": 0.5},
        "samples": {
            "attempted": 100,
            "successful": 100,
            "failed": 0,
            # Id-only diagnostic; the comparator must gate on successful_sha256 instead.
            "attempted_ids_sha256": "a" * 64,
            "successful_sha256": "a" * 64,
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
        # SingGuard is local: it must prove its classification heads covered the taxonomy.
        head_manifest={
            "side": "query",
            "expected_domains": [UNSAFE_DOMAIN],
            "loaded_domains": [UNSAFE_DOMAIN],
            "missing_domains": [],
            "unexpected_domains": [],
            "complete": True,
            "head_count": 1,
        },
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
    singguard["samples"]["successful_sha256"] = "c" * 64

    markdown = render_markdown(jev, singguard)

    assert _row(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    assert "| samples.successful_sha256 |" in markdown
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


# --------------------------------------------------------------------------- #
# ISSUE 3: SDK-internal retries must not bypass the request limiter.
# --------------------------------------------------------------------------- #


def test_real_sdk_client_retries_are_disabled_and_the_limiter_sees_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 500 must be retried by the runner, not inside the SDK, so the limiter counts it."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    transport = _StubTransport(failures=1)
    factory = _install_real_client_factory(monkeypatch, transport, _stub_rows()[:1])
    acquires = _install_counting_limiter(monkeypatch)
    waits = _install_sleep_recorder(monkeypatch)

    report = asyncio.run(benchmark_jev.run_benchmark(_jev_args(limit=1, retries=1)))

    assert factory.kwargs["retry"].max_retries == 0  # the SDK never retries on its own
    assert transport.attempts == 2  # one 500, then the retry that the runner made
    assert acquires[0] == 2  # ... and the limiter saw both attempts
    assert len(waits) == 1
    assert report["samples"]["request_attempts"] == 2
    assert report["samples"]["retried_requests"] == 1
    assert report["samples"]["successful"] == 1
    assert report["samples"]["failed"] == 0


def test_retryable_failure_is_retried_and_the_limiter_sees_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _stub_rows()[:1]
    report, recorder, acquires, waits = _run_jev_with_guard_stub(
        monkeypatch,
        errors=[_server_error(500), None],
        rows=rows,
        retries=1,
    )

    assert recorder.screen_calls == ["row-1", "row-1"]
    assert acquires[0] == 2  # one limiter acquire per real attempt
    assert len(waits) == 1
    assert 0.0 < waits[0] <= 0.5  # jittered exponential backoff, and never a real sleep
    assert report["samples"]["successful"] == 1
    assert report["samples"]["failed"] == 0
    assert report["samples"]["request_attempts"] == 2
    assert report["samples"]["retried_requests"] == 1
    assert report["samples"]["successful_sha256"] == benchmark_jev._rows_fingerprint(rows)
    assert recorder.fallback_calls == []


def test_non_retryable_failure_fails_the_row_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    error = TypeSafeAuthenticationError(401, {"error": "nope"}, httpx2.Headers())
    report, recorder, acquires, waits = _run_jev_with_guard_stub(
        monkeypatch,
        errors=[error, None],
        rows=_stub_rows()[:1],
        retries=5,
    )

    assert recorder.screen_calls == ["row-1"]  # a 401 is not worth a second attempt
    assert acquires[0] == 1
    assert waits == []
    assert report["samples"]["successful"] == 0
    assert report["samples"]["failed"] == 1
    assert report["samples"]["request_attempts"] == 1
    assert report["samples"]["retried_requests"] == 0
    assert report["failures"][0]["id"] == "q-1"
    assert report["failures"][0]["error"].startswith("TypeSafeAuthenticationError")
    assert report["failures"][0]["attempts"] == 1


def test_exhausted_retries_are_recorded_as_a_failure_never_as_a_safe_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, recorder, acquires, waits = _run_jev_with_guard_stub(
        monkeypatch,
        errors=[_server_error(500), _server_error(503), _server_error(502)],
        rows=_stub_rows()[:1],
        retries=2,
    )

    assert len(recorder.screen_calls) == 3  # --retries 2 means at most three attempts
    assert acquires[0] == 3
    assert len(waits) == 2
    assert report["samples"]["attempted"] == 1
    assert report["samples"]["successful"] == 0
    assert report["samples"]["failed"] == 1
    assert report["samples"]["request_attempts"] == 3
    assert report["samples"]["retried_requests"] == 1
    # No row produced a scored result, so nothing is scored and nothing is claimed safe.
    assert report["samples"]["successful_sha256"] == benchmark_jev._rows_fingerprint([])
    assert report["quality"] is None
    assert report["failures"][0]["attempts"] == 3
    assert recorder.fallback_calls == []


def test_retry_after_from_the_sdk_error_sets_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    rate_limited = TypeSafeRateLimitError(429, {"error": "slow down"}, httpx2.Headers({"retry-after-ms": "2500"}))
    assert rate_limited.retry_after_ms == 2500.0

    report, recorder, acquires, waits = _run_jev_with_guard_stub(
        monkeypatch,
        errors=[rate_limited, None],
        rows=_stub_rows()[:1],
        retries=1,
        timeout=60.0,
    )

    assert waits == [2.5]  # straight from the error's retry_after_ms, not from backoff
    assert acquires[0] == 2
    assert report["samples"]["request_attempts"] == 2
    assert report["samples"]["successful"] == 1
    assert recorder.fallback_calls == []


def test_retry_wait_reads_retry_after_headers_and_otherwise_backs_off() -> None:
    header_only = TypeSafeInternalServerError(500, {}, httpx2.Headers({"retry-after": "3"}))
    assert benchmark_jev.retry_wait_seconds(1, header_only) == 3.0

    millisecond_header = TypeSafeAPIError(503, {}, httpx2.Headers({"retry-after-ms": "1500"}))
    assert benchmark_jev.retry_wait_seconds(1, millisecond_header) == 1.5

    # No Retry-After: exponential backoff from 0.5s with up to 25% jitter, capped at 8s.
    assert 0.375 <= benchmark_jev.retry_wait_seconds(1, _server_error(500)) <= 0.5
    assert 6.0 <= benchmark_jev.retry_wait_seconds(6, _server_error(500)) <= 8.0
    assert 6.0 <= benchmark_jev.retry_wait_seconds(30, _server_error(500)) <= 8.0

    # A server-requested wait is authoritative; the row budget decides whether it can be used.
    long_wait = TypeSafeRateLimitError(429, {}, httpx2.Headers({"retry-after": "120"}))
    assert benchmark_jev.retry_wait_seconds(1, long_wait) == 120.0


def test_retryable_classification_matches_the_sdk_default_policy() -> None:
    policy = RetryPolicy()  # SDK defaults: 408, 429, 500-599 plus connection/timeout errors
    candidates: list[BaseException] = [
        TypeSafeInternalServerError(500, {}, httpx2.Headers()),
        TypeSafeAPIError(408, {}, httpx2.Headers()),
        TypeSafeRateLimitError(429, {}, httpx2.Headers()),
        TypeSafeAPIError(599, {}, httpx2.Headers()),
        TypeSafeAPIError(600, {}, httpx2.Headers()),
        TypeSafeAuthenticationError(401, {}, httpx2.Headers()),
        TypeSafeAPIError(400, {}, httpx2.Headers()),
        TypeSafeAPIError(404, {}, httpx2.Headers()),
        TypeSafeAPITimeoutError(15.0),
        ValueError("malformed response"),
    ]
    for error in candidates:
        # Explicit parity check against the SDK seam the runner replaces.
        assert benchmark_jev._is_retryable(error) is policy._retryable(error), error  # noqa: SLF001
    assert benchmark_jev._is_retryable(TypeSafeAPITimeoutError(15.0))
    assert not benchmark_jev._is_retryable(ValueError("malformed response"))


def test_a_retry_is_not_started_when_its_wait_would_exceed_the_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(benchmark_jev, "monotonic", lambda: next(ticks))

    report, recorder, acquires, waits = _run_jev_with_guard_stub(
        monkeypatch,
        errors=[_server_error(500), None],
        rows=_stub_rows()[:1],
        retries=5,
        timeout=15.0,
    )

    assert recorder.screen_calls == ["row-1"]  # the attempt budget was already spent
    assert acquires[0] == 1
    assert waits == []
    assert report["samples"]["failed"] == 1
    assert report["samples"]["successful"] == 0


# --------------------------------------------------------------------------- #
# ISSUE 4: --model must stay unset so TYPESAFE_DEFAULT_MODEL can resolve.
# --------------------------------------------------------------------------- #


def test_model_flags_default_to_none() -> None:
    parser = cli.build_parser()

    assert parser.parse_args(["screen", "text", "--side", "query"]).model is None
    assert parser.parse_args(["bench-jev"]).model is None
    assert parser.parse_args(["screen", "text", "--side", "query", "--model", "explicit"]).model == "explicit"
    assert parser.parse_args(["bench-jev", "--model", "explicit"]).model == "explicit"


def test_screen_model_reaches_the_guard_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[dict[str, Any]] = []

    class RecordingGuard:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

        def __enter__(self) -> "RecordingGuard":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def screen(self, text: str, side: Side) -> GuardResult:
            return _guard_result(Side(side))

    monkeypatch.setattr(cli, "JevGuard", RecordingGuard)

    for argv, expected in (
        (["jevguard-nsfa", "screen", "text", "--side", "query"], None),
        (["jevguard-nsfa", "screen", "text", "--side", "query", "--model", "explicit-model"], "explicit-model"),
    ):
        captured.clear()
        monkeypatch.setattr(sys, "argv", argv)
        assert cli.main() == 0
        assert captured[0]["model"] == expected
    capsys.readouterr()


def test_bench_jev_model_comes_from_the_environment_when_no_flag_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "model-from-env")
    factory = _install_real_client_factory(monkeypatch, _StubTransport(model="jev-api-model"), _stub_rows()[:1])

    report = asyncio.run(benchmark_jev.run_benchmark(_jev_args(limit=1)))

    assert factory.kwargs["model"] is None
    # Reaches into the SDK's private resolved config on purpose: this is the resolution the
    # runner relies on, and the SDK does not expose the resolved model anywhere else.
    assert factory.clients[0]._config.default_model == "model-from-env"
    # The emitted model stays the API-reported one rather than the requested name.
    assert report["model"] == "jev-api-model"


def test_explicit_model_overrides_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "model-from-env")
    factory = _install_real_client_factory(monkeypatch, _StubTransport(), _stub_rows()[:1])

    asyncio.run(benchmark_jev.run_benchmark(_jev_args(limit=1, model="explicit-model")))

    assert factory.kwargs["model"] == "explicit-model"
    assert factory.clients[0]._config.default_model == "explicit-model"


def test_report_model_is_null_rather_than_invented_when_nothing_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, *_ = _run_jev_with_guard_stub(
        monkeypatch,
        errors=[_server_error(500)],
        rows=_stub_rows()[:1],
        retries=0,
        model=None,
    )

    assert report["model"] is None


# --------------------------------------------------------------------------- #
# ISSUE 5: the dataset fingerprint covers the sample content, not just the ids.
# --------------------------------------------------------------------------- #


def test_dataset_fingerprint_tracks_row_content_not_only_ids() -> None:
    rows = _stub_rows()
    stable = benchmark_jev._rows_fingerprint(rows)
    assert stable == benchmark_jev._rows_fingerprint(_stub_rows())

    text_changed = [replace(rows[0], text="row-1-edited"), *rows[1:]]
    domains_changed = [replace(rows[0], domains=("resource_abuse",)), *rows[1:]]
    side_changed = [replace(rows[0], side=Side.RESPONSE), *rows[1:]]

    assert benchmark_jev._rows_fingerprint(text_changed) != stable
    assert benchmark_jev._rows_fingerprint(domains_changed) != stable
    assert benchmark_jev._rows_fingerprint(side_changed) != stable
    # The id-only digest deliberately ignores all three edits, which is why the comparator
    # can never gate on it.
    for edited in (text_changed, domains_changed, side_changed):
        assert benchmark_jev._rows_id_digest(edited) == benchmark_jev._rows_id_digest(rows)


def test_dataset_fingerprint_is_byte_identical_to_benchmark_singguard() -> None:
    rows = _stub_rows()

    assert [benchmark_jev._sample_identity(row) for row in rows] == [
        benchmark_singguard._sample_identity(row) for row in rows
    ]
    assert benchmark_jev._rows_fingerprint(rows) == benchmark_singguard._rows_fingerprint(rows)


def test_sample_identity_is_a_compact_utf8_json_array() -> None:
    row = replace(_stub_rows()[0], text="惡意提示", domains=(UNSAFE_DOMAIN, "resource_abuse"))

    assert benchmark_jev._sample_identity(row) == (
        f'["q-1","惡意提示",1,"query",["{UNSAFE_DOMAIN}","resource_abuse"],"en"]'
    )
    assert benchmark_singguard._sample_identity(row) == benchmark_jev._sample_identity(row)
