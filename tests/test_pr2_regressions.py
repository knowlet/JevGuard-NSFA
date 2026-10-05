"""Offline regressions for PR #2: deadlines, workflow, warmup and report checks."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import runpy
import subprocess
from pathlib import Path

import pytest
import yaml
from typesafe_sdk import TypeSafeAPITimeoutError

from jevguard_nsfa import benchmark_jev as jev
from jevguard_nsfa import benchmark_open as opened
from jevguard_nsfa.dataset import BenchmarkRow
from jevguard_nsfa.matrix import alignment
from jevguard_nsfa.models import Side, ThresholdPolicy
from jevguard_nsfa.taxonomy import domains_for

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    async def sleep(self, seconds):
        self.value += seconds


class Limiter:
    def __init__(self, clock, waits=()):
        self.clock = clock
        self.waits = iter(waits)
        self.calls = 0

    async def acquire(self):
        self.calls += 1
        self.clock.value += next(self.waits, 0.0)


def row():
    return BenchmarkRow(id="fixture", text="ordinary text", label=0, side=Side.QUERY, domains=(), lang="en")


def result():
    return opened._guard_result_from_scores(
        side=Side.QUERY,
        scores={domain.id: 0.1 for domain in domains_for(Side.QUERY)},
        policy=ThresholdPolicy(), latency_ms=1.0, model="fixture",
    )


@pytest.mark.parametrize("timeout,budget,expected", [(30.0, 5.0, 5.0), (None, 5.0, 5.0),
                                                       (5.0, 30.0, 5.0), (None, None, None)])
def test_first_attempt_is_bounded_and_initial_queue_is_excluded(timeout, budget, expected):
    clock = Clock()
    limiter = Limiter(clock, [10.0])
    observed = []

    class Guard:
        async def screen(self, text, side, **kwargs):
            observed.append(kwargs.get("timeout"))
            clock.value += 0.25
            return result()

    scored, error, attempts = asyncio.run(jev._screen_with_attempts(
        row(), Guard(), limiter, retries=0, timeout=timeout, budget=budget, clock=clock,
    ))
    assert error is None
    assert attempts == limiter.calls == 1
    assert observed == [expected]
    assert scored.latency_ms == pytest.approx(250.0)


@pytest.mark.parametrize("limiter_wait,expected_attempts,expected_timeout", [(3.0, 2, 1.0), (6.0, 1, None)])
def test_retry_recomputes_remaining_after_limiter(monkeypatch, limiter_wait, expected_attempts, expected_timeout):
    clock = Clock()
    limiter = Limiter(clock, [0.0, limiter_wait])
    observed = []
    monkeypatch.setattr(jev, "retry_wait_seconds", lambda *args: 0.5)

    class Guard:
        async def screen(self, text, side, **kwargs):
            observed.append(kwargs.get("timeout"))
            if len(observed) == 1:
                clock.value += 0.5
                raise TypeSafeAPITimeoutError(0.5)
            clock.value += 0.25
            return result()

    scored, error, attempts = asyncio.run(jev._screen_with_attempts(
        row(), Guard(), limiter, retries=2, timeout=30.0, budget=5.0, clock=clock, sleep=clock.sleep,
    ))
    assert attempts == len(observed) == expected_attempts
    if expected_timeout is None:
        assert scored is None
        assert "budget exhausted" in error
    else:
        assert error is None
        assert observed == pytest.approx([5.0, expected_timeout])
        assert scored.latency_ms == pytest.approx(4250.0)


def test_late_non_yielding_result_is_not_scored():
    clock = Clock()

    class Guard:
        async def screen(self, *args, **kwargs):
            clock.value += 6.0
            return result()

    scored, error, attempts = asyncio.run(jev._screen_with_attempts(
        row(), Guard(), Limiter(clock), retries=0, timeout=30.0, budget=5.0, clock=clock,
    ))
    assert scored is None and "budget exhausted" in error and attempts == 1


def test_budget_cancels_inflight_attempt_and_releases_semaphore():
    async def scenario():
        cancelled = []
        semaphore = asyncio.Semaphore(1)

        class Guard:
            async def screen(self, *args, **kwargs):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(True)

        outcome = await asyncio.wait_for(jev._run_one(
            row(), Guard(), semaphore, Limiter(Clock()), timeout=None, budget=0.03,
        ), timeout=2.0)
        assert outcome[1] is None and "budget exhausted" in outcome[2] and outcome[3] == 1
        assert cancelled == [True]
        assert not semaphore.locked()

    asyncio.run(scenario())


def test_budget_cancels_retry_limiter_without_reserving_slot(monkeypatch):
    monkeypatch.setattr(jev, "retry_wait_seconds", lambda *args: 0.0)

    async def scenario():
        limiter = jev.RequestStartLimiter(1.0)
        reservations = []

        class Guard:
            async def screen(self, *args, **kwargs):
                reservations.append(limiter._next)
                raise TypeSafeAPITimeoutError(0.001)

        scored, error, attempts = await asyncio.wait_for(jev._screen_with_attempts(
            row(), Guard(), limiter, retries=2, timeout=1.0, budget=0.03,
        ), timeout=2.0)
        assert scored is None and "budget exhausted" in error and attempts == 1
        assert len(reservations) == 1 and limiter._next == reservations[0]
        assert not limiter._lock.locked()
        await asyncio.wait_for(limiter._lock.acquire(), timeout=0.2)
        limiter._lock.release()

    asyncio.run(scenario())


def test_external_cancellation_is_not_converted_to_row_failure():
    async def scenario():
        entered = asyncio.Event()

        class Guard:
            async def screen(self, *args, **kwargs):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(jev._screen_with_attempts(
            row(), Guard(), Limiter(Clock()), retries=2, timeout=30.0, budget=60.0,
        ))
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_inner_timeout_error_is_not_mislabeled_as_row_deadline():
    class Guard:
        async def screen(self, *args, **kwargs):
            raise TimeoutError("backend-specific failure")

    scored, error, attempts = asyncio.run(jev._screen_with_attempts(
        row(), Guard(), Limiter(Clock()), retries=2, timeout=30.0, budget=60.0,
    ))
    assert scored is None and error == "TimeoutError: backend-specific failure" and attempts == 1


@pytest.mark.parametrize("rpm,seconds,allowed", [(900.0, 3600.0, False), (900.0, 6000.0, True),
                                                (500.0, 7200.0, False)])
def test_full_query_pacing_budget(rpm, seconds, allowed):
    if not allowed:
        with pytest.raises(ValueError, match="pacing alone"):
            jev.check_pacing_budget(63431, 0, rpm, seconds)
    else:
        assert jev.check_pacing_budget(63431, 0, rpm, seconds) == pytest.approx(4228.6666667)


def test_pacing_includes_warmup_and_uses_actual_selected_count():
    assert jev.check_pacing_budget(1000, 8, 900.0, None) == pytest.approx(1007 / 15)
    with pytest.raises(ValueError, match="pacing alone"):
        jev.check_pacing_budget(100001, 0, 900.0, 6000.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_rpm_is_rejected(value):
    with pytest.raises(ValueError, match="rpm"):
        jev.check_pacing_budget(1, 0, value, None)
    with pytest.raises(ValueError, match="rpm"):
        jev.RequestStartLimiter(value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_pacing_budget_is_rejected(value):
    with pytest.raises(ValueError, match="pacing-budget"):
        jev.check_pacing_budget(1, 0, 900.0, value)


def test_impossible_selection_fails_before_api_client(monkeypatch):
    parser = argparse.ArgumentParser()
    jev.add_arguments(parser)
    args = parser.parse_args(["--full", "--rpm", "1", "--pacing-budget-seconds", "1"])
    monkeypatch.setattr(jev, "iter_huggingface_rows", lambda **kwargs: iter([row(), row()]))

    def forbidden_client(**kwargs):
        pytest.fail("An impossible schedule must not construct the API client")

    monkeypatch.setattr(jev, "AsyncTypeSafeClient", forbidden_client)
    with pytest.raises(ValueError, match="pacing alone"):
        asyncio.run(jev.run_benchmark(args))


def workflow():
    return yaml.load((ROOT / ".github/workflows/benchmark-jev.yml").read_text(), Loader=yaml.BaseLoader)


def test_workflow_timeouts_validation_and_missing_artifact_gate():
    config = workflow()
    job = config["jobs"]["benchmark"]
    assert job["timeout-minutes"] == "${{ inputs.full && 120 || 60 }}"
    steps = {step.get("name"): step for step in job["steps"]}
    assert steps["Run benchmark"]["timeout-minutes"] == "${{ inputs.full && 110 || 50 }}"
    assert "validate_benchmark_results.py" in steps["Validate benchmark report"]["run"]
    assert 'report["samples"]["failed"] != 0' in steps["Validate benchmark report"]["run"]
    assert steps["Upload benchmark report"]["with"]["if-no-files-found"] == "error"
    assert steps["Upload benchmark report"]["with"]["path"] == "benchmark-results/jevguard.json"


@pytest.mark.parametrize("full", [False, True])
def test_workflow_shell_forwards_pinned_protocol(tmp_path, full):
    capture = tmp_path / "argv.json"
    shim = tmp_path / "jevguard-nsfa"
    shim.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'open(os.environ["ARGV_CAPTURE"], "w").write(json.dumps(sys.argv[1:]))\n')
    shim.chmod(0o755)
    step = next(step for step in workflow()["jobs"]["benchmark"]["steps"] if step.get("name") == "Run benchmark")
    assert "${{ inputs." not in step["run"]
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], ARGV_CAPTURE=str(capture),
               BENCHMARK="query", FULL=str(full).lower(), LIMIT="1000", DATASET_REVISION="pinned-revision",
               CONCURRENCY="8", RPM="900", MODEL="model name with spaces")
    subprocess.run(["bash", "-n", "-c", step["run"]], check=True, env=env)
    subprocess.run(["bash", "-e", "-c", step["run"]], check=True, env=env)
    argv = json.loads(capture.read_text())
    assert argv[0] == "bench-jev"
    assert argv[argv.index("--model") + 1] == "model name with spaces"
    assert argv[argv.index("--dataset-revision") + 1] == "pinned-revision"
    assert argv[argv.index("--timeout") + 1] == ("180" if full else "15")
    assert argv[argv.index("--retries") + 1] == ("2" if full else "0")
    assert argv[argv.index("--pacing-budget-seconds") + 1] == ("6000" if full else "2400")
    assert ("--full" in argv) is full
    assert ("--limit" in argv) is not full
    if full:
        assert argv[argv.index("--row-budget") + 1] == "600"


@pytest.mark.parametrize("warmup,fail_fast", [(1, False), (5, False), (0, False), (1, True)])
def test_open_warmup_failure_policy_and_accounting(monkeypatch, warmup, fail_fast):
    parser = argparse.ArgumentParser()
    opened.add_arguments(parser)
    args = parser.parse_args(["--backend", "systemone-http", "--engine", "fixture", "--warmup", str(warmup)])
    args.fail_fast = fail_fast

    class Backend:
        mode = "fixture"

        def __init__(self):
            self.calls = 0
            self.closed = False

        def screen(self, *args):
            self.calls += 1
            if warmup and self.calls == 1:
                raise RuntimeError("warmup failure")
            return result()

        def close(self):
            self.closed = True

        def metadata(self):
            return {}

    backend = Backend()
    monkeypatch.setattr(opened, "build_backend", lambda args: backend)
    monkeypatch.setattr(opened, "iter_huggingface_rows", lambda **kwargs: iter([row()]))
    if fail_fast:
        with pytest.raises(RuntimeError, match="warmup failure"):
            opened.run_benchmark(args)
        assert backend.calls == 1
    else:
        report = opened.run_benchmark(args)
        assert report["warmup"]["requested"] == warmup
        assert report["warmup"]["attempted"] == min(warmup, 1)
        assert report["warmup"]["failed"] == min(warmup, 1)
        assert report["samples"]["failed"] == 0 and report["samples"]["successful"] == 1
        assert report["samples"]["successful_sha256"] == report["dataset"]["fingerprint"]
        assert report["failures"] == [] and report["quality"] is not None
        assert report["latency_ms"]["p50"] == 1.0
    assert backend.closed


@pytest.mark.parametrize("path,value,quality_ok,check", [
    ("dataset.benchmark", "response", False, "dataset.benchmark"),
    ("parameters.threshold", 0.6, False, "parameters.threshold"),
    ("latency_scope", "batch", True, "latency_scope=request"),
])
def test_matrix_mismatches_are_independently_gated(path, value, quality_ok, check):
    from test_matrix import _report

    left, right = _report(), _report()
    target = right
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    checked = alignment({"left": left, "right": right})
    assert checked["checks"][check] is False
    assert checked["quality_comparable"] is quality_ok
    assert checked["latency_comparable"] is False


def valid_report():
    return {"engine": "fixture", "dataset": {"fingerprint": "a" * 64},
            "samples": {"attempted": 4, "successful": 4, "failed": 0, "successful_sha256": "a" * 64},
            "quality": {"binary": {"tp": 1, "tn": 1, "fp": 1, "fn": 1,
                                   "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                                   "brier": 0.1, "log_loss": 0.2, "expected_calibration_error": 0.1}}}


@pytest.mark.parametrize("field", ["brier", "log_loss", "expected_calibration_error", "accuracy"])
@pytest.mark.parametrize("bad", [10**400, True, float("inf"), "not a number"])
def test_validator_records_invalid_numeric_fields_without_crashing(tmp_path, field, bad):
    validate = runpy.run_path(str(ROOT / "scripts/validate_benchmark_results.py"))["validate_report"]
    data = valid_report()
    data["quality"]["binary"][field] = bad
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    checked = validate(path)
    assert checked["ok"] is False
    assert any(field in message for message in checked["errors"])


@pytest.mark.parametrize("missing", [False, True])
def test_validator_keeps_legacy_calibration_null_or_missing_compatible(tmp_path, missing):
    validate = runpy.run_path(str(ROOT / "scripts/validate_benchmark_results.py"))["validate_report"]
    data = valid_report()
    for field in ("brier", "log_loss", "expected_calibration_error"):
        if missing:
            del data["quality"]["binary"][field]
        else:
            data["quality"]["binary"][field] = None
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    assert validate(path)["ok"] is True


def test_comparison_rounds_after_subtraction():
    from test_reporting import _aligned_reports

    from jevguard_nsfa.compare import render_markdown

    left, right = _aligned_reports()
    left["quality"]["binary"]["brier"] = 0.087849
    right["quality"]["binary"]["brier"] = 0.136051
    rendered = render_markdown(left, right)
    assert "| Brier score | 0.0878 | 0.1361 | -0.0482 |" in rendered
    assert "rounded independently" in rendered


def test_documents_disclose_full_set_evidence_status_and_rounding():
    validation = (ROOT / "BENCHMARK_VALIDATION.md").read_text()
    assert "author-reported" in validation
    assert "原始 JSON" in validation and "未隨本 PR 提供" in validation
    assert "Rounding rule" in validation and "not been regenerated" in validation
    readme = (ROOT / "README.md").read_text()
    assert "Decider-2b on cross-source-query only" in readme
    assert "pending independent artifact and fingerprint verification" in readme
