from __future__ import annotations

from copy import deepcopy

import pytest

from jevguard_nsfa.matrix import alignment, render_markdown


def _report(*, fingerprint: str = "a" * 64, revision: str | None = "dataset-rev", batch_size: int | None = None):
    parameters = {"threshold": 0.5}
    if batch_size is not None:
        parameters["batch_size"] = batch_size
    return {
        "engine": "test",
        "dataset": {
            "benchmark": "query",
            "fingerprint": fingerprint,
            "revision": revision,
        },
        "parameters": parameters,
        "latency_scope": "request",
        "samples": {"successful": 100, "failed": 0, "successful_sha256": fingerprint},
        "quality": {
            "binary": {
                "f1": 0.8,
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "brier": 0.1,
            },
            "positive_domain_accuracy": 0.7,
        },
        "latency_ms": {"p50": 10.0, "p95": 20.0},
        "throughput": {"successful_requests_per_second": 100.0},
        "usage": {"cost_per_1000_successful_requests_usd": 0.01},
    }


def test_aligned_reports_allow_quality_and_latency_comparison() -> None:
    result = alignment({"a": _report(), "b": _report()})
    assert result["quality_comparable"] is True
    assert result["latency_comparable"] is True


def test_batch_throughput_run_keeps_quality_but_withholds_latency() -> None:
    result = alignment({"online": _report(), "batch": _report(batch_size=16)})
    assert result["quality_comparable"] is True
    assert result["latency_comparable"] is False
    assert result["checks"]["batch_size<=1"] is False


def test_fingerprint_mismatch_withholds_quality() -> None:
    result = alignment({"a": _report(), "b": _report(fingerprint="b" * 64)})
    assert result["quality_comparable"] is False
    assert result["latency_comparable"] is False


def test_conflicting_pinned_revisions_withhold_quality() -> None:
    result = alignment({"a": _report(revision="one"), "b": _report(revision="two")})
    assert result["quality_comparable"] is False


def test_unpinned_revision_does_not_contradict_matching_content() -> None:
    result = alignment({"pinned": _report(revision="one"), "legacy": _report(revision=None)})
    assert result["quality_comparable"] is True


def test_render_markdown_contains_all_engine_labels() -> None:
    reports = {"Jev": _report(), "Kev": deepcopy(_report()), "Laya": deepcopy(_report())}
    rendered = render_markdown(reports)
    assert "| Metric | Jev | Kev | Laya |" in rendered
    assert "Quality comparable: **true**" in rendered


def test_successful_subset_mismatch_withholds_quality() -> None:
    left = _report()
    right = _report()
    right["samples"]["successful"] = 99
    right["samples"]["successful_sha256"] = "c" * 64
    result = alignment({"left": left, "right": right})
    assert result["quality_comparable"] is False
    assert result["checks"]["samples.successful"] is False
    assert result["checks"]["samples.successful_sha256"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**400])
@pytest.mark.parametrize("section,key,row", [
    ("latency_ms", "p50", "Latency p50 (ms)"),
    ("samples", "failed", "Failed"),
])
def test_invalid_numeric_metrics_render_as_unavailable(value, section, key, row) -> None:
    report = _report()
    report[section][key] = value
    rendered = render_markdown({"invalid": report, "control": _report()})
    expected = "10.00" if section == "latency_ms" else "0"
    assert f"| {row} | n/a | {expected} |" in rendered
    assert "| F1 | 0.8000 | 0.8000 |" in rendered


def test_matrix_cli_handles_json_numeric_overflow(tmp_path, capsys) -> None:
    import argparse
    import json

    from jevguard_nsfa.matrix import main_from_args

    report = _report()
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report).replace('"failed": 0', '"failed": 1e309'))
    output = tmp_path / "nested" / "matrix.md"
    assert main_from_args(argparse.Namespace(report=[("test", source)], output=output)) == 0
    assert "| Failed | n/a |" in output.read_text()
    assert output.read_text().strip() == capsys.readouterr().out.strip()


@pytest.mark.parametrize("value,expected", [(0, "0"), (3, "3"), (3.0, "3"), (3.9, "3")])
def test_finite_count_format_is_preserved(value, expected) -> None:
    report = _report()
    report["samples"]["failed"] = value
    assert f"| Failed | {expected} |" in render_markdown({"test": report})
