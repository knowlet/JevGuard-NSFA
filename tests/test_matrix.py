from __future__ import annotations

from copy import deepcopy

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
        "samples": {"failed": 0},
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
