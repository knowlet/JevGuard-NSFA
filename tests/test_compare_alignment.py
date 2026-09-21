"""Comparator alignment tests for the frozen cross-runner report contract.

The data checks gate the quality deltas and the latency checks gate the latency
deltas. These tests pin the content-complete identity keys (``dataset.fingerprint``
now covers id, text, label, side, domains and lang), ``dataset.revision`` and the
SingGuard-only ``head_manifest.complete`` gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from typing import Any

import pytest

from jevguard_nsfa.compare import alignment_checks, render_markdown


def _sample_identity(row: dict[str, Any]) -> str:
    """The frozen sample identity: byte-identical in both runners and in this test file."""
    return json.dumps(
        [row["id"], row["text"], row["label"], row["side"], list(row["domains"]), row["lang"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(_sample_identity(row) for row in rows).encode("utf-8")).hexdigest()


def _ids_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode("utf-8")).hexdigest()


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "q-1",
        "text": "hello",
        "label": 1,
        "side": "query",
        "domains": ["resource_abuse"],
        "lang": "en",
    }
    row.update(overrides)
    return row


def _head_manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "side": "query",
        "expected_domains": ["prompt_injection_and_jailbreak", "resource_abuse"],
        "loaded_domains": ["prompt_injection_and_jailbreak", "resource_abuse"],
        "missing_domains": [],
        "unexpected_domains": [],
        "complete": True,
        "head_count": 2,
    }
    manifest.update(overrides)
    return manifest


def _report(engine: str, rows: list[dict[str, Any]], *, f1: float) -> dict[str, Any]:
    """A contract-compliant report for one engine."""
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
            "fingerprint": _rows_fingerprint(rows),
            "revision": "abc123",
        },
        "parameters": {"threshold": 0.5},
        "samples": {
            "attempted": len(rows),
            "successful": len(rows),
            "failed": 0,
            "attempted_ids_sha256": _ids_digest(rows),
            "successful_sha256": _rows_fingerprint(rows),
        },
        "latency_scope": "request",
        "quality": {"binary": {"f1": f1, "precision": f1, "recall": f1, "accuracy": f1, "brier": 0.1}},
        "latency_ms": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
        "throughput": {"successful_requests_per_second": 5.0},
        "usage": {"cost_per_1000_successful_requests_usd": 0.01},
    }
    if engine == "singguard-nsfa":
        report["parameters"] = {"threshold": 0.5, "batch_size": 1}
        report["head_manifest"] = _head_manifest()
    return report


def _pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """An aligned pair: same rows, same revision, a complete SingGuard head set."""
    rows = [_row()]
    return _report("jevguard-nsfa", rows, f1=0.9), _report("singguard-nsfa", rows, f1=0.4)


def _row_cells(markdown: str, label: str) -> list[str]:
    line = next(line for line in markdown.splitlines() if line.startswith(f"| {label} |"))
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _check(markdown: str, name: str) -> list[str]:
    return _row_cells(markdown, name)


def _status_of(jev: dict[str, Any], singguard: dict[str, Any], name: str) -> str:
    return next(check.status for check in alignment_checks(jev, singguard) if check.name == name)


# --------------------------------------------------------------------------- #
# Content-complete identity: a text or domains change must break alignment.
# --------------------------------------------------------------------------- #


def test_aligned_contract_reports_keep_quality_and_latency_deltas() -> None:
    jev, singguard = _pair()

    markdown = render_markdown(jev, singguard)

    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "0.5000"]
    assert _row_cells(markdown, "Latency p50 (ms)") == ["Latency p50 (ms)", "10.00", "10.00", "0.00"]
    assert "All alignment checks are ok: quality deltas and latency deltas are reported." in markdown


def test_differing_row_text_is_a_fingerprint_mismatch_that_withholds_quality_deltas() -> None:
    jev_rows = [_row()]
    singguard_rows = [_row(text="hello there")]
    # Same ids, labels, side and lang: only the text differs.
    assert jev_rows[0]["id"] == singguard_rows[0]["id"]
    assert jev_rows[0]["label"] == singguard_rows[0]["label"]
    assert jev_rows[0]["side"] == singguard_rows[0]["side"]
    assert jev_rows[0]["lang"] == singguard_rows[0]["lang"]
    assert _ids_digest(jev_rows) == _ids_digest(singguard_rows)
    assert _rows_fingerprint(jev_rows) != _rows_fingerprint(singguard_rows)

    jev = _report("jevguard-nsfa", jev_rows, f1=0.9)
    singguard = _report("singguard-nsfa", singguard_rows, f1=0.4)

    markdown = render_markdown(jev, singguard)

    assert _status_of(jev, singguard, "dataset.fingerprint") == "mismatch"
    assert _status_of(jev, singguard, "samples.successful_sha256") == "mismatch"
    assert _check(markdown, "dataset.fingerprint")[3] == "mismatch"
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    assert "quality deltas are withheld" in markdown


def test_differing_row_domains_is_a_fingerprint_mismatch_that_withholds_quality_deltas() -> None:
    jev_rows = [_row()]
    singguard_rows = [_row(domains=["prompt_injection_and_jailbreak"])]
    assert _ids_digest(jev_rows) == _ids_digest(singguard_rows)
    assert _rows_fingerprint(jev_rows) != _rows_fingerprint(singguard_rows)

    jev = _report("jevguard-nsfa", jev_rows, f1=0.9)
    singguard = _report("singguard-nsfa", singguard_rows, f1=0.4)

    markdown = render_markdown(jev, singguard)

    assert _status_of(jev, singguard, "dataset.fingerprint") == "mismatch"
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    assert "quality deltas are withheld" in markdown


def test_attempted_ids_sha256_stays_a_diagnostic_only() -> None:
    """The id-only digest must not gate the quality deltas on its own."""
    jev, singguard = _pair()
    singguard["samples"]["attempted_ids_sha256"] = "d" * 64

    markdown = render_markdown(jev, singguard)

    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "0.5000"]
    assert "All alignment checks are ok" in markdown


# --------------------------------------------------------------------------- #
# dataset.revision: mismatch only when both sides recorded a revision.
# --------------------------------------------------------------------------- #


def test_differing_revisions_withhold_quality_deltas() -> None:
    jev, singguard = _pair()
    singguard = copy.deepcopy(singguard)
    singguard["dataset"]["revision"] = "def456"

    markdown = render_markdown(jev, singguard)

    assert _check(markdown, "dataset.revision") == ["dataset.revision", "abc123", "def456", "mismatch"]
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    assert "quality deltas are withheld" in markdown


def test_both_null_revisions_are_ok() -> None:
    jev, singguard = _pair()
    jev = copy.deepcopy(jev)
    singguard = copy.deepcopy(singguard)
    jev["dataset"]["revision"] = None
    singguard["dataset"]["revision"] = None

    markdown = render_markdown(jev, singguard)

    assert _check(markdown, "dataset.revision") == ["dataset.revision", "null", "null", "ok"]
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "0.5000"]


def test_exactly_one_null_revision_is_ok() -> None:
    """The full-content fingerprint is the primary guarantee; a null revision is not a mismatch."""
    jev, singguard = _pair()
    jev = copy.deepcopy(jev)
    jev["dataset"]["revision"] = None

    markdown = render_markdown(jev, singguard)

    assert _check(markdown, "dataset.revision") == ["dataset.revision", "null", "abc123", "ok"]
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "0.5000"]


def test_missing_revision_is_unknown_and_withholds_quality_deltas() -> None:
    """A missing key is not a null: the report predates the field, so alignment is unknown."""
    jev, singguard = _pair()
    jev = copy.deepcopy(jev)
    singguard = copy.deepcopy(singguard)
    del jev["dataset"]["revision"]
    del singguard["dataset"]["revision"]

    markdown = render_markdown(jev, singguard)

    assert _check(markdown, "dataset.revision") == ["dataset.revision", "missing", "missing", "unknown"]
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]


# --------------------------------------------------------------------------- #
# head_manifest.complete: a SingGuard-only data gate.
# --------------------------------------------------------------------------- #


def test_incomplete_head_manifest_withholds_quality_deltas_and_names_the_missing_domains() -> None:
    jev, singguard = _pair()
    singguard = copy.deepcopy(singguard)
    singguard["head_manifest"] = _head_manifest(
        loaded_domains=["prompt_injection_and_jailbreak"],
        missing_domains=["resource_abuse"],
        complete=False,
        head_count=1,
    )

    markdown = render_markdown(jev, singguard)

    assert _check(markdown, "head_manifest.complete (SingGuard)") == [
        "head_manifest.complete (SingGuard)",
        "n/a (managed API)",
        "false",
        "mismatch",
    ]
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    quality_warning = next(line for line in markdown.splitlines() if "quality deltas are withheld" in line)
    assert "SingGuard ran with an incomplete head set (missing: resource_abuse)" in quality_warning
    # The head set is a data property, so the latency deltas stay available.
    assert _row_cells(markdown, "Latency p50 (ms)") == ["Latency p50 (ms)", "10.00", "10.00", "n/a"]


def test_incomplete_head_manifest_warning_is_printed_to_stdout(
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jevguard_nsfa.compare import main_from_args

    jev, singguard = _pair()
    singguard = copy.deepcopy(singguard)
    singguard["head_manifest"] = _head_manifest(missing_domains=["resource_abuse"], complete=False)
    jev_path = tmp_path / "jev.json"
    sing_path = tmp_path / "singguard.json"
    jev_path.write_text(json.dumps(jev), encoding="utf-8")
    sing_path.write_text(json.dumps(singguard), encoding="utf-8")

    exit_code = main_from_args(argparse.Namespace(jev=jev_path, singguard=sing_path, output=None))
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert stdout.splitlines()[0].startswith("WARNING: benchmark runs are not comparable")
    assert "missing: resource_abuse" in stdout.splitlines()[0]
    assert stdout.splitlines()[1] == "# JevGuard-NSFA vs SingGuard-NSFA"


def test_missing_head_manifest_is_unknown_and_withholds_quality_deltas() -> None:
    jev, singguard = _pair()
    singguard = copy.deepcopy(singguard)
    del singguard["head_manifest"]

    markdown = render_markdown(jev, singguard)

    assert _check(markdown, "head_manifest.complete (SingGuard)") == [
        "head_manifest.complete (SingGuard)",
        "n/a (managed API)",
        "missing",
        "unknown",
    ]
    assert _row_cells(markdown, "Binary F1") == ["Binary F1", "0.9000", "0.4000", "n/a"]
    quality_warning = next(line for line in markdown.splitlines() if "quality deltas are withheld" in line)
    assert "head_manifest" in quality_warning


# --------------------------------------------------------------------------- #
# The frozen identity function is shared by both runners.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module_name", ["jevguard_nsfa.benchmark_jev", "jevguard_nsfa.benchmark_singguard"])
def test_runner_sample_identity_matches_the_frozen_definition(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - optional benchmark extra
        pytest.skip(f"{module_name} is not importable: {exc}")
    sample_identity = getattr(module, "_sample_identity", None)
    if sample_identity is None:  # pragma: no cover - runner not updated yet
        pytest.skip(f"{module_name} does not define _sample_identity yet")

    row = _row()
    assert sample_identity(_as_benchmark_row(row)) == _sample_identity(row)


def _as_benchmark_row(row: dict[str, Any]) -> Any:
    from jevguard_nsfa.dataset import BenchmarkRow
    from jevguard_nsfa.models import Side

    return BenchmarkRow(
        id=row["id"],
        text=row["text"],
        label=row["label"],
        side=Side(row["side"]),
        domains=tuple(row["domains"]),
        lang=row["lang"],
    )
