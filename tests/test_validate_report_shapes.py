"""Offline object-shape regressions for benchmark report validation."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_benchmark_results.py"
validate_report = runpy.run_path(str(SCRIPT))["validate_report"]
NON_OBJECTS = [None, [], "invalid", 7, True]


def valid_report(engine="fixture"):
    return {
        "engine": engine,
        "dataset": {"benchmark": "query", "fingerprint": "a" * 64},
        "samples": {
            "attempted": 4, "successful": 4, "failed": 0, "successful_sha256": "a" * 64,
        },
        "quality": {"binary": {
            "tp": 1, "tn": 1, "fp": 1, "fn": 1,
            "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
        }},
        "head_manifest": {"complete": True},
        "baseline_complete": True,
        "model_revision": {"resolved": "pinned"},
    }


def write_report(tmp_path, data, name="report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("engine", ["fixture", "singguard-nsfa"])
def test_valid_report_summary_is_unchanged(tmp_path, engine):
    path = write_report(tmp_path, valid_report(engine))
    assert validate_report(path) == {
        "report": str(path), "engine": engine, "benchmark": "query",
        "attempted": 4, "successful": 4, "failed": 0,
        "fingerprint": "a" * 64, "ok": True, "errors": [],
    }


@pytest.mark.parametrize("bad", NON_OBJECTS)
@pytest.mark.parametrize("field", ["report", "samples", "dataset", "model_revision"])
def test_non_object_shapes_produce_field_specific_errors(tmp_path, field, bad):
    data = valid_report("singguard-nsfa")
    if field == "report":
        data = bad
    else:
        data[field] = bad
    path = write_report(tmp_path, data)
    checked = validate_report(path)
    assert checked["ok"] is False
    assert checked["report"] == str(path)
    assert f"{field} must be a JSON object" in checked["errors"]
    assert set(checked) == {
        "report", "engine", "benchmark", "attempted", "successful",
        "failed", "fingerprint", "ok", "errors",
    }


@pytest.mark.parametrize("field,expected", [
    ("samples", [
        "samples.attempted/successful/failed must be integers",
        "samples.successful_sha256 is not a SHA-256 hex digest",
        "confusion matrix total 4 differs from successful count None",
    ]),
    ("dataset", [
        "dataset.fingerprint is not a SHA-256 hex digest",
        "dataset.fingerprint differs from samples.successful_sha256",
    ]),
    ("model_revision", ["SingGuard model_revision.resolved is missing"]),
])
def test_missing_object_fields_keep_existing_errors(tmp_path, field, expected):
    data = valid_report("singguard-nsfa")
    del data[field]
    checked = validate_report(write_report(tmp_path, data))
    assert checked["ok"] is False
    assert checked["errors"] == expected


@pytest.mark.parametrize("bad", NON_OBJECTS)
def test_other_engines_do_not_require_model_revision_object(tmp_path, bad):
    data = valid_report()
    data["model_revision"] = bad
    assert validate_report(write_report(tmp_path, data))["ok"] is True


@pytest.mark.parametrize("field", ["report", "samples", "dataset", "model_revision"])
def test_cli_emits_mixed_invalid_and_valid_reports(tmp_path, field):
    data = valid_report("singguard-nsfa")
    if field == "report":
        data = []
    else:
        data[field] = None
    invalid = write_report(tmp_path, data, "invalid.json")
    valid = write_report(tmp_path, valid_report(), "valid.json")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(invalid), "--report", str(valid)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    assert summary["ok"] is False
    assert [report["report"] for report in summary["reports"]] == [str(invalid), str(valid)]
    assert [report["ok"] for report in summary["reports"]] == [False, True]
    assert f"{field} must be a JSON object" in summary["reports"][0]["errors"]
