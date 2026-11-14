"""Offline CLI regressions for non-negative JSON integer sample counts."""

import json
import subprocess
import sys

import pytest
from test_validate_report_shapes import SCRIPT, valid_report, validate_report, write_report


def report_with_counts(attempted, successful, failed):
    data = valid_report()
    data["samples"].update(attempted=attempted, successful=successful, failed=failed)
    binary = data["quality"]["binary"]
    # Keep the confusion matrix valid independently of malformed sample counts.
    count = max(0, int(successful))
    binary.update(tp=count, tn=0, fp=0, fn=0)
    for field in ("accuracy", "precision", "recall", "f1"):
        binary[field] = 1.0 if count else 0.0
    return data


@pytest.mark.parametrize("counts", [
    (True, 1, 0), (1, True, 0), (1, 1, False), (1, 0, True),
    (-1, 1, -2), (0, -1, 1), (-1, 0, -1),
])
def test_cli_rejects_invalid_sample_counts_without_aborting_batch(tmp_path, counts):
    invalid = write_report(tmp_path, report_with_counts(*counts), "invalid.json")
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
    assert any("must be" in error and "samples.attempted/successful/failed" in error
               for error in summary["reports"][0]["errors"])
    assert summary["reports"][1]["errors"] == []


@pytest.mark.parametrize("counts", [(0, 0, 0), (7, 4, 3), (10**400 + 3, 10**400, 3)])
def test_cli_preserves_nonnegative_integer_boundaries(tmp_path, counts):
    path = write_report(tmp_path, report_with_counts(*counts))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    assert summary["ok"] is True
    report = summary["reports"][0]
    assert tuple(report[field] for field in ("attempted", "successful", "failed")) == counts
    assert report["errors"] == []


@pytest.mark.parametrize("field", ["attempted", "successful", "failed"])
def test_missing_sample_count_keeps_existing_error(tmp_path, field):
    data = valid_report()
    del data["samples"][field]
    report = validate_report(write_report(tmp_path, data))
    assert "samples.attempted/successful/failed must be integers" in report["errors"]
