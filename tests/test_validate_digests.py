"""Reject malformed SHA-256 representations without changing alignment semantics."""

import json
import subprocess
import sys

import pytest
from test_validate_report_shapes import SCRIPT, valid_report, validate_report, write_report


@pytest.mark.parametrize("value", ["g" * 64, " " * 64, "a" * 63 + "\n", "０" * 64, "a" * 63 + "_"])
@pytest.mark.parametrize("section,field", [("dataset", "fingerprint"), ("samples", "successful_sha256")])
def test_digest_rejects_non_hex_characters(tmp_path, section, field, value):
    data = valid_report()
    data[section][field] = value
    result = validate_report(write_report(tmp_path, data))
    assert not result["ok"]
    assert f"{section}.{field} is not a SHA-256 hex digest" in result["errors"]


@pytest.mark.parametrize("value", ["0" * 64, "0123456789abcdef" * 4, "0123456789ABCDEF" * 4])
def test_valid_hex_digest_is_preserved(tmp_path, value):
    data = valid_report()
    data["dataset"]["fingerprint"] = data["samples"]["successful_sha256"] = value
    result = validate_report(write_report(tmp_path, data))
    assert result["ok"]
    assert result["fingerprint"] == value


def test_equal_malformed_digests_fail_cli_without_losing_valid_report(tmp_path):
    data = valid_report()
    data["dataset"]["fingerprint"] = data["samples"]["successful_sha256"] = "z" * 64
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
    assert [item["ok"] for item in summary["reports"]] == [False, True]
    assert summary["reports"][0]["errors"] == [
        "dataset.fingerprint is not a SHA-256 hex digest",
        "samples.successful_sha256 is not a SHA-256 hex digest",
    ]


def test_distinct_valid_digests_still_fail_alignment(tmp_path):
    data = valid_report()
    data["samples"]["successful_sha256"] = "b" * 64
    result = validate_report(write_report(tmp_path, data))
    assert result["errors"] == ["dataset.fingerprint differs from samples.successful_sha256"]
    assert not result["ok"]
