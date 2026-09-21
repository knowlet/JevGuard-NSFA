"""Independently validate emitted benchmark result JSON reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _close(actual: float, reported: Any) -> bool:
    return isinstance(reported, (int, float)) and math.isclose(
        actual,
        float(reported),
        rel_tol=1e-10,
        abs_tol=1e-12,
    )


def validate_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    samples = data.get("samples", {})
    attempted = samples.get("attempted")
    successful = samples.get("successful")
    failed = samples.get("failed")
    if not all(isinstance(value, int) for value in (attempted, successful, failed)):
        errors.append("samples.attempted/successful/failed must be integers")
    elif attempted != successful + failed:
        errors.append(
            f"sample accounting mismatch: attempted={attempted}, "
            f"successful={successful}, failed={failed}"
        )

    dataset = data.get("dataset", {})
    fingerprint = dataset.get("fingerprint")
    successful_digest = samples.get("successful_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        errors.append("dataset.fingerprint is not a SHA-256 hex digest")
    if not isinstance(successful_digest, str) or len(successful_digest) != 64:
        errors.append("samples.successful_sha256 is not a SHA-256 hex digest")
    if failed == 0 and fingerprint != successful_digest:
        errors.append("dataset.fingerprint differs from samples.successful_sha256")

    quality = data.get("quality")
    binary = quality.get("binary") if isinstance(quality, dict) else None
    if not isinstance(binary, dict):
        errors.append("quality.binary is missing")
    else:
        try:
            tp = int(binary["tp"])
            tn = int(binary["tn"])
            fp = int(binary["fp"])
            fn = int(binary["fn"])
            total = tp + tn + fp + fn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            accuracy = (tp + tn) / total if total else 0.0
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"invalid quality.binary confusion matrix: {exc}")
        else:
            if successful != total:
                errors.append(
                    f"confusion matrix total {total} differs from successful count {successful}"
                )
            for name, actual in (
                ("accuracy", accuracy),
                ("precision", precision),
                ("recall", recall),
                ("f1", f1),
            ):
                if not _close(actual, binary.get(name)):
                    errors.append(f"quality.binary.{name} does not match TP/FP/TN/FN")

            for name in ("brier", "log_loss"):
                value = binary.get(name)
                if value is not None and (
                    not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0
                ):
                    errors.append(f"quality.binary.{name} must be a finite non-negative number")
            ece = binary.get("expected_calibration_error")
            if ece is not None and (
                not isinstance(ece, (int, float))
                or not math.isfinite(float(ece))
                or not 0.0 <= float(ece) <= 1.0
            ):
                errors.append("quality.binary.expected_calibration_error must be in [0, 1]")

    if data.get("engine") == "singguard-nsfa":
        manifest = data.get("head_manifest")
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            errors.append("SingGuard head_manifest.complete is not true")
        if data.get("baseline_complete") is not True:
            errors.append("SingGuard baseline_complete is not true")
        revision = data.get("model_revision", {}).get("resolved")
        if not isinstance(revision, str) or not revision:
            errors.append("SingGuard model_revision.resolved is missing")

    return {
        "report": str(path),
        "engine": data.get("engine"),
        "benchmark": dataset.get("benchmark"),
        "attempted": attempted,
        "successful": successful,
        "failed": failed,
        "fingerprint": fingerprint,
        "ok": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    args = parser.parse_args()
    reports = [validate_report(path) for path in args.report]
    result = {"ok": all(report["ok"] for report in reports), "reports": reports}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
