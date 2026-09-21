"""Render an N-way NSFA benchmark comparison matrix.

Quality deltas are meaningful only when reports describe the same benchmark
selection and threshold. Latency comparisons additionally require request-level
latency with no batching. This module therefore reports alignment explicitly
instead of silently comparing incompatible runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_MISSING = object()

_METRICS: tuple[tuple[str, str, int], ...] = (
    ("F1", "quality.binary.f1", 4),
    ("Accuracy", "quality.binary.accuracy", 4),
    ("Precision", "quality.binary.precision", 4),
    ("Recall", "quality.binary.recall", 4),
    ("Brier", "quality.binary.brier", 4),
    ("Log loss", "quality.binary.log_loss", 4),
    ("ECE", "quality.binary.expected_calibration_error", 4),
    ("Positive L1 accuracy", "quality.positive_domain_accuracy", 4),
    ("Latency p50 (ms)", "latency_ms.p50", 2),
    ("Latency p95 (ms)", "latency_ms.p95", 2),
    ("Throughput (req/s)", "throughput.successful_requests_per_second", 2),
    ("Cost / 1k (USD)", "usage.cost_per_1000_successful_requests_usd", 6),
    ("Failed", "samples.failed", 0),
)


def _get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _raw(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _fmt(value: Any, digits: int) -> str:
    if value is None or value is _MISSING:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if digits else str(int(value))
    return str(value)


def _same_present(values: list[Any]) -> bool:
    known = [value for value in values if value is not _MISSING]
    return bool(known) and all(value == known[0] for value in known[1:])


def _revisions_compatible(revisions: list[Any]) -> bool:
    """Null means unpinned/unknown; conflicting non-null revisions are incompatible."""
    known = {value for value in revisions if value not in (_MISSING, None)}
    return len(known) <= 1


def alignment(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")

    values = list(reports.values())
    benchmark_values = [_raw(report, "dataset.benchmark") for report in values]
    fingerprint_values = [_raw(report, "dataset.fingerprint") for report in values]
    revision_values = [_raw(report, "dataset.revision") for report in values]
    threshold_values = [_raw(report, "parameters.threshold") for report in values]
    latency_scope_values = [_raw(report, "latency_scope") for report in values]
    batch_sizes = [_raw(report, "parameters.batch_size") for report in values]

    benchmark_ok = _same_present(benchmark_values)
    fingerprint_ok = _same_present(fingerprint_values)
    threshold_ok = _same_present(threshold_values)
    revision_ok = _revisions_compatible(revision_values)
    quality_ok = benchmark_ok and fingerprint_ok and threshold_ok and revision_ok

    latency_scope_ok = all(value == "request" for value in latency_scope_values)
    batch_ok = all(value in (_MISSING, 1) for value in batch_sizes)
    latency_ok = quality_ok and latency_scope_ok and batch_ok

    return {
        "quality_comparable": quality_ok,
        "latency_comparable": latency_ok,
        "checks": {
            "dataset.benchmark": benchmark_ok,
            "dataset.fingerprint": fingerprint_ok,
            "dataset.revision": revision_ok,
            "parameters.threshold": threshold_ok,
            "latency_scope=request": latency_scope_ok,
            "batch_size<=1": batch_ok,
        },
    }


def render_markdown(reports: dict[str, dict[str, Any]]) -> str:
    check = alignment(reports)
    labels = list(reports)
    header = "| Metric | " + " | ".join(labels) + " |"
    divider = "|---|" + "|".join("---:" for _ in labels) + "|"
    lines = [
        "# NSFA benchmark matrix",
        "",
        header,
        divider,
    ]
    for label, path, digits in _METRICS:
        values = [_fmt(_get(reports[name], path), digits) for name in labels]
        lines.append("| " + label + " | " + " | ".join(values) + " |")

    lines += [
        "",
        "## Alignment",
        "",
        f"- Quality comparable: **{str(check['quality_comparable']).lower()}**",
        f"- Latency comparable: **{str(check['latency_comparable']).lower()}**",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for name, ok in check["checks"].items():
        lines.append(f"| {name} | {'ok' if ok else 'not aligned'} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- Quality values should only be compared when quality_comparable is true.",
        "- Latency values should only be compared when latency_comparable is true.",
        "- Throughput is shown as reported but may represent sequential requests, batching, or a load test; read each report's mode/notes before treating it as a capacity comparison.",
        "- Local in-process, local HTTP, and managed-API measurements include different runtime boundaries.",
        "",
    ]
    return "\n".join(lines)


def _parse_report(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from exc
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("report label must not be empty")
    path = Path(raw_path)
    return label, path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--report",
        action="append",
        type=_parse_report,
        required=True,
        metavar="LABEL=PATH",
        help="Benchmark report to include; repeat for multiple engines",
    )
    parser.add_argument("--output", type=Path, default=None)


def main_from_args(args: argparse.Namespace) -> int:
    reports: dict[str, dict[str, Any]] = {}
    for label, path in args.report:
        if label in reports:
            raise ValueError(f"duplicate report label {label!r}")
        reports[label] = json.loads(path.read_text(encoding="utf-8"))
    markdown = render_markdown(reports)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return main_from_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
