"""Compare JevGuard and SingGuard benchmark result JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _get(data: dict[str, Any], path: str) -> float | int | None:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(jev: dict[str, Any], singguard: dict[str, Any]) -> str:
    metrics = [
        ("Binary F1", "quality.binary.f1", 4),
        ("Precision", "quality.binary.precision", 4),
        ("Recall", "quality.binary.recall", 4),
        ("Accuracy", "quality.binary.accuracy", 4),
        ("Brier score", "quality.binary.brier", 4),
        ("Positive L1 accuracy", "quality.positive_domain_accuracy", 4),
        ("Latency p50 (ms)", "latency_ms.p50", 2),
        ("Latency p95 (ms)", "latency_ms.p95", 2),
        ("Latency p99 (ms)", "latency_ms.p99", 2),
        ("Throughput (req/s)", "throughput.successful_requests_per_second", 2),
        ("Cost / 1k requests (USD)", "usage.cost_per_1000_successful_requests_usd", 6),
        ("Failed requests", "samples.failed", 0),
    ]

    lines = [
        "# JevGuard-NSFA vs SingGuard-NSFA",
        "",
        "| Metric | JevGuard-NSFA | SingGuard-NSFA | Jev - SingGuard |",
        "|---|---:|---:|---:|",
    ]
    for label, path, digits in metrics:
        left = _get(jev, path)
        right = _get(singguard, path)
        delta = None if left is None or right is None else float(left) - float(right)
        lines.append(f"| {label} | {_fmt(left, digits)} | {_fmt(right, digits)} | {_fmt(delta, digits)} |")

    lines += [
        "",
        "## Method notes",
        "",
        f"- Jev engine: {jev.get('model', 'unknown')} / mode {jev.get('mode', 'unknown')}.",
        f"- SingGuard engine: {singguard.get('model', 'unknown')} / mode {singguard.get('mode', 'unknown')}.",
        "- Compare quality only when both runs used the same dataset subset, sample order/seed, side, and threshold.",
        "- Jev latency is managed-API end-to-end latency. SingGuard local latency is local inference latency unless served behind an endpoint.",
        "- SingGuard cost is reported only when an explicit GPU hourly price was supplied. Missing cost intentionally remains n/a.",
        "- Cold-start/model-load time is reported separately from steady-state inference and is not mixed into online p50/p95.",
        "",
    ]
    return "\n".join(lines)


def main_from_args(args: argparse.Namespace) -> int:
    jev = json.loads(args.jev.read_text(encoding="utf-8"))
    singguard = json.loads(args.singguard.read_text(encoding="utf-8"))
    markdown = render_markdown(jev, singguard)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jev", type=Path, required=True)
    parser.add_argument("--singguard", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return main_from_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
