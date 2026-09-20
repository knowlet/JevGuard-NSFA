"""Compare JevGuard and SingGuard benchmark result JSON files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MISSING = object()

# Row groups: accuracy rows, latency rows, and the remaining operational rows.
QUALITY = "quality"
LATENCY = "latency"
OTHER = "other"


@dataclass(frozen=True)
class AlignmentCheck:
    """One metadata item that must line up before a delta is meaningful."""

    name: str
    jev: Any
    singguard: Any
    status: str  # "ok" | "mismatch" | "unknown"


def _raw(data: dict[str, Any], path: str) -> Any:
    """Return the JSON value at ``path``, or ``_MISSING`` when the key is absent.

    A present JSON ``null`` is a real value here (for example ``dataset.id_contains``
    or ``dataset.languages`` are legitimately null when no filter was applied), so it
    is returned as ``None`` rather than as missing metadata.
    """
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _get(data: dict[str, Any], path: str) -> float | int | None:
    value = _raw(data, path)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _render_value(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    if value is None:
        return "null"
    rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    # Keeps a value containing a pipe from breaking the markdown table.
    return rendered.replace("|", "\\|")


_DATA_PATHS: tuple[tuple[str, str], ...] = (
    ("dataset.name", "dataset.name"),
    ("dataset.split", "dataset.split"),
    ("dataset.benchmark", "dataset.benchmark"),
    ("dataset.languages", "dataset.languages"),
    ("dataset.id_contains", "dataset.id_contains"),
    ("dataset.seed", "dataset.seed"),
    ("dataset.fingerprint", "dataset.fingerprint"),
    ("parameters.threshold", "parameters.threshold"),
    ("samples.attempted", "samples.attempted"),
    ("samples.successful_ids_sha256", "samples.successful_ids_sha256"),
)


def _equality_check(name: str, jev: dict[str, Any], singguard: dict[str, Any], path: str) -> AlignmentCheck:
    jev_value = _raw(jev, path)
    singguard_value = _raw(singguard, path)
    if jev_value is _MISSING or singguard_value is _MISSING:
        status = "unknown"
    else:
        status = "ok" if jev_value == singguard_value else "mismatch"
    return AlignmentCheck(name, jev_value, singguard_value, status)


def _latency_checks(jev: dict[str, Any], singguard: dict[str, Any]) -> list[AlignmentCheck]:
    jev_scope = _raw(jev, "latency_scope")
    singguard_scope = _raw(singguard, "latency_scope")
    if jev_scope is _MISSING or singguard_scope is _MISSING:
        scope_status = "unknown"
    elif jev_scope == "request" and singguard_scope == "request":
        scope_status = "ok"
    else:
        scope_status = "mismatch"

    batch_size = _raw(singguard, "parameters.batch_size")
    if batch_size is _MISSING:
        batch_status = "unknown"
    else:
        batch_status = "ok" if batch_size == 1 else "mismatch"
    # The Jev runner is not batched, so it has no batch size to show here.
    return [
        AlignmentCheck("latency_scope", jev_scope, singguard_scope, scope_status),
        AlignmentCheck("parameters.batch_size (SingGuard)", "n/a (not batched)", batch_size, batch_status),
    ]


@dataclass(frozen=True)
class Alignment:
    """Alignment checks grouped by what they gate.

    The data checks describe the measured selection and decision threshold, so they
    gate the quality deltas. The latency checks describe whether both engines measured
    an online per-request latency, so they gate the latency deltas -- a throughput run
    with a large batch size still classifies exactly the same samples, so it must keep
    its quality deltas.
    """

    data: list[AlignmentCheck]
    latency: list[AlignmentCheck]

    @property
    def checks(self) -> list[AlignmentCheck]:
        return [*self.data, *self.latency]

    @property
    def data_ok(self) -> bool:
        return all(check.status == "ok" for check in self.data)

    @property
    def latency_ok(self) -> bool:
        return all(check.status == "ok" for check in self.latency)


def alignment(jev: dict[str, Any], singguard: dict[str, Any]) -> Alignment:
    """Verify every metadata item that must agree before a delta is meaningful."""
    data = [_equality_check(name, jev, singguard, path) for name, path in _DATA_PATHS]
    return Alignment(data=data, latency=_latency_checks(jev, singguard))


def alignment_checks(jev: dict[str, Any], singguard: dict[str, Any]) -> list[AlignmentCheck]:
    """Every alignment check, data checks first, each classified ok / mismatch / unknown."""
    return alignment(jev, singguard).checks


def _quality_delta_allowed(checks: Alignment) -> bool:
    """Quality deltas need only the data alignment checks to be ok."""
    return checks.data_ok


def _latency_delta_allowed(checks: Alignment) -> bool:
    """Latency deltas need the data checks and the latency checks to be ok.

    A SingGuard batch size above one measures a batch-completion wait rather than an
    online request latency, so it is not comparable with Jev request latency. The Jev
    runner is not batched and reports ``latency_scope == "request"`` directly.
    """
    return checks.data_ok and checks.latency_ok


def _not_ok(checks: list[AlignmentCheck]) -> list[AlignmentCheck]:
    return [check for check in checks if check.status != "ok"]


def _latency_withheld_reasons(latency: list[AlignmentCheck]) -> list[str]:
    """Human-readable reasons why the latency rows are not comparable."""
    reasons: list[str] = []
    for check in _not_ok(latency):
        if check.name == "latency_scope":
            if check.status == "unknown":
                sides = [
                    side
                    for side, value in (("JevGuard-NSFA", check.jev), ("SingGuard-NSFA", check.singguard))
                    if value is _MISSING
                ]
                reasons.append(
                    f"latency_scope is missing on {' and '.join(sides)} (both reports must report \"request\")"
                )
            else:
                reasons.append(
                    f"latency_scope is {_render_value(check.jev)} on JevGuard-NSFA and "
                    f"{_render_value(check.singguard)} on SingGuard-NSFA (both must be \"request\")"
                )
        elif check.status == "unknown":
            reasons.append("SingGuard parameters.batch_size is missing (required value: 1)")
        else:
            reasons.append(
                f"SingGuard parameters.batch_size is {_render_value(check.singguard)} (required value: 1, because "
                "a batch above 1 measures a batch-completion wait, not an online request latency)"
            )
    return reasons


_METRICS: tuple[tuple[str, str, int, str], ...] = (
    ("Binary F1", "quality.binary.f1", 4, QUALITY),
    ("Precision", "quality.binary.precision", 4, QUALITY),
    ("Recall", "quality.binary.recall", 4, QUALITY),
    ("Accuracy", "quality.binary.accuracy", 4, QUALITY),
    ("Brier score", "quality.binary.brier", 4, QUALITY),
    ("Positive L1 accuracy", "quality.positive_domain_accuracy", 4, QUALITY),
    ("Latency p50 (ms)", "latency_ms.p50", 2, LATENCY),
    ("Latency p95 (ms)", "latency_ms.p95", 2, LATENCY),
    ("Latency p99 (ms)", "latency_ms.p99", 2, LATENCY),
    ("Throughput (req/s)", "throughput.successful_requests_per_second", 2, OTHER),
    ("Cost / 1k requests (USD)", "usage.cost_per_1000_successful_requests_usd", 6, OTHER),
    ("Failed requests", "samples.failed", 0, OTHER),
)


def _alignment_lines(checks: Alignment) -> list[str]:
    lines = [
        "Data checks gate the quality deltas; latency checks gate the latency deltas.",
        "",
        "| Check | JevGuard-NSFA | SingGuard-NSFA | Status |",
        "|---|---|---|---|",
    ]
    for check in checks.checks:
        lines.append(
            f"| {check.name} | {_render_value(check.jev)} | {_render_value(check.singguard)} | {check.status} |"
        )
    lines.append("")
    if checks.data_ok and checks.latency_ok:
        lines.append("All alignment checks are ok: quality deltas and latency deltas are reported.")
        return lines
    if checks.data_ok:
        lines.append("Data alignment is ok: quality deltas are reported.")
    else:
        reasons = ", ".join(f"{check.name} ({check.status})" for check in _not_ok(checks.data))
        lines.append(
            "WARNING: these runs are not comparable ("
            f"{reasons}); quality deltas are withheld and rendered n/a."
        )
    latency_reasons = _latency_withheld_reasons(checks.latency)
    if latency_reasons:
        data_note = "" if checks.data_ok else "; the data alignment checks above are also not ok"
        lines.append("WARNING: latency deltas are withheld: " + "; ".join(latency_reasons) + data_note + ".")
    elif checks.data_ok:
        lines.append("Latency alignment is ok: latency deltas are reported.")
    else:
        lines.append("Latency deltas are withheld as well: the data alignment checks above are not ok.")
    return lines


def render_markdown(jev: dict[str, Any], singguard: dict[str, Any]) -> str:
    checks = alignment(jev, singguard)
    quality_allowed = _quality_delta_allowed(checks)
    latency_allowed = _latency_delta_allowed(checks)

    lines = [
        "# JevGuard-NSFA vs SingGuard-NSFA",
        "",
        "| Metric | JevGuard-NSFA | SingGuard-NSFA | Jev - SingGuard |",
        "|---|---:|---:|---:|",
    ]
    for label, path, digits, kind in _METRICS:
        left = _get(jev, path)
        right = _get(singguard, path)
        allowed = latency_allowed if kind == LATENCY else quality_allowed
        delta = None if left is None or right is None or not allowed else float(left) - float(right)
        lines.append(f"| {label} | {_fmt(left, digits)} | {_fmt(right, digits)} | {_fmt(delta, digits)} |")

    lines += ["", "## Alignment", ""]
    lines += _alignment_lines(checks)

    lines += [
        "",
        "## Method notes",
        "",
        f"- Jev engine: {jev.get('model', 'unknown')} / mode {jev.get('mode', 'unknown')}.",
        f"- SingGuard engine: {singguard.get('model', 'unknown')} / mode {singguard.get('mode', 'unknown')}.",
        "- Compare quality only when both runs used the same dataset subset, sample order/seed, side, and threshold.",
        "- Jev latency is managed-API end-to-end latency. SingGuard local latency is local inference latency unless served behind an endpoint.",
        "- Latency deltas require `latency_scope` == `request` on both runs and a SingGuard `parameters.batch_size` of 1.",
        "- SingGuard cost is reported only when an explicit GPU hourly price was supplied. Missing cost intentionally remains n/a.",
        "- Cold-start/model-load time is reported separately from steady-state inference and is not mixed into online p50/p95.",
        "",
    ]
    return "\n".join(lines)


def main_from_args(args: argparse.Namespace) -> int:
    jev = json.loads(args.jev.read_text(encoding="utf-8"))
    singguard = json.loads(args.singguard.read_text(encoding="utf-8"))
    checks = alignment(jev, singguard)
    markdown = render_markdown(jev, singguard)
    data_mismatches = [check.name for check in _not_ok(checks.data) if check.status == "mismatch"]
    latency_mismatches = [check.name for check in _not_ok(checks.latency) if check.status == "mismatch"]
    if data_mismatches:
        print(
            "WARNING: benchmark runs are not comparable; mismatched alignment checks: "
            + ", ".join(data_mismatches)
            + ". Quality deltas are withheld."
        )
    if latency_mismatches:
        print(
            "WARNING: latency deltas are withheld; mismatched latency checks: "
            + ", ".join(latency_mismatches)
            + "."
        )
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
