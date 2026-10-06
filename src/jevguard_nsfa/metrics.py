"""Dependency-free benchmark metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .dataset import BenchmarkRow
from .models import GuardResult
from .taxonomy import domains_for


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    brier: float
    log_loss: float
    expected_calibration_error: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(labels: Sequence[int], predicted: Sequence[bool], probabilities: Sequence[float]) -> BinaryMetrics:
    if not (len(labels) == len(predicted) == len(probabilities)):
        raise ValueError("labels, predictions and probabilities must have the same length")
    if not labels:
        raise ValueError("at least one sample is required")

    tp = fp = tn = fn = 0
    brier_sum = 0.0
    log_loss_sum = 0.0
    calibration_bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for truth, guess, probability in zip(labels, predicted, probabilities, strict=True):
        if truth not in (0, 1):
            raise ValueError("labels must be binary")
        if truth and guess:
            tp += 1
        elif not truth and guess:
            fp += 1
        elif not truth and not guess:
            tn += 1
        else:
            fn += 1
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        brier_sum += (probability - truth) ** 2
        clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
        log_loss_sum -= truth * math.log(clipped) + (1 - truth) * math.log(1.0 - clipped)
        bin_index = min(int(probability * 10), 9)
        calibration_bins[bin_index].append((probability, truth))

    ece = 0.0
    for bucket in calibration_bins:
        if not bucket:
            continue
        mean_probability = sum(probability for probability, _ in bucket) / len(bucket)
        empirical_rate = sum(truth for _, truth in bucket) / len(bucket)
        ece += len(bucket) / len(labels) * abs(mean_probability - empirical_rate)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return BinaryMetrics(
        accuracy=_safe_div(tp + tn, len(labels)),
        precision=precision,
        recall=recall,
        f1=_safe_div(2 * precision * recall, precision + recall),
        false_positive_rate=_safe_div(fp, fp + tn),
        false_negative_rate=_safe_div(fn, fn + tp),
        brier=brier_sum / len(labels),
        log_loss=log_loss_sum / len(labels),
        expected_calibration_error=ece,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, or ``None`` when the sample is empty."""
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float | None]:
    """Summarize per-request latencies; every field is ``None`` when nothing was measured.

    An empty sample means the measurement is undefined rather than NaN: ``None``
    serializes as JSON ``null`` and keeps ``json.dumps(..., allow_nan=False)`` valid.
    """
    if not latencies_ms:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    values = [float(value) for value in latencies_ms]
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def _validate_score_sets(rows: Sequence[BenchmarkRow], results: Sequence[GuardResult]) -> None:
    """Reject a row whose result cannot be scored for its own side.

    An absent domain score is not a zero probability. Defaulting it would turn
    "this head never ran" into a confident "risk probability 0.0", which flips the
    per-domain metrics (a missed positive becomes a false negative) and makes an
    unmeasured domain look like a real measurement. Every result must therefore
    carry the complete score set of its own side, and a row and its result must
    describe the same side.
    """
    for row, result in zip(rows, results, strict=True):
        if row.side != result.side:
            raise ValueError(
                f"row {row.id!r} is {row.side.value}-side but its result reports the {result.side.value} side"
            )
        missing = sorted(domain.id for domain in domains_for(result.side) if domain.id not in result.scores)
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"{result.side.value}-side result for row {row.id!r} is missing domain scores: {missing_text}"
            )


def evaluate_guard_results(
    rows: Sequence[BenchmarkRow],
    results: Sequence[GuardResult],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    if len(rows) != len(results):
        raise ValueError("rows and results must have the same length")
    _validate_score_sets(rows, results)
    labels = [row.label for row in rows]
    guesses = [result.unsafe for result in results]
    probabilities = [result.max_risk for result in results]
    binary = binary_metrics(labels, guesses, probabilities)

    positives = [(row, result) for row, result in zip(rows, results, strict=True) if row.label == 1 and row.domains]
    # With no positive rows the domain-level accuracy is undefined, not NaN, so the
    # report stays strict-JSON serializable.
    domain_accuracy: float | None = (
        sum(result.predicted_domain in row.domains for row, result in positives) / len(positives)
        if positives
        else None
    )

    domains = sorted({domain for row in rows for domain in row.domains})
    per_domain: dict[str, Mapping[str, float | int]] = {}
    for domain in domains:
        domain_labels = [int(row.label == 1 and domain in row.domains) for row in rows]
        domain_guesses = [result.scores[domain] > threshold for result in results]
        domain_probs = [result.scores[domain] for result in results]
        per_domain[domain] = binary_metrics(domain_labels, domain_guesses, domain_probs).to_dict()

    return {
        "binary": binary.to_dict(),
        "positive_domain_accuracy": domain_accuracy,
        "per_domain": per_domain,
    }
