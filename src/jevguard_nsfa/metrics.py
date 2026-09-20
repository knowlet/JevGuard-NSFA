"""Dependency-free benchmark metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .dataset import BenchmarkRow
from .models import GuardResult


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    brier: float
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
        brier_sum += (float(probability) - truth) ** 2

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
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
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


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan, "min": math.nan, "max": math.nan}
    values = [float(value) for value in latencies_ms]
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def evaluate_guard_results(
    rows: Sequence[BenchmarkRow],
    results: Sequence[GuardResult],
) -> dict[str, object]:
    if len(rows) != len(results):
        raise ValueError("rows and results must have the same length")
    labels = [row.label for row in rows]
    guesses = [result.unsafe for result in results]
    probabilities = [result.max_risk for result in results]
    binary = binary_metrics(labels, guesses, probabilities)

    positives = [(row, result) for row, result in zip(rows, results, strict=True) if row.label == 1 and row.domain]
    domain_accuracy = (
        sum(result.predicted_domain == row.domain for row, result in positives) / len(positives) if positives else math.nan
    )

    domains = sorted({row.domain for row in rows if row.domain})
    per_domain: dict[str, Mapping[str, float | int]] = {}
    for domain in domains:
        domain_labels = [int(row.label == 1 and row.domain == domain) for row in rows]
        domain_guesses = [result.scores.get(domain, 0.0) >= 0.5 for result in results]
        domain_probs = [result.scores.get(domain, 0.0) for result in results]
        per_domain[domain] = binary_metrics(domain_labels, domain_guesses, domain_probs).to_dict()

    return {
        "binary": binary.to_dict(),
        "positive_domain_accuracy": domain_accuracy,
        "per_domain": per_domain,
    }
