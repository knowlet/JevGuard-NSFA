from __future__ import annotations

from jevguard_nsfa.dataset import canonical_domains, row_from_mapping
from jevguard_nsfa.metrics import binary_metrics, evaluate_guard_results
from jevguard_nsfa.models import Decision, GuardResult, Side


def test_multi_label_ground_truth_is_preserved() -> None:
    row = row_from_mapping(
        {
            "id": "AgentHarm-0119",
            "text": "example",
            "label": 1,
            "L1-Risk": "sensitive_info_stealing;danger_ops_and_tool_abuse",
            "lang": "en",
        },
        forced_side=Side.QUERY,
    )
    assert row.domains == (
        "sensitive_information_stealing",
        "dangerous_operations_and_tool_abuse",
    )
    assert row.side is Side.QUERY


def test_no_risk_has_no_domain() -> None:
    assert canonical_domains("No_Risk") == ()


def test_binary_metrics_known_confusion_matrix() -> None:
    metrics = binary_metrics(
        labels=[1, 1, 0, 0],
        predicted=[True, False, True, False],
        probabilities=[0.9, 0.4, 0.6, 0.1],
    )
    assert (metrics.tp, metrics.fn, metrics.fp, metrics.tn) == (1, 1, 1, 1)
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_positive_domain_accuracy_accepts_any_ground_truth_label() -> None:
    rows = [
        row_from_mapping(
            {
                "id": "x",
                "text": "example",
                "label": 1,
                "L1-Risk": "sensitive_info_stealing;danger_ops_and_tool_abuse",
                "lang": "en",
            },
            forced_side=Side.QUERY,
        )
    ]
    results = [
        GuardResult(
            side=Side.QUERY,
            scores={
                "sensitive_information_stealing": 0.7,
                "dangerous_operations_and_tool_abuse": 0.9,
            },
            unsafe=True,
            decision=Decision.BLOCK,
            predicted_domain="dangerous_operations_and_tool_abuse",
            max_risk=0.9,
            latency_ms=1.0,
            model="test",
        )
    ]
    metrics = evaluate_guard_results(rows, results)
    assert metrics["positive_domain_accuracy"] == 1.0
