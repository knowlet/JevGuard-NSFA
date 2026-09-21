from __future__ import annotations

import pytest

from jevguard_nsfa.dataset import BenchmarkRow, canonical_domains, row_from_mapping
from jevguard_nsfa.metrics import binary_metrics, evaluate_guard_results
from jevguard_nsfa.models import Decision, GuardResult, Side
from jevguard_nsfa.taxonomy import domains_for


def _scores_for(side: Side, **overrides: float) -> dict[str, float]:
    """A complete, low-risk score set for every domain of ``side``."""
    scores = {domain.id: 0.1 for domain in domains_for(side)}
    scores.update(overrides)
    return scores


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


def test_official_response_domain_aliases_are_canonicalized() -> None:
    assert canonical_domains("hazardous_action_output") == ("hazardous_action_generation",)
    assert canonical_domains("sensitive_info_output") == ("sensitive_information_leakage",)
    assert canonical_domains("hazardous_action_output;sensitive_info_output") == (
        "hazardous_action_generation",
        "sensitive_information_leakage",
    )


def test_official_singguard_head_domain_alias_is_canonicalized() -> None:
    assert canonical_domains("Dangerous_Operations_Tool_Abuse") == (
        "dangerous_operations_and_tool_abuse",
    )


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
            scores=_scores_for(
                Side.QUERY,
                sensitive_information_stealing=0.7,
                dangerous_operations_and_tool_abuse=0.9,
            ),
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


def test_complete_score_set_evaluates_exactly_as_before() -> None:
    rows = [
        BenchmarkRow(
            id="q-1",
            text="a",
            label=1,
            side=Side.QUERY,
            domains=("prompt_injection_and_jailbreak",),
            lang="en",
        ),
        BenchmarkRow(id="q-2", text="b", label=0, side=Side.QUERY, domains=(), lang="en"),
    ]
    results = [
        GuardResult(
            side=Side.QUERY,
            scores=_scores_for(Side.QUERY, prompt_injection_and_jailbreak=0.9),
            unsafe=True,
            decision=Decision.BLOCK,
            predicted_domain="prompt_injection_and_jailbreak",
            max_risk=0.9,
            latency_ms=1.0,
            model="test",
        ),
        GuardResult(
            side=Side.QUERY,
            scores=_scores_for(Side.QUERY, prompt_injection_and_jailbreak=0.4, malicious_code_and_cyberattack=0.6),
            unsafe=True,
            decision=Decision.BLOCK,
            predicted_domain="malicious_code_and_cyberattack",
            max_risk=0.6,
            latency_ms=1.0,
            model="test",
        ),
    ]

    metrics = evaluate_guard_results(rows, results)

    assert metrics["binary"] == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 * 0.5 * 1.0 / (0.5 + 1.0),
        "false_positive_rate": 1.0,
        "false_negative_rate": 0.0,
        "brier": ((0.9 - 1) ** 2 + (0.6 - 0) ** 2) / 2,
        "tp": 1,
        "fp": 1,
        "tn": 0,
        "fn": 0,
    }
    assert metrics["positive_domain_accuracy"] == 1.0
    # Only the domains that appear in the rows get per-domain metrics.
    assert set(metrics["per_domain"]) == {"prompt_injection_and_jailbreak"}
    assert metrics["per_domain"]["prompt_injection_and_jailbreak"] == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "brier": ((0.9 - 1) ** 2 + (0.4 - 0) ** 2) / 2,
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 0,
    }


def test_missing_side_domain_raises_instead_of_scoring_a_zero_probability() -> None:
    # A result that carries only some of its side's domains used to make the absent
    # domain a confident 0.0 risk probability (fn == positives) instead of an error.
    rows = [
        row_from_mapping(
            {"id": "q-1", "text": "example", "label": 1, "L1-Risk": "resource_abuse", "lang": "en"},
            forced_side=Side.QUERY,
        )
    ]
    scores = _scores_for(Side.QUERY)
    del scores["resource_abuse"]
    results = [
        GuardResult(
            side=Side.QUERY,
            scores=scores,
            unsafe=False,
            decision=Decision.ALLOW,
            predicted_domain="prompt_injection_and_jailbreak",
            max_risk=0.1,
            latency_ms=1.0,
            model="test",
        )
    ]

    with pytest.raises(ValueError) as excinfo:
        evaluate_guard_results(rows, results)

    message = str(excinfo.value)
    assert "query-side" in message
    assert "resource_abuse" in message
    assert "q-1" in message


def test_row_and_result_side_mismatch_raises() -> None:
    rows = [BenchmarkRow(id="q-1", text="a", label=0, side=Side.QUERY, domains=(), lang="en")]
    results = [
        GuardResult(
            side=Side.RESPONSE,
            scores=_scores_for(Side.RESPONSE),
            unsafe=False,
            decision=Decision.ALLOW,
            predicted_domain=None,
            max_risk=0.1,
            latency_ms=1.0,
            model="test",
        )
    ]

    with pytest.raises(ValueError) as excinfo:
        evaluate_guard_results(rows, results)

    message = str(excinfo.value)
    assert "q-1" in message
    assert "query-side" in message
    assert "response" in message


def test_complete_response_side_score_set_is_scored_with_its_own_domains() -> None:
    rows = [
        BenchmarkRow(
            id="r-1",
            text="a",
            label=1,
            side=Side.RESPONSE,
            domains=("sensitive_information_leakage",),
            lang="en",
        )
    ]
    results = [
        GuardResult(
            side=Side.RESPONSE,
            scores=_scores_for(Side.RESPONSE, sensitive_information_leakage=0.8),
            unsafe=True,
            decision=Decision.BLOCK,
            predicted_domain="sensitive_information_leakage",
            max_risk=0.8,
            latency_ms=1.0,
            model="test",
        )
    ]

    metrics = evaluate_guard_results(rows, results)

    assert metrics["positive_domain_accuracy"] == 1.0
    assert set(metrics["per_domain"]) == {"sensitive_information_leakage"}
    assert metrics["per_domain"]["sensitive_information_leakage"]["tp"] == 1
