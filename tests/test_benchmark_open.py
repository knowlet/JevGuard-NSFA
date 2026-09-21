from __future__ import annotations

import argparse

import pytest

from jevguard_nsfa.benchmark_open import (
    _effective_limit,
    _guard_result_from_scores,
    _laya_scores,
    _rlcd_scores,
    _rlcd_true_probability,
    plain_questions_for,
    rlcd_schema_for,
)
from jevguard_nsfa.models import Side, ThresholdPolicy
from jevguard_nsfa.taxonomy import domains_for


def _domain_ids(side: Side) -> set[str]:
    return {domain.id for domain in domains_for(side)}


def _scores(side: Side, value: float = 0.1) -> dict[str, float]:
    return {domain.id: value for domain in domains_for(side)}


@pytest.mark.parametrize("side", [Side.QUERY, Side.RESPONSE])
def test_plain_systemone_questions_cover_every_nsfa_domain(side: Side) -> None:
    questions = plain_questions_for(side)
    assert set(questions) == _domain_ids(side)
    for question in questions.values():
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}


@pytest.mark.parametrize("side", [Side.QUERY, Side.RESPONSE])
def test_rlcd_schema_covers_every_nsfa_domain(side: Side) -> None:
    schema = rlcd_schema_for(side)
    assert set(schema) == _domain_ids(side)
    assert all(field["type"] == "boolean" for field in schema.values())


def test_laya_scores_reads_noul_probabilities() -> None:
    response = {
        "answers": {
            domain.id: {"noul": 0.75 if index == 0 else 0.1}
            for index, domain in enumerate(domains_for(Side.QUERY))
        }
    }
    scores = _laya_scores(response, Side.QUERY)
    assert set(scores) == _domain_ids(Side.QUERY)
    assert max(scores.values()) == 0.75


def test_laya_scores_rejects_missing_domain() -> None:
    with pytest.raises(ValueError, match="missing Noul answer"):
        _laya_scores({"answers": {}}, Side.RESPONSE)


def test_rlcd_true_probability_prefers_full_distribution() -> None:
    field = {
        "value": False,
        "confidence": 0.9,
        "top_choices": [
            {"choice": "false", "probability": 0.7},
            {"choice": "true", "probability": 0.3},
        ],
    }
    assert _rlcd_true_probability(field) == pytest.approx(0.3)


def test_rlcd_true_probability_complements_false_winner() -> None:
    assert _rlcd_true_probability({"value": False, "confidence": 0.8}) == pytest.approx(0.2)
    assert _rlcd_true_probability({"value": True, "confidence": 0.8}) == pytest.approx(0.8)


def test_rlcd_scores_reads_field_telemetry() -> None:
    response = {
        "field_telemetry": {
            domain.id: {
                "top_choices": [
                    {"choice": "false", "probability": 0.8},
                    {"choice": "true", "probability": 0.2},
                ]
            }
            for domain in domains_for(Side.RESPONSE)
        }
    }
    scores = _rlcd_scores(response, Side.RESPONSE)
    assert scores == {domain.id: pytest.approx(0.2) for domain in domains_for(Side.RESPONSE)}


def test_rlcd_scores_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing field"):
        _rlcd_scores({"field_telemetry": {}}, Side.QUERY)


def test_guard_result_requires_complete_score_vector() -> None:
    scores = _scores(Side.QUERY)
    scores.pop(next(iter(scores)))
    with pytest.raises(ValueError, match="incomplete NSFA score set"):
        _guard_result_from_scores(
            side=Side.QUERY,
            scores=scores,
            policy=ThresholdPolicy(),
            latency_ms=1.0,
            model="test",
        )


def test_guard_result_rejects_out_of_range_probability() -> None:
    scores = _scores(Side.RESPONSE)
    first = next(iter(scores))
    scores[first] = 1.1
    with pytest.raises(ValueError, match="out-of-range"):
        _guard_result_from_scores(
            side=Side.RESPONSE,
            scores=scores,
            policy=ThresholdPolicy(),
            latency_ms=1.0,
            model="test",
        )


def test_guard_result_maps_valid_scores_to_binary_decision() -> None:
    scores = _scores(Side.QUERY)
    first = next(iter(scores))
    scores[first] = 0.9
    result = _guard_result_from_scores(
        side=Side.QUERY,
        scores=scores,
        policy=ThresholdPolicy(default_threshold=0.5),
        latency_ms=3.0,
        model="test",
    )
    assert result.unsafe is True
    assert result.predicted_domain == first
    assert result.max_risk == pytest.approx(0.9)


def test_full_mode_ignores_limit() -> None:
    assert _effective_limit(argparse.Namespace(full=True, limit=100)) is None
    assert _effective_limit(argparse.Namespace(full=False, limit=500)) == 500
