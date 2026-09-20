from __future__ import annotations

from jevguard_nsfa.models import Decision, Side, ThresholdPolicy
from jevguard_nsfa.taxonomy import DOMAINS, QUERY_DOMAINS, RESPONSE_DOMAINS, questions_for


def test_taxonomy_has_seven_independent_l1_domains() -> None:
    assert len(DOMAINS) == 7
    assert len(QUERY_DOMAINS) == 5
    assert len(RESPONSE_DOMAINS) == 2
    assert len({domain.id for domain in DOMAINS}) == 7
    assert set(questions_for(Side.QUERY)) == {domain.id for domain in QUERY_DOMAINS}
    assert set(questions_for(Side.RESPONSE)) == {domain.id for domain in RESPONSE_DOMAINS}


def test_threshold_policy_separates_binary_label_from_review_band() -> None:
    policy = ThresholdPolicy(default_threshold=0.5, review_margin=0.1)
    assert policy.is_unsafe({"risk": 0.50})
    assert not policy.is_unsafe({"risk": 0.49})
    assert policy.decision({"risk": 0.30}) is Decision.ALLOW
    assert policy.decision({"risk": 0.45}) is Decision.REVIEW
    assert policy.decision({"risk": 0.65}) is Decision.BLOCK


def test_domain_threshold_override() -> None:
    policy = ThresholdPolicy(default_threshold=0.5, domain_thresholds={"a": 0.8})
    assert not policy.is_unsafe({"a": 0.7})
    assert policy.is_unsafe({"b": 0.7})
