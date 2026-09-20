from __future__ import annotations

from types import SimpleNamespace

import pytest

from jevguard_nsfa.guard import JevGuard
from jevguard_nsfa.models import Decision, Side, ThresholdPolicy
from jevguard_nsfa.taxonomy import questions_for


class FakeClient:
    def __init__(self, score: float) -> None:
        self.score = score

    def system_one(self, *, state: object, questions: dict[str, object]) -> object:
        assert isinstance(state, dict)
        return SimpleNamespace(
            model="jev-test",
            usage=SimpleNamespace(input_tokens=123, output_tokens=0),
            nouls={
                name: SimpleNamespace(noul=self.score)
                for name in questions
            },
        )


class FailingClient:
    def system_one(self, **_: object) -> object:
        raise RuntimeError("transport failure")


def test_guard_returns_all_side_specific_scores_and_usage() -> None:
    guard = JevGuard(client=FakeClient(0.2))
    result = guard.screen("hello", Side.QUERY)
    assert set(result.scores) == set(questions_for(Side.QUERY))
    assert result.model == "jev-test"
    assert result.input_tokens == 123
    assert result.output_tokens == 0
    assert result.decision is Decision.ALLOW


def test_semantic_review_can_fallback() -> None:
    guard = JevGuard(
        client=FakeClient(0.51),
        policy=ThresholdPolicy(default_threshold=0.5, review_margin=0.1),
    )
    calls: list[str] = []

    def fallback(text: str, side: Side, result: object) -> str:
        calls.append(text)
        assert side is Side.QUERY
        return "system-two"

    assert guard.screen_or_fallback("ambiguous", Side.QUERY, fallback) == "system-two"
    assert calls == ["ambiguous"]


def test_transport_failure_never_triggers_semantic_fallback() -> None:
    guard = JevGuard(client=FailingClient())
    called = False

    def fallback(*_: object) -> str:
        nonlocal called
        called = True
        return "should-not-run"

    with pytest.raises(RuntimeError, match="transport failure"):
        guard.screen_or_fallback("anything", Side.QUERY, fallback)
    assert called is False
