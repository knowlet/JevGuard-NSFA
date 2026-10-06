from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from jevguard_nsfa.benchmark_open import (
    LayaBackend,
    RLCDHTTPBackend,
    SystemOneHTTPBackend,
    _effective_limit,
    _guard_result_from_scores,
    _laya_scores,
    _rlcd_scores,
    _rlcd_true_probability,
    plain_questions_for,
    rlcd_schema_for,
)
from jevguard_nsfa.dataset import BenchmarkRow
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



def _row(side: Side = Side.QUERY, *, lang: str = "en") -> BenchmarkRow:
    return BenchmarkRow(
        id="row-1",
        text="untrusted benchmark text",
        label=1,
        side=side,
        domains=(next(iter(_domain_ids(side))),),
        lang=lang,
    )


def test_systemone_http_backend_uses_complete_noul_response(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[object] = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.request = None
            created.append(self)

        def system_one(self, **request):
            self.request = request
            answers = {
                domain.id: SimpleNamespace(noul=0.9 if index == 0 else 0.1)
                for index, domain in enumerate(domains_for(Side.QUERY))
            }
            return SimpleNamespace(
                nouls=answers,
                model="fake-systemone",
                usage=SimpleNamespace(input_tokens=12, output_tokens=0),
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr("jevguard_nsfa.benchmark_open.TypeSafeClient", FakeClient)
    backend = SystemOneHTTPBackend(
        engine="kev-4b",
        base_url="http://127.0.0.1:8009",
        model="kev-latest",
        api_key="local",
        timeout=5.0,
    )
    result = backend.screen(_row(), ThresholdPolicy())
    assert result.model == "fake-systemone"
    assert set(result.scores) == _domain_ids(Side.QUERY)
    assert result.max_risk == pytest.approx(0.9)
    client = created[0]
    assert client.request["state"] == {"untrusted_text": "untrusted benchmark text"}
    assert set(client.request["questions"]) == _domain_ids(Side.QUERY)
    backend.close()
    assert client.closed is True


def test_laya_routed_backend_preloads_and_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRouter:
        def __init__(self, *, device=None, max_loaded=1):
            self.device = device
            self.max_loaded = max_loaded
            self.preloaded = []

        def preload(self, names):
            self.preloaded = list(names)

        def predict(self, state, questions, *, lang=None):
            assert state == {"untrusted_text": "untrusted benchmark text"}
            assert lang == "fr"
            return {
                "answers": {name: {"noul": 0.2} for name in questions},
                "routing": {"model": "multilingual", "repo": "convaiinnovations/laya/multilingual"},
            }

    fake_laya = SimpleNamespace(__version__="test", Router=FakeRouter, load=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "laya", fake_laya)

    backend = LayaBackend(model="routed", device="cuda")
    assert backend.runtime.preloaded == ["english", "multilingual"]
    result = backend.screen(_row(lang="fr"), ThresholdPolicy())
    assert set(result.scores) == _domain_ids(Side.QUERY)
    assert result.model == "convaiinnovations/laya/multilingual"


def test_laya_explicit_multilingual_uses_upstream_subfolder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeAgent:
        def predict(self, state, questions):
            return {"answers": {name: {"noul": 0.1} for name in questions}}

    def fake_load(repo, **kwargs):
        calls.append((repo, kwargs))
        return FakeAgent()

    fake_laya = SimpleNamespace(__version__="test", Router=object, load=fake_load)
    monkeypatch.setitem(sys.modules, "laya", fake_laya)

    backend = LayaBackend(model="multilingual", device="cuda")
    backend.screen(_row(), ThresholdPolicy())
    assert calls == [
        (
            "convaiinnovations/laya",
            {"device": "cuda", "subfolder": "multilingual"},
        )
    ]


def test_rlcd_http_backend_sends_boolean_schema_and_parses_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            payload = {
                "field_telemetry": {
                    domain.id: {
                        "top_choices": [
                            {"choice": "false", "probability": 0.75},
                            {"choice": "true", "probability": 0.25},
                        ]
                    }
                    for domain in domains_for(Side.RESPONSE)
                }
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse()

    monkeypatch.setattr("jevguard_nsfa.benchmark_open.urllib.request.urlopen", fake_urlopen)
    backend = RLCDHTTPBackend(
        engine="qwen-rlcd",
        base_url="http://127.0.0.1:8000/",
        model="harshatheg/Qwen-2.5-1B-RLCD",
        timeout=7.0,
        temperature=1.0,
    )
    result = backend.screen(_row(Side.RESPONSE), ThresholdPolicy())
    assert captured["url"] == "http://127.0.0.1:8000/api/run-parallel"
    assert captured["timeout"] == 7.0
    assert set(captured["payload"]["schema_def"]) == _domain_ids(Side.RESPONSE)
    assert all(
        field["type"] == "boolean"
        for field in captured["payload"]["schema_def"].values()
    )
    assert set(result.scores) == _domain_ids(Side.RESPONSE)
    assert all(value == pytest.approx(0.25) for value in result.scores.values())
