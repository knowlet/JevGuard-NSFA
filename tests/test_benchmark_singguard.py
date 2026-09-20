"""Dependency-free regression tests for the SingGuard realtime benchmark harness.

CI installs only ``.[dev]``, so torch/vllm/datasets/transformers are unavailable here. Every
test replaces ``benchmark_singguard._lazy_runtime`` and the surrounding seams with stubs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from jevguard_nsfa import benchmark_jev as bench_jev
from jevguard_nsfa import benchmark_singguard as bench
from jevguard_nsfa import compare
from jevguard_nsfa.dataset import BenchmarkRow
from jevguard_nsfa.models import Side
from jevguard_nsfa.taxonomy import domains_for

# A real NSFA Level-1 query domain: results must score the whole side's taxonomy.
HEAD = "prompt_injection_and_jailbreak"


def _full_scores(value: float, side: Side = Side.QUERY) -> dict[str, float]:
    """A complete per-side score set, as ``_infer_batch`` produces for a full head set."""
    return {domain.id: value for domain in domains_for(side)}


# --------------------------------------------------------------------------- ISSUE 1


class ModernPoolerConfig:
    """vLLM main: `use_activation: bool | None = None`; `normalize` no longer exists."""

    def __init__(self, pooling_type: object = None, task: object = None, use_activation: object = None) -> None:
        self.pooling_type = pooling_type
        self.task = task
        self.use_activation = use_activation


class LegacyPoolerConfig:
    """vLLM <= 0.11: `normalize: Optional[bool] = None`, documented as defaulting to True."""

    def __init__(self, pooling_type: object = None, task: object = None, normalize: object = None) -> None:
        self.pooling_type = pooling_type
        self.task = task
        self.normalize = normalize


class PoolingTypeOnlyPoolerConfig:
    """Accepts `pooling_type` only: neither switch can be requested."""

    def __init__(self, pooling_type: object = None) -> None:
        self.pooling_type = pooling_type


class PermissivePoolerConfig:
    """Accepts and drops every keyword: the switches construct but are unverifiable."""

    def __init__(self, pooling_type: object = None, **_: object) -> None:
        self.pooling_type = pooling_type


class FakeLLM:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _engine_args(*, with_runner: bool) -> type:
    if with_runner:

        class EngineArgs:
            def __init__(self, model: object, runner: object = None, pooler_config: object = None) -> None:  # noqa: ARG002
                self.runner = runner
                self.pooler_config = pooler_config

    else:

        class EngineArgs:
            def __init__(  # noqa: ARG002
                self,
                model: object,
                task: object = None,
                override_pooler_config: object = None,
            ) -> None:
                self.task = task
                self.override_pooler_config = override_pooler_config

    return EngineArgs


def _fake_lazy_runtime(monkeypatch: pytest.MonkeyPatch, pooler_config: type, *, with_runner: bool) -> None:
    runtime = (None, None, None, None, FakeLLM, pooler_config, _engine_args(with_runner=with_runner))
    monkeypatch.setattr(bench, "_lazy_runtime", lambda: runtime)


def test_pooler_config_prefers_modern_use_activation_switch() -> None:
    config = bench._pooler_config(ModernPoolerConfig)
    assert config.pooling_type == "LAST"
    assert config.task == "embed"
    assert config.use_activation is False
    assert bench._pooler_is_activation_free(config) is True


def test_pooler_config_falls_back_to_legacy_normalize_switch() -> None:
    config = bench._pooler_config(LegacyPoolerConfig)
    assert config.pooling_type == "LAST"
    assert config.normalize is False
    assert bench._pooler_is_activation_free(config) is True


@pytest.mark.parametrize("pooler_config", [PoolingTypeOnlyPoolerConfig, PermissivePoolerConfig])
def test_pooler_config_raises_instead_of_silently_enabling_normalisation(pooler_config: type) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        bench._pooler_config(pooler_config, model="inclusionAI/SingGuard-NSFA-0.8B")
    message = str(excinfo.value)
    assert "activation-free" in message
    assert "inclusionAI/SingGuard-NSFA-0.8B" in message
    # Every attempted kwargs set must be reported with its failure reason.
    assert message.count("PoolerConfig(**") == len(bench._POOLER_ATTEMPTS)
    assert "use_activation" in message
    assert "normalize" in message


def test_make_llm_runner_branch_uses_verified_modern_pooler(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_lazy_runtime(monkeypatch, ModernPoolerConfig, with_runner=True)
    llm = bench._make_llm("fake/model", 8192, 0.9, 1, "auto")
    assert llm.kwargs["runner"] == "pooling"
    assert llm.kwargs["pooler_config"].use_activation is False
    assert "override_pooler_config" not in llm.kwargs
    assert "task" not in llm.kwargs


def test_make_llm_override_branch_uses_verified_legacy_pooler(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_lazy_runtime(monkeypatch, LegacyPoolerConfig, with_runner=False)
    llm = bench._make_llm("fake/model", 8192, 0.9, 1, "auto")
    assert llm.kwargs["task"] == "embed"
    assert llm.kwargs["override_pooler_config"].normalize is False
    assert "runner" not in llm.kwargs


@pytest.mark.parametrize("with_runner", [True, False])
def test_make_llm_propagates_unverifiable_pooler_failure(
    monkeypatch: pytest.MonkeyPatch, with_runner: bool
) -> None:
    _fake_lazy_runtime(monkeypatch, PoolingTypeOnlyPoolerConfig, with_runner=with_runner)
    with pytest.raises(RuntimeError, match="activation-free"):
        bench._make_llm("fake/model", 8192, 0.9, 1, "auto")


# ------------------------------------------------------------------ end-to-end harness


class _Vec:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def numel(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]


class _ProbMatrix:
    """Stands in for a softmaxed [num_heads, batch, 2] tensor."""

    def __init__(self, data: list[list[_Vec]]) -> None:
        self.data = data

    def __getitem__(self, key: tuple[int, int]) -> _Vec:
        head_index, sample_index = key
        return self.data[head_index][sample_index]

    def detach(self) -> _ProbMatrix:
        return self

    def cpu(self) -> _ProbMatrix:
        return self


class _FakeTensor:
    def __init__(self, data: list[list[float]]) -> None:
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)


def _fake_torch() -> SimpleNamespace:
    torch = SimpleNamespace()
    torch.float32 = "float32"
    torch.tensor = lambda data, dtype=None, device=None: _FakeTensor(data)  # noqa: ARG005
    torch.inference_mode = contextlib.nullcontext
    torch.softmax = lambda logits, dim=-1: logits  # noqa: ARG005
    torch.cuda = SimpleNamespace(is_available=lambda: False, get_device_name=lambda _index: "FakeGPU")
    return torch


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> _FakeTokenizer:  # noqa: ARG003
        return cls()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ANN001, ANN201, ARG002
        return "|".join(str(message["content"]) for message in messages)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        return list(range(min(len(str(text)), 40)))

    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001, ANN201, ARG002
        return f"decoded-{len(ids)}"


class _FakeRuntimeLLM:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.embed_calls: list[list[str]] = []
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))

    def embed(self, prompts, use_tqdm=False):  # noqa: ANN001, ANN201, ARG002
        self.embed_calls.append(list(prompts))
        return [SimpleNamespace(outputs=SimpleNamespace(embedding=[0.1, 0.2])) for _ in prompts]


class _CountingHeadForward:
    """Counts `_build_parallel_head_forward` invocations; returns an inert forward."""

    def __init__(self, probability: float = 0.3) -> None:
        self.probability = probability
        self.calls: list[list[object]] = []

    def __call__(self, modules):  # noqa: ANN001, ANN201
        self.calls.append(list(modules))
        head_count = len(modules)

        def forward(params, buffers, embeddings):  # noqa: ANN001, ANN201, ARG001
            batch = len(embeddings)
            return _ProbMatrix(
                [[_Vec([1.0 - self.probability, self.probability]) for _ in range(batch)] for _ in range(head_count)]
            )

        return forward, {"w": [0.0]}, {"b": [0.0]}


def _make_row(index: int, *, label: int = 1, side: Side = Side.QUERY, lang: str = "en") -> BenchmarkRow:
    return BenchmarkRow(
        id=f"row-{index}",
        text=f"sample text {index}",
        label=label,
        side=side,
        domains=(HEAD,) if label else (),
        lang=lang,
    )


def _make_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "dataset": "fake/dataset",
        "split": "train",
        "dataset_revision": None,
        "benchmark": "query",
        "language": None,
        "id_contains": None,
        "limit": 1,
        "seed": 42,
        "model": "fake/model",
        "heads_dir": None,
        "allow_partial_heads": False,
        "threshold": 0.5,
        "review_margin": 0.1,
        "batch_size": 1,
        "warmup": 0,
        "device": "cpu",
        "max_tokens": 8192,
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "gpu_hourly_usd": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _Harness:
    def __init__(self, builder: _CountingHeadForward, llm: _FakeRuntimeLLM) -> None:
        self.builder = builder
        self.llm = llm
        self.llm_requests: list[dict[str, object]] = []


def _fake_head(task: str = "query") -> dict[str, object]:
    return {"head": object(), "task": task, "max_tokens": 8192, "system_prompt": None}


def _head_map(
    side: Side = Side.QUERY,
    *,
    task: str | None = None,
    drop: tuple[str, ...] = (),
    extra: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """A complete NSFA Level-1 head set for ``side``, optionally perturbed.

    ``drop`` removes domains (a partial ``--heads-dir``), ``extra`` adds heads under an
    explicit task label (a mislabeled or hand-repacked head).
    """
    heads = {
        domain.id: _fake_head(task or side.value) for domain in domains_for(side) if domain.id not in drop
    }
    for name, head_task in (extra or {}).items():
        heads[name] = _fake_head(head_task)
    return heads


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[BenchmarkRow],
    *,
    heads: dict[str, dict[str, object]] | None = None,
) -> _Harness:
    builder = _CountingHeadForward()
    llm = _FakeRuntimeLLM()
    harness = _Harness(builder, llm)
    runtime = (_fake_torch(), None, None, _FakeTokenizer, _FakeRuntimeLLM, None, None)
    monkeypatch.setattr(bench, "_lazy_runtime", lambda: runtime)
    monkeypatch.setattr(bench, "iter_huggingface_rows", lambda **_: iter(rows))

    def fake_make_llm(*args: object, **kwargs: object) -> _FakeRuntimeLLM:
        harness.llm_requests.append(dict(kwargs))
        return llm

    monkeypatch.setattr(bench, "_make_llm", fake_make_llm)
    resolved = _head_map() if heads is None else heads
    monkeypatch.setattr(bench, "_load_heads", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(bench, "_build_parallel_head_forward", builder)
    return harness


# ------------------------------------------------------------------- ISSUE 2 coverage


def test_parallel_head_state_is_prepared_once_per_task(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(3)]
    harness = _install_fakes(monkeypatch, rows)

    report = bench.run_benchmark(_make_args(limit=3, batch_size=1, warmup=1))

    # The old implementation rebuilt the vmap'd state for the warmup plus every batch (4x).
    assert len(harness.builder.calls) == 1
    # Warmup + three single-sample batches still executed the real inference path.
    assert len(harness.llm.embed_calls) == 4
    cold_start = report["cold_start"]
    assert "classification_head_runner_seconds" in cold_start
    assert 0.0 <= cold_start["classification_head_runner_seconds"] <= cold_start["model_and_head_load_seconds"]


def test_head_runner_is_reused_across_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(6)]
    harness = _install_fakes(monkeypatch, rows)

    bench.run_benchmark(_make_args(limit=6, batch_size=2, warmup=2))

    assert len(harness.builder.calls) == 1
    assert len(harness.llm.embed_calls) == 4  # 1 warmup batch + 3 measurement batches


def test_prepare_head_runner_rejects_unknown_task() -> None:
    with pytest.raises(RuntimeError, match="No classification heads found"):
        bench._prepare_head_runner(
            {HEAD: {"head": object(), "task": "query", "max_tokens": 1, "system_prompt": None}},
            "response",
            "cpu",
        )


def test_runner_for_side_rejects_unprepared_task() -> None:
    with pytest.raises(RuntimeError, match="No classification heads found for task='response'"):
        bench._runner_for_side({}, Side.RESPONSE)


# --------------------------------------------------- ISSUE 2b: complete head set gate


def test_missing_query_head_fails_loudly_before_any_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(3)]
    harness = _install_fakes(monkeypatch, rows, heads=_head_map(drop=("resource_abuse",)))

    with pytest.raises(RuntimeError) as excinfo:
        bench.run_benchmark(_make_args(limit=3, warmup=1))

    message = str(excinfo.value)
    assert "query" in message
    assert "resource_abuse" in message
    assert "--heads-dir" in message
    # The gate runs before the model load, the warmup and every measurement.
    assert harness.llm_requests == []
    assert harness.llm.embed_calls == []


def test_missing_response_head_fails_a_response_run(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index, side=Side.RESPONSE) for index in range(3)]
    harness = _install_fakes(
        monkeypatch,
        rows,
        heads=_head_map(Side.RESPONSE, drop=("sensitive_information_leakage",)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        bench.run_benchmark(_make_args(benchmark="response", limit=3))

    message = str(excinfo.value)
    assert "response" in message
    assert "sensitive_information_leakage" in message
    assert "--heads-dir" in message
    assert harness.llm.embed_calls == []


def test_mislabeled_head_fails_as_unexpected_instead_of_being_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(3)]
    # A response-side domain whose task metadata claims it is a query head.
    heads = _head_map(extra={"hazardous_action_generation": "query"})
    harness = _install_fakes(monkeypatch, rows, heads=heads)

    with pytest.raises(RuntimeError) as excinfo:
        bench.run_benchmark(_make_args(limit=3))

    message = str(excinfo.value)
    assert "unexpected domains ['hazardous_action_generation']" in message
    assert "missing domains []" in message
    assert harness.llm.embed_calls == []


def test_fully_query_labeled_head_fails_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(0)]
    # Every query head was repacked under the response task, so nothing can be scored.
    _install_fakes(monkeypatch, rows, heads=_head_map(task="response"))

    with pytest.raises(RuntimeError, match="Incomplete SingGuard classification-head set"):
        bench.run_benchmark(_make_args(limit=1))


def test_only_the_benchmarked_side_needs_its_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(2)]
    heads = _head_map()
    heads.update({domain.id: _fake_head("response") for domain in domains_for(Side.RESPONSE)})
    harness = _install_fakes(monkeypatch, rows, heads=heads)

    report = bench.run_benchmark(_make_args(limit=2, warmup=1))

    query_domains = [domain.id for domain in domains_for(Side.QUERY)]
    assert report["head_manifest"]["complete"] is True
    assert report["head_manifest"]["side"] == "query"
    assert report["head_manifest"]["head_count"] == len(query_domains)
    # The runner stacks the benchmarked side only; the unused side is never prepared.
    assert harness.builder.calls
    assert all(len(call) == len(query_domains) for call in harness.builder.calls)


def test_allow_partial_heads_reports_a_non_baseline_run(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(2)]
    harness = _install_fakes(monkeypatch, rows, heads=_head_map(drop=("resource_abuse",)))

    report = bench.run_benchmark(_make_args(limit=2, warmup=0, batch_size=1, allow_partial_heads=True))

    assert report["head_manifest"] == {
        "side": "query",
        "expected_domains": [domain.id for domain in domains_for(Side.QUERY)],
        "loaded_domains": sorted(
            domain.id for domain in domains_for(Side.QUERY) if domain.id != "resource_abuse"
        ),
        "missing_domains": ["resource_abuse"],
        "unexpected_domains": [],
        "complete": False,
        "head_count": 4,
    }
    assert report["baseline_complete"] is False
    assert "--allow-partial-heads" in report["baseline_note"]
    assert "not a complete nsfa baseline" in report["baseline_note"].lower()
    # The missing head has no score, and an absent head is never reported as risk 0.0.
    assert report["quality"] is None
    assert "quality is reported as null" in report["baseline_note"]
    # Opting in really did measure something, and stays strict JSON.
    assert harness.llm.embed_calls
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False)


def test_partial_opt_in_still_refuses_a_side_without_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(0)]
    heads = {domain.id: _fake_head("response") for domain in domains_for(Side.RESPONSE)}
    _install_fakes(monkeypatch, rows, heads=heads)

    with pytest.raises(RuntimeError, match="No query-side SingGuard classification heads"):
        bench.run_benchmark(_make_args(limit=1, allow_partial_heads=True))


def test_main_from_args_warns_about_a_partial_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]  # noqa: ANN001
) -> None:
    rows = [_make_row(index) for index in range(2)]
    _install_fakes(monkeypatch, rows, heads=_head_map(drop=("resource_abuse",)))
    args = _make_args(limit=2, warmup=0, allow_partial_heads=True)
    args.output = tmp_path / "singguard.json"

    assert bench.main_from_args(args) == 0

    assert "partial" in capsys.readouterr().err.lower()
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["baseline_complete"] is False
    assert report["head_manifest"]["missing_domains"] == ["resource_abuse"]


def test_head_manifest_is_complete_and_ordered_for_a_full_head_set(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(2)]
    _install_fakes(monkeypatch, rows)

    report = bench.run_benchmark(_make_args(limit=2, warmup=0))

    assert report["head_manifest"] == {
        "side": "query",
        # Taxonomy order, not alphabetical order.
        "expected_domains": [domain.id for domain in domains_for(Side.QUERY)],
        "loaded_domains": sorted(domain.id for domain in domains_for(Side.QUERY)),
        "missing_domains": [],
        "unexpected_domains": [],
        "complete": True,
        "head_count": len(domains_for(Side.QUERY)),
    }
    assert report["baseline_complete"] is True


def test_add_arguments_registers_partial_heads_and_dataset_revision() -> None:
    parser = argparse.ArgumentParser()
    bench.add_arguments(parser)

    defaults = parser.parse_args([])
    assert defaults.allow_partial_heads is False
    assert defaults.dataset_revision is None

    parsed = parser.parse_args(["--allow-partial-heads", "--dataset-revision", "abc123"])
    assert parsed.allow_partial_heads is True
    assert parsed.dataset_revision == "abc123"


# ------------------------------------------------------------------- ISSUE 3 coverage


def test_latency_ms_is_request_latency_and_amortized_is_reported_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_make_row(index) for index in range(4)]
    _install_fakes(monkeypatch, rows)
    monkeypatch.setattr(bench, "_infer_batch", lambda **_: ([_full_scores(0.9) for _ in range(4)], 80.0))

    report = bench.run_benchmark(_make_args(limit=4, batch_size=4, warmup=0))

    assert report["parameters"]["batch_size"] == 4
    assert report["latency_scope"] == "request"
    assert report["latency_ms"]["p50"] == 80.0
    assert report["amortized_ms_per_sample"]["p50"] == 20.0
    assert report["batch_latency_ms"]["p50"] == 80.0
    assert report["latency_notes"]["pairing"]


def test_guard_result_latency_is_not_batch_amortized(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(index) for index in range(4)]
    _install_fakes(monkeypatch, rows)
    monkeypatch.setattr(bench, "_infer_batch", lambda **_: ([_full_scores(0.9) for _ in range(4)], 80.0))
    seen: list[float] = []

    class RecordingGuardResult(bench.GuardResult):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            seen.append(self.latency_ms)

    monkeypatch.setattr(bench, "GuardResult", RecordingGuardResult)
    bench.run_benchmark(_make_args(limit=4, batch_size=4, warmup=0))

    assert seen == [80.0, 80.0, 80.0, 80.0]


def test_report_round_trips_through_strict_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    rows = [_make_row(index, label=index % 2) for index in range(4)]
    _install_fakes(monkeypatch, rows)
    args = _make_args(limit=4, batch_size=2, warmup=0)
    args.output = tmp_path / "singguard.json"

    assert bench.main_from_args(args) == 0

    rendered = args.output.read_text(encoding="utf-8")
    report = json.loads(rendered)
    assert report["latency_scope"] == "request"
    assert report["parameters"]["batch_size"] == 2
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False)


def test_main_from_args_rejects_non_finite_report(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    args = _make_args(limit=1, batch_size=1, warmup=0)
    args.output = tmp_path / "singguard.json"
    monkeypatch.setattr(bench, "run_benchmark", lambda _: {"latency_ms": {"p50": float("nan")}})

    with pytest.raises(ValueError):
        bench.main_from_args(args)


# ------------------------------------------------------- alignment digest contract


def _singguard_report(monkeypatch: pytest.MonkeyPatch, rows: list[BenchmarkRow], **overrides: object) -> dict:
    _install_fakes(monkeypatch, rows)
    return bench.run_benchmark(_make_args(limit=len(rows), warmup=0, **overrides))


def _sample_rows() -> list[BenchmarkRow]:
    return [_make_row(index, label=index % 2, lang="zh" if index % 2 else "en") for index in range(5)]


def _jev_shaped_report(singguard: dict, **overrides: object) -> dict:
    """A Jev-shaped report that reuses the SingGuard run's selection metadata."""
    report = {
        "engine": "jevguard-nsfa",
        "mode": "managed-api-online",
        "model": "jev-test",
        "dataset": dict(singguard["dataset"]),
        "parameters": {"threshold": singguard["parameters"]["threshold"]},
        "samples": dict(singguard["samples"]),
        "latency_scope": "request",
        "quality": {"binary": {"f1": 0.5, "precision": 0.5, "recall": 0.5, "accuracy": 0.5, "brier": 0.25}},
        "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
        "throughput": {"successful_requests_per_second": 100.0},
        "usage": {"cost_per_1000_successful_requests_usd": 0.01},
    }
    report.update(overrides)
    return report


def test_report_carries_sha256_alignment_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _singguard_report(monkeypatch, _sample_rows(), batch_size=1)

    digests = {
        "dataset.fingerprint": report["dataset"]["fingerprint"],
        "samples.attempted_ids_sha256": report["samples"]["attempted_ids_sha256"],
        "samples.successful_sha256": report["samples"]["successful_sha256"],
    }
    for name, digest in digests.items():
        assert isinstance(digest, str), name
        assert len(digest) == 64, name
        assert all(character in "0123456789abcdef" for character in digest), name
    # Every attempted SingGuard row is scored, so the content fingerprints agree; the id-only
    # diagnostic digest is deliberately different because it does not cover the row content.
    assert digests["dataset.fingerprint"] == digests["samples.successful_sha256"]
    assert digests["samples.attempted_ids_sha256"] != digests["samples.successful_sha256"]
    assert "successful_ids_sha256" not in report["samples"]


def test_report_digests_are_byte_identical_to_benchmark_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _sample_rows()
    report = _singguard_report(monkeypatch, rows, batch_size=1)

    fingerprint = bench_jev._rows_fingerprint(rows)
    id_digest = bench_jev._rows_id_digest(rows)
    print(f"cross-engine dataset.fingerprint      = {fingerprint}")
    print(f"cross-engine samples.*_ids_sha256     = {id_digest}")

    assert bench._sha256_lines(["a", "b"]) == bench_jev._sha256_lines(["a", "b"])
    assert bench._rows_fingerprint(rows) == fingerprint
    assert bench._rows_id_digest(rows) == id_digest
    assert bench._sample_identity(rows[0]) == bench_jev._sample_identity(rows[0])
    assert report["dataset"]["fingerprint"] == fingerprint
    assert report["samples"]["attempted_ids_sha256"] == id_digest
    assert report["samples"]["successful_sha256"] == fingerprint


def test_singguard_report_aligns_with_the_comparator(monkeypatch: pytest.MonkeyPatch) -> None:
    singguard = _singguard_report(monkeypatch, _sample_rows(), batch_size=1)
    jev = _jev_shaped_report(singguard)

    checks = {check.name: check for check in compare.alignment_checks(jev, singguard)}
    assert checks["dataset.fingerprint"].status == "ok"
    assert checks["samples.successful_sha256"].status == "ok"
    assert compare.alignment(jev, singguard).data_ok is True

    markdown = compare.render_markdown(jev, singguard)
    fingerprint = singguard["dataset"]["fingerprint"]
    ids_digest = singguard["samples"]["successful_sha256"]
    assert f"| dataset.fingerprint | {fingerprint} | {fingerprint} | ok |" in markdown
    assert f"| samples.successful_sha256 | {ids_digest} | {ids_digest} | ok |" in markdown
    # The quality delta is no longer withheld now that the selection is provably shared.
    f1_line = next(line for line in markdown.splitlines() if line.startswith("| Binary F1"))
    assert not f1_line.endswith("| n/a |")


def test_comparator_reports_a_genuinely_different_selection_as_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _sample_rows()
    singguard = _singguard_report(monkeypatch, rows, batch_size=1)
    other_digest = bench_jev._rows_id_digest(rows)
    jev = _jev_shaped_report(singguard)
    jev["dataset"] = {**jev["dataset"], "fingerprint": other_digest}

    checks = {check.name: check for check in compare.alignment_checks(jev, singguard)}
    assert other_digest != singguard["dataset"]["fingerprint"]
    assert checks["dataset.fingerprint"].status == "mismatch"


# ------------------------------------------------- fingerprint content and sample scope


def test_fingerprint_covers_row_content_and_is_stable() -> None:
    row = _make_row(0)
    base = bench._rows_fingerprint([row])

    assert base == bench._rows_fingerprint([replace(row)])
    assert base != bench._rows_fingerprint([replace(row, id="row-999")])
    assert base != bench._rows_fingerprint([replace(row, text=row.text + " (revised)")])
    assert base != bench._rows_fingerprint([replace(row, label=1 - row.label)])
    assert base != bench._rows_fingerprint([replace(row, side=Side.RESPONSE)])
    assert base != bench._rows_fingerprint([replace(row, domains=("resource_abuse",))])
    assert base != bench._rows_fingerprint([replace(row, lang="zh")])
    # Order still matters: the fingerprint identifies an emitted selection.
    other = _make_row(1)
    assert bench._rows_fingerprint([row, other]) != bench._rows_fingerprint([other, row])


def test_successful_sha256_covers_only_the_rows_that_were_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _sample_rows()
    _install_fakes(monkeypatch, rows)
    scored: list[BenchmarkRow] = []

    def fake_infer_batch(*, rows: list[BenchmarkRow], **_: object):  # noqa: ANN202
        scored.extend(rows)
        return [_full_scores(0.9) for _ in rows], 12.0

    monkeypatch.setattr(bench, "_infer_batch", fake_infer_batch)

    report = bench.run_benchmark(_make_args(limit=len(rows), batch_size=2, warmup=0))

    assert report["samples"]["successful"] == len(scored) == len(rows)
    assert report["samples"]["successful_sha256"] == bench._rows_fingerprint(scored)
    assert report["samples"]["successful_sha256"] == report["dataset"]["fingerprint"]
    # The id-only digest survives as a diagnostic field, and is not the fingerprint.
    assert report["samples"]["attempted_ids_sha256"] == bench._rows_id_digest(rows)
    assert report["samples"]["attempted_ids_sha256"] != report["samples"]["successful_sha256"]


def test_dataset_revision_is_recorded_and_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_make_row(0)]
    _install_fakes(monkeypatch, rows)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        bench,
        "iter_huggingface_rows",
        lambda **kwargs: (seen.update(kwargs), iter(rows))[1],
    )

    report = bench.run_benchmark(_make_args(limit=1, dataset_revision="9f2c1ab"))
    assert seen["revision"] == "9f2c1ab"
    assert report["dataset"]["revision"] == "9f2c1ab"

    report = bench.run_benchmark(_make_args(limit=1, dataset_revision=None))
    assert seen["revision"] is None
    assert report["dataset"]["revision"] is None
