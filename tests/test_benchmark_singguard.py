"""Dependency-free regression tests for the SingGuard realtime benchmark harness.

CI installs only ``.[dev]``, so torch/vllm/datasets/transformers are unavailable here. Every
test replaces ``benchmark_singguard._lazy_runtime`` and the surrounding seams with stubs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from types import SimpleNamespace

import pytest

from jevguard_nsfa import benchmark_jev as bench_jev
from jevguard_nsfa import benchmark_singguard as bench
from jevguard_nsfa import compare
from jevguard_nsfa.dataset import BenchmarkRow
from jevguard_nsfa.models import Side

HEAD = "jailbreak"


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
        "benchmark": "query",
        "language": None,
        "id_contains": None,
        "limit": 1,
        "seed": 42,
        "model": "fake/model",
        "heads_dir": None,
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


def _install_fakes(monkeypatch: pytest.MonkeyPatch, rows: list[BenchmarkRow]) -> _Harness:
    builder = _CountingHeadForward()
    llm = _FakeRuntimeLLM()
    runtime = (_fake_torch(), None, None, _FakeTokenizer, _FakeRuntimeLLM, None, None)
    monkeypatch.setattr(bench, "_lazy_runtime", lambda: runtime)
    monkeypatch.setattr(bench, "iter_huggingface_rows", lambda **_: iter(rows))
    monkeypatch.setattr(bench, "_make_llm", lambda *args, **kwargs: llm)
    monkeypatch.setattr(
        bench,
        "_load_heads",
        lambda *args, **kwargs: {
            HEAD: {"head": object(), "task": "query", "max_tokens": 8192, "system_prompt": None}
        },
    )
    monkeypatch.setattr(bench, "_build_parallel_head_forward", builder)
    return _Harness(builder, llm)


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


# ------------------------------------------------------------------- ISSUE 3 coverage


def test_latency_ms_is_request_latency_and_amortized_is_reported_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_make_row(index) for index in range(4)]
    _install_fakes(monkeypatch, rows)
    monkeypatch.setattr(bench, "_infer_batch", lambda **_: ([{HEAD: 0.9} for _ in range(4)], 80.0))

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
    monkeypatch.setattr(bench, "_infer_batch", lambda **_: ([{HEAD: 0.9} for _ in range(4)], 80.0))
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
        "samples.successful_ids_sha256": report["samples"]["successful_ids_sha256"],
    }
    for name, digest in digests.items():
        assert isinstance(digest, str), name
        assert len(digest) == 64, name
        assert all(character in "0123456789abcdef" for character in digest), name
    # Every attempted SingGuard row is scored today, so the two id digests agree.
    assert digests["samples.attempted_ids_sha256"] == digests["samples.successful_ids_sha256"]


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
    assert report["dataset"]["fingerprint"] == fingerprint
    assert report["samples"]["attempted_ids_sha256"] == id_digest
    assert report["samples"]["successful_ids_sha256"] == id_digest


def test_singguard_report_aligns_with_the_comparator(monkeypatch: pytest.MonkeyPatch) -> None:
    singguard = _singguard_report(monkeypatch, _sample_rows(), batch_size=1)
    jev = _jev_shaped_report(singguard)

    checks = {check.name: check for check in compare.alignment_checks(jev, singguard)}
    assert checks["dataset.fingerprint"].status == "ok"
    assert checks["samples.successful_ids_sha256"].status == "ok"
    assert compare.alignment(jev, singguard).data_ok is True

    markdown = compare.render_markdown(jev, singguard)
    fingerprint = singguard["dataset"]["fingerprint"]
    ids_digest = singguard["samples"]["successful_ids_sha256"]
    assert f"| dataset.fingerprint | {fingerprint} | {fingerprint} | ok |" in markdown
    assert f"| samples.successful_ids_sha256 | {ids_digest} | {ids_digest} | ok |" in markdown
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
