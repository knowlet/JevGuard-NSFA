"""Local real-time classification benchmark for the original SingGuard-NSFA."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .dataset import BenchmarkRow, canonical_domain, iter_huggingface_rows
from .metrics import evaluate_guard_results, latency_summary
from .models import GuardResult, Side, ThresholdPolicy
from .taxonomy import domains_for


def _sha256_lines(lines: Iterable[str]) -> str:
    """SHA-256 hex digest of ``"\n".join(lines)`` encoded as UTF-8.

    Byte-identical to ``benchmark_jev._sha256_lines`` so the digests of a shared sample
    selection match across engines and the comparator can classify the runs as aligned.
    """
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _sample_identity(row: BenchmarkRow) -> str:
    """Canonical JSON identity of one sample, byte-identical to ``benchmark_jev._sample_identity``.

    The fingerprint must cover everything that changes what was measured, not just the row
    id: two runs over the same ids with different texts, labels, sides, L1 domains or
    languages are different selections and must not be reported as aligned.
    """
    return json.dumps(
        [row.id, row.text, row.label, row.side.value, list(row.domains), row.lang],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _rows_fingerprint(rows: Sequence[BenchmarkRow]) -> str:
    """Identify the attempted sample selection in emitted order."""
    return _sha256_lines(_sample_identity(row) for row in rows)


def _rows_id_digest(rows: Sequence[BenchmarkRow]) -> str:
    """Identify the row ids of a sample subset in emitted order."""
    return _sha256_lines(row.id for row in rows)


def _lazy_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.config import PoolerConfig
        from vllm.engine.arg_utils import EngineArgs
    except ImportError as exc:
        raise RuntimeError("Install the SingGuard benchmark dependencies: pip install -e '.[singguard]'") from exc
    return torch, nn, snapshot_download, AutoTokenizer, LLM, PoolerConfig, EngineArgs


def _head_class(nn: Any) -> type:
    class EmbeddingHead(nn.Module):
        def __init__(
            self,
            input_size: int,
            num_classes: int = 2,
            hidden_dims: list[int] | None = None,
            dropout_rate: float = 0.3,
            use_layer_norm: bool = True,
            activation: str = "relu",
            **_: Any,
        ) -> None:
            super().__init__()
            activations = {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "silu": nn.SiLU,
                "tanh": nn.Tanh,
            }
            try:
                activation_cls = activations[activation.lower()]
            except KeyError as exc:
                raise ValueError(f"Unsupported head activation {activation!r}") from exc

            dims = [input_size, *(hidden_dims or [])]
            self.layers = nn.ModuleList()
            for source, target in zip(dims, dims[1:], strict=False):
                modules: list[Any] = [nn.Linear(source, target)]
                if use_layer_norm:
                    modules.append(nn.LayerNorm(target))
                modules.extend([activation_cls(), nn.Dropout(dropout_rate)])
                self.layers.append(nn.Sequential(*modules))
            self.output_layer = nn.Linear(dims[-1], num_classes)

        def forward(self, value: Any) -> Any:
            for layer in self.layers:
                value = layer(value)
            return self.output_layer(value)

    return EmbeddingHead


def _snapshot_revision(path: str | Path) -> str | None:
    name = Path(path).name
    return name if re.fullmatch(r"[0-9a-f]{40}", name) else None


def _resolve_model_snapshot(
    model: str,
    revision: str | None,
    snapshot_download: Any,
) -> tuple[Path, str | None]:
    """Resolve one model snapshot shared by heads, tokenizer, and vLLM."""
    local_model = Path(model)
    if local_model.exists() or snapshot_download is None:
        # A caller-supplied local snapshot cannot expose the Hub commit through the
        # filesystem name. When the operator supplies --model-revision, retain it as
        # the verified artifact identity recorded in the benchmark report.
        return local_model, _snapshot_revision(local_model) or revision

    kwargs: dict[str, Any] = {"repo_id": model}
    if revision is not None:
        kwargs["revision"] = revision
    root = Path(snapshot_download(**kwargs))
    return root, _snapshot_revision(root) or revision


def _resolve_heads_dir(model: str, explicit: str | None, snapshot_download: Any) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        local_model = Path(model)
        if local_model.exists():
            path = local_model / "nsfa_heads"
        else:
            root = Path(snapshot_download(repo_id=model, allow_patterns=["nsfa_heads/*.pth"]))
            path = root / "nsfa_heads"
    if not path.exists():
        raise FileNotFoundError(f"Classification-head directory not found: {path}")
    return path


def _load_heads(model: str, heads_dir: str | None, device: str) -> dict[str, dict[str, Any]]:
    torch, nn, snapshot_download, _, _, _, _ = _lazy_runtime()
    head_type = _head_class(nn)
    resolved = _resolve_heads_dir(model, heads_dir, snapshot_download)

    heads: dict[str, dict[str, Any]] = {}
    for path in sorted(resolved.glob("*.pth")):
        try:
            # These files may come from a remote model repository. Restricted loading
            # must happen on CPU before validating the checkpoint schema and moving
            # tensors to the selected device.
            payload = torch.load(path, weights_only=True, map_location="cpu")
        except TypeError as exc:
            raise RuntimeError(
                "Safe SingGuard checkpoint loading requires torch.load(weights_only=True); "
                "refusing unrestricted pickle loading"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Could not safely load SingGuard checkpoint {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"SingGuard checkpoint {path} must contain a mapping payload")
        if "head_state_dict" not in payload:
            continue
        config_payload = payload.get("head_config")
        state_dict = payload.get("head_state_dict")
        if not isinstance(config_payload, Mapping) or not isinstance(state_dict, Mapping):
            raise ValueError(f"SingGuard checkpoint {path} has an invalid head schema")
        config = dict(config_payload)
        allowed = {
            "input_size",
            "num_classes",
            "hidden_dims",
            "dropout_rate",
            "use_layer_norm",
            "activation",
            "label_smoothing",
            "class_weight",
        }
        head = head_type(**{key: value for key, value in config.items() if key in allowed})
        head.load_state_dict(state_dict)
        head.eval().to(dtype=torch.float32, device=device)
        if "sub_task_name" not in payload or "task" not in payload:
            raise ValueError(f"SingGuard checkpoint {path} is missing task metadata")
        raw_name = str(payload["sub_task_name"])
        name = canonical_domain(raw_name)
        if name is None:
            raise ValueError(f"Unknown SingGuard NSFA head name: {raw_name!r}")
        if name in heads:
            raise ValueError(f"Duplicate SingGuard NSFA head for domain {name!r}")
        heads[name] = {
            "head": head,
            "task": str(payload["task"]).lower(),
            "max_tokens": int(payload.get("max_tokens", 8192)),
            "system_prompt": payload.get("system_prompt"),
        }

    if not heads:
        raise RuntimeError(f"No valid NSFA classification heads found in {resolved}")
    return heads


_POOLER_ATTEMPTS: tuple[dict[str, Any], ...] = (
    # Modern vLLM (main): `use_activation` replaced the old `normalize` flag.
    {"pooling_type": "LAST", "task": "embed", "use_activation": False},
    {"pooling_type": "LAST", "use_activation": False},
    # Legacy vLLM (<= 0.11): `normalize: Optional[bool] = None` and defaults to True.
    {"pooling_type": "LAST", "task": "embed", "normalize": False},
    {"pooling_type": "LAST", "normalize": False},
)

_POOLER_SWITCH_FIELDS: tuple[str, ...] = ("use_activation", "normalize")


def _pooler_switches(config: Any) -> dict[str, Any]:
    return {field: getattr(config, field, "<unset>") for field in _POOLER_SWITCH_FIELDS}


def _pooler_is_activation_free(config: Any) -> bool:
    """Accept a config only when it *explicitly* disables pooling activation/normalisation.

    vLLM documents both `use_activation` and the legacy `normalize` as defaulting to True;
    a missing or `None` switch therefore means normalisation is enabled.
    """
    return any(getattr(config, field, None) is False for field in _POOLER_SWITCH_FIELDS)


def _pooler_config(PoolerConfig: Any, *, model: str | None = None) -> Any:
    """Build a vLLM PoolerConfig that provably disables embedding normalisation.

    The SingGuard NSFA classification heads consume the raw LAST-token hidden state, so a
    silently normalised pooler would change the input distribution the heads were trained
    on. A candidate that merely *constructs* is not enough: the resulting object must carry
    an explicitly disabled switch, otherwise the constructor's default re-enables it.
    """
    failures: list[str] = []
    for kwargs in _POOLER_ATTEMPTS:
        try:
            config = PoolerConfig(**kwargs)
        except (TypeError, ValueError) as exc:
            failures.append(f"  PoolerConfig(**{kwargs!r}) raised {type(exc).__name__}: {exc}")
            continue
        if _pooler_is_activation_free(config):
            return config
        failures.append(
            f"  PoolerConfig(**{kwargs!r}) constructed but left activation enabled: {_pooler_switches(config)}"
        )

    subject = f"model {model!r}" if model else "the configured model"
    raise RuntimeError(
        f"Could not build an activation-free vLLM PoolerConfig for {subject}. "
        "The SingGuard NSFA realtime benchmark feeds raw, un-normalised LAST-token embeddings "
        "to the classification heads, so a PoolerConfig with `use_activation`/`normalize` left "
        "unset (vLLM defaults those to True) -- or a bare PoolerConfig() -- would silently change "
        "the head input distribution. Attempts:\n" + "\n".join(failures)
    )


def _make_llm(
    model: str,
    max_tokens: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    dtype: str,
) -> Any:
    _, _, _, _, LLM, PoolerConfig, EngineArgs = _lazy_runtime()

    kwargs: dict[str, Any] = {
        "model": model,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_tokens,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "disable_log_stats": True,
    }
    if "runner" in inspect.signature(EngineArgs.__init__).parameters:
        kwargs["runner"] = "pooling"
        kwargs["pooler_config"] = _pooler_config(PoolerConfig, model=model)
    else:
        kwargs["task"] = "embed"
        kwargs["override_pooler_config"] = _pooler_config(PoolerConfig, model=model)
    return LLM(**kwargs)


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _prepare_prompt(text: str, side: Side, tokenizer: Any, max_tokens: int, system_prompt: str | None) -> str:
    calibration = "This is a test string"
    tag = "untrusted_input" if side is Side.QUERY else "untrusted_output"

    def wrap(value: str) -> str:
        return f"<{tag}>\n{value}\n</{tag}>"

    calibration_messages: list[dict[str, str]] = []
    if system_prompt:
        calibration_messages.append({"role": "system", "content": system_prompt})
    calibration_messages.append({"role": "user", "content": wrap(calibration)})
    rendered = tokenizer.apply_chat_template(
        calibration_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    overhead = max(
        len(tokenizer.encode(rendered, add_special_tokens=False))
        - len(tokenizer.encode(calibration, add_special_tokens=False)),
        0,
    )
    token_budget = max(max_tokens - overhead - 200, 1)
    escaped = _xml_escape(str(text))
    token_ids = tokenizer.encode(escaped, add_special_tokens=False)
    if len(token_ids) > token_budget:
        escaped = tokenizer.decode(token_ids[-token_budget:], skip_special_tokens=True)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": wrap(escaped)})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_parallel_head_forward(head_modules: list[Any]) -> Any:
    from torch.func import functional_call, stack_module_state, vmap

    params, buffers = stack_module_state(head_modules)
    meta_model = copy.deepcopy(head_modules[0]).to("meta")

    def one(p: Any, b: Any, data: Any) -> Any:
        return functional_call(meta_model, (p, b), (data,))

    return vmap(one, in_dims=(0, 0, None)), params, buffers


@dataclass(frozen=True)
class _HeadRunner:
    """Reusable parallel-head execution state for one detection side.

    Built once per task by :func:`_prepare_head_runner`, outside the warmup and the
    timed measurement loop, so the one-off `torch.func` setup is not charged to
    steady-state per-request latency.
    """

    task: str
    names: tuple[str, ...]
    forward: Any
    params: Any
    buffers: Any
    max_tokens: int
    system_prompt: str | None


def _prepare_head_runner(heads: dict[str, dict[str, Any]], task: str, device: str) -> _HeadRunner:
    """Stack the task's heads and build the vmap'd forward exactly once.

    `device` mirrors the signature of `_load_heads`, which has already materialised every
    head on it; the stacked parameters/buffers inherit that placement.
    """
    matching = _heads_for_task(heads, task)
    if not matching:
        raise RuntimeError(f"No classification heads found for task={task!r}")
    names = tuple(sorted(matching))
    exemplar = matching[names[0]]
    forward, params, buffers = _build_parallel_head_forward([matching[name]["head"] for name in names])
    return _HeadRunner(
        task=task,
        names=names,
        forward=forward,
        params=params,
        buffers=buffers,
        max_tokens=int(exemplar["max_tokens"]),
        system_prompt=exemplar["system_prompt"],
    )


def _runner_for_side(runners: dict[str, _HeadRunner], side: Side) -> _HeadRunner:
    try:
        return runners[side.value]
    except KeyError as exc:
        raise RuntimeError(f"No classification heads found for task={side.value!r}") from exc


def _heads_for_task(heads: Mapping[str, dict[str, Any]], task: str) -> dict[str, dict[str, Any]]:
    """The heads ``_prepare_head_runner`` would stack for ``task`` (same filter, same order)."""
    return {name: info for name, info in heads.items() if info["task"] == task}


def _head_manifest(side: Side, heads: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Describe the head set that will actually run for ``side``.

    ``--heads-dir`` is a plain directory anyone can repack, and a single missing ``.pth``
    used to degrade silently: the domain never ran, yet ``metrics.evaluate_guard_results``
    reported it as a ``0.0`` risk probability (``fn == positives``). The manifest is the
    source of truth for the completeness gate and is always recorded in the report.
    """
    expected = tuple(domain.id for domain in domains_for(side))
    loaded = tuple(sorted(_heads_for_task(heads, side.value)))
    missing = tuple(sorted(set(expected) - set(loaded)))
    unexpected = tuple(sorted(set(loaded) - set(expected)))
    return {
        "side": side.value,
        "expected_domains": list(expected),
        "loaded_domains": list(loaded),
        "missing_domains": list(missing),
        "unexpected_domains": list(unexpected),
        "complete": not missing and not unexpected,
        "head_count": len(loaded),
    }


def _require_complete_head_set(
    manifest: Mapping[str, Any],
    *,
    side: Side,
    heads_dir: str | None,
    allow_partial: bool,
) -> None:
    """Refuse to benchmark a partial head set unless the caller explicitly opted in.

    Runs before warmup and before any measurement, so an incomplete set can never produce a
    report that looks like a complete NSFA baseline.
    """
    if manifest["complete"] or allow_partial:
        return
    raise RuntimeError(
        f"Incomplete SingGuard classification-head set for the {side.value}-side benchmark: "
        f"missing domains {manifest['missing_domains']!r}, unexpected domains "
        f"{manifest['unexpected_domains']!r} (loaded: {manifest['loaded_domains']!r}). "
        f"Point --heads-dir at a directory holding every NSFA Level-1 {side.value}-side head "
        f"({manifest['expected_domains']!r}); currently checked {heads_dir or 'the model snapshot'!r}. "
        "A domain without a head would be scored as a 0.0 risk probability, so the run is "
        "refused instead of silently reported. Pass --allow-partial-heads to measure a "
        "partial head set as an explicitly non-baseline run."
    )


def _infer_batch(
    *,
    llm: Any,
    runner: _HeadRunner,
    tokenizer: Any,
    rows: list[BenchmarkRow],
    device: str,
    model_max_tokens: int,
) -> tuple[list[dict[str, float]], float]:
    torch, _, _, _, _, _, _ = _lazy_runtime()
    if not rows:
        return [], 0.0
    side = rows[0].side
    if any(row.side is not side for row in rows):
        raise ValueError("A SingGuard batch may contain only one detection side")
    if side.value != runner.task:
        raise RuntimeError(f"Prepared head runner is for task={runner.task!r} but the batch is {side.value!r}")

    effective_max = min(model_max_tokens, runner.max_tokens)
    started = perf_counter()
    prompts = [
        _prepare_prompt(row.text, side, tokenizer, effective_max, runner.system_prompt)
        for row in rows
    ]

    outputs = llm.embed(prompts, use_tqdm=False)
    embeddings = torch.tensor(
        [output.outputs.embedding for output in outputs],
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        logits = runner.forward(runner.params, runner.buffers, embeddings)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
    elapsed_ms = (perf_counter() - started) * 1000.0

    scores: list[dict[str, float]] = []
    for sample_index in range(len(rows)):
        item: dict[str, float] = {}
        for head_index, name in enumerate(runner.names):
            row_probs = probabilities[head_index, sample_index]
            if row_probs.numel() < 2:
                raise RuntimeError(f"Expected binary head probabilities for {name!r}")
            item[name] = float(row_probs[1])
        scores.append(item)
    return scores, elapsed_ms


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_side = Side.RESPONSE if args.benchmark == "response" else Side.QUERY
    dataset_revision = getattr(args, "dataset_revision", None)
    model_revision = getattr(args, "model_revision", None)
    allow_partial_heads = bool(getattr(args, "allow_partial_heads", False))
    languages = set(args.language) if args.language else None
    rows = list(
        iter_huggingface_rows(
            dataset_name=args.dataset,
            split=args.split,
            benchmark=args.benchmark,
            side=benchmark_side,
            languages=languages,
            id_contains=args.id_contains,
            limit=args.limit,
            seed=args.seed,
            revision=dataset_revision,
        )
    )
    if not rows:
        raise RuntimeError("No benchmark rows matched the requested filters")

    torch, _, snapshot_download, AutoTokenizer, _, _, _ = _lazy_runtime()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the default original SingGuard realtime benchmark")

    load_started = perf_counter()
    # The head set is validated before the tokenizer/model load and, crucially, before warmup
    # and any measurement: a partial set fails the run instead of silently scoring the missing
    # domains as 0.0 risk probabilities.
    model_path, resolved_model_revision = _resolve_model_snapshot(
        args.model,
        model_revision,
        snapshot_download,
    )
    model_ref = str(model_path)
    heads = _load_heads(model_ref, args.heads_dir, args.device)
    head_manifest = _head_manifest(benchmark_side, heads)
    _require_complete_head_set(
        head_manifest,
        side=benchmark_side,
        heads_dir=args.heads_dir,
        allow_partial=allow_partial_heads,
    )
    if not head_manifest["loaded_domains"]:
        raise RuntimeError(
            f"No {benchmark_side.value}-side SingGuard classification heads were loaded from "
            f"{args.heads_dir or args.model!r}: expected {head_manifest['expected_domains']!r}. "
            "Point --heads-dir at the NSFA classification-head directory."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_ref, truncation_side="left", use_fast=True)
    llm = _make_llm(
        model_ref,
        args.max_tokens,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
        args.dtype,
    )
    model_max_tokens = min(
        int(llm.llm_engine.model_config.max_model_len),
        args.max_tokens,
    )
    runner_started = perf_counter()
    # Only the benchmarked side is stacked. The unused side's heads cannot influence this
    # run's numbers, so demanding them would reject a perfectly valid single-side head set.
    runners = {benchmark_side.value: _prepare_head_runner(heads, benchmark_side.value, args.device)}
    classification_head_runner_seconds = perf_counter() - runner_started
    model_load_seconds = perf_counter() - load_started

    policy = ThresholdPolicy(default_threshold=args.threshold, review_margin=args.review_margin)

    if args.warmup > 0:
        warm_rows = rows[: min(args.warmup, len(rows))]
        _infer_batch(
            llm=llm,
            runner=_runner_for_side(runners, warm_rows[0].side),
            tokenizer=tokenizer,
            rows=warm_rows,
            device=args.device,
            model_max_tokens=model_max_tokens,
        )

    results: list[GuardResult] = []
    successful_rows: list[BenchmarkRow] = []
    batch_latencies: list[float] = []
    sample_latencies: list[float] = []
    amortized_latencies: list[float] = []
    started = perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        batch_scores, batch_ms = _infer_batch(
            llm=llm,
            runner=_runner_for_side(runners, batch[0].side),
            tokenizer=tokenizer,
            rows=batch,
            device=args.device,
            model_max_tokens=model_max_tokens,
        )
        batch_latencies.append(batch_ms)
        # Every request in a batch waits for the batch to finish, so the request latency
        # is the full batch wall time; the per-sample cost is a separate, clearly labelled
        # throughput figure so it cannot be mistaken for a request latency.
        sample_latencies.extend([batch_ms] * len(batch))
        amortized_latencies.extend([batch_ms / len(batch)] * len(batch))
        for row, scores in zip(batch, batch_scores, strict=True):
            predicted_domain = max(scores, key=scores.get)
            max_risk = scores[predicted_domain]
            results.append(
                GuardResult(
                    side=row.side,
                    scores=scores,
                    unsafe=policy.is_unsafe(scores),
                    decision=policy.decision(scores),
                    predicted_domain=predicted_domain,
                    max_risk=max_risk,
                    latency_ms=batch_ms,
                    model=args.model,
                )
            )
            successful_rows.append(row)
    wall_seconds = perf_counter() - started

    if head_manifest["complete"]:
        quality = evaluate_guard_results(rows, results, threshold=args.threshold)
    else:
        # A partial head set has no score for the missing domains, and
        # metrics.evaluate_guard_results refuses to invent one -- a head that never ran is not
        # a 0.0 risk probability. The measurement itself is still reported, without quality.
        quality = None
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device
    steady_cost = (
        wall_seconds * args.gpu_hourly_usd / 3600.0
        if args.gpu_hourly_usd is not None
        else None
    )
    cost_per_1k = (
        steady_cost * 1000.0 / len(results)
        if steady_cost is not None and results
        else None
    )
    cold_start_cost = (
        model_load_seconds * args.gpu_hourly_usd / 3600.0
        if args.gpu_hourly_usd is not None
        else None
    )

    return {
        "schema_version": 1,
        "engine": "singguard-nsfa",
        "mode": "local-realtime-classification",
        "model": args.model,
        "model_revision": {
            "requested": model_revision,
            "resolved": resolved_model_revision,
        },
        "hardware": {
            "device": args.device,
            "gpu_name": gpu_name,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "benchmark": args.benchmark,
            "languages": sorted(languages) if languages else None,
            "id_contains": args.id_contains,
            "seed": args.seed,
            "revision": dataset_revision,
            "fingerprint": _rows_fingerprint(rows),
        },
        "parameters": {
            "threshold": args.threshold,
            "review_margin": args.review_margin,
            "batch_size": args.batch_size,
            "warmup_samples": args.warmup,
            "max_tokens": args.max_tokens,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_hourly_usd": args.gpu_hourly_usd,
        },
        "samples": {
            "attempted": len(rows),
            "successful": len(results),
            "failed": 0,
            # Diagnostic only, and deliberately id-only: the comparator must gate on the
            # content-covering fingerprints below, never on this digest.
            "attempted_ids_sha256": _rows_id_digest(rows),
            # Covers the rows that actually produced a scored result; SingGuard scores every
            # attempted row or fails the run, so this equals dataset.fingerprint today.
            "successful_sha256": _rows_fingerprint(successful_rows),
        },
        "head_manifest": head_manifest,
        "baseline_complete": head_manifest["complete"],
        "baseline_note": (
            "Complete NSFA Level-1 classification-head set for the benchmarked side."
            if head_manifest["complete"]
            else (
                "Partial classification-head set (--allow-partial-heads): missing "
                f"{head_manifest['missing_domains']!r} and unexpected "
                f"{head_manifest['unexpected_domains']!r}. This run is NOT a complete NSFA "
                "baseline and its quality numbers are not comparable to a full head set; "
                "quality is reported as null because a head that never ran is not a 0.0 risk "
                "probability."
            )
        ),
        "quality": quality,
        "cold_start": {
            "model_and_head_load_seconds": model_load_seconds,
            "classification_head_runner_seconds": classification_head_runner_seconds,
            "allocated_gpu_cost_usd": cold_start_cost,
        },
        "latency_scope": "request",
        "latency_notes": {
            "latency_ms": (
                "Per-request latency: every request in a batch waits for the whole batch, so each "
                "sample is charged the full batch wall time."
            ),
            "amortized_ms_per_sample": (
                "Batch wall time divided by the number of samples: a throughput-style per-sample "
                "cost, not a request latency."
            ),
            "batch_latency_ms": "Wall time per batch.",
            "pairing": (
                "amortized_ms_per_sample == latency_ms / parameters.batch_size; compare latency_ms, "
                "not amortized_ms_per_sample, against per-request managed-API latency."
            ),
        },
        "latency_ms": latency_summary(sample_latencies),
        "amortized_ms_per_sample": latency_summary(amortized_latencies),
        "batch_latency_ms": latency_summary(batch_latencies),
        "throughput": {
            "wall_seconds": wall_seconds,
            "successful_requests_per_second": len(results) / wall_seconds if wall_seconds else None,
        },
        "usage": {
            "gpu_hourly_usd": args.gpu_hourly_usd,
            "steady_state_allocated_gpu_cost_usd": steady_cost,
            "cost_per_1000_successful_requests_usd": cost_per_1k,
        },
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="inclusionAI/NSFA_Benchmarks")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Dataset revision (commit/tag) to load; recorded as dataset.revision",
    )
    parser.add_argument(
        "--benchmark",
        choices=["query", "response", "cross-source-query"],
        default="query",
    )
    parser.add_argument("--language", action="append")
    parser.add_argument("--id-contains", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="inclusionAI/SingGuard-NSFA-0.8B")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Hugging Face model revision (commit/tag) used for heads, tokenizer, and vLLM",
    )
    parser.add_argument("--heads-dir", default=None)
    parser.add_argument(
        "--allow-partial-heads",
        action="store_true",
        help=(
            "Measure a partial classification-head set. The run is reported with "
            "head_manifest.complete=false and baseline_complete=false instead of pretending "
            "to be a complete NSFA baseline"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--review-margin", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-hourly-usd", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/singguard.json"))


def main_from_args(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    report = run_benchmark(args)
    if not report.get("baseline_complete", True):
        print(f"WARNING: {report.get('baseline_note', 'incomplete baseline')}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return main_from_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
