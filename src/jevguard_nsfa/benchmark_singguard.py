"""Local real-time classification benchmark for the original SingGuard-NSFA."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .dataset import BenchmarkRow, canonical_domain, iter_huggingface_rows
from .metrics import evaluate_guard_results, latency_summary
from .models import Decision, GuardResult, Side, ThresholdPolicy


def _lazy_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from huggingface_hub import snapshot_download
        from torch.func import functional_call, stack_module_state, vmap
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
            for source, target in zip(dims, dims[1:], strict=True):
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
        payload = torch.load(path, weights_only=False, map_location=device)
        if "head_state_dict" not in payload:
            continue
        config = dict(payload["head_config"])
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
        head.load_state_dict(payload["head_state_dict"])
        head.eval().to(dtype=torch.float32, device=device)
        raw_name = str(payload["sub_task_name"])
        name = canonical_domain(raw_name)
        if name is None:
            raise ValueError(f"Unknown SingGuard NSFA head name: {raw_name!r}")
        heads[name] = {
            "head": head,
            "task": str(payload["task"]).lower(),
            "max_tokens": int(payload.get("max_tokens", 8192)),
            "system_prompt": payload.get("system_prompt"),
        }

    if not heads:
        raise RuntimeError(f"No valid NSFA classification heads found in {resolved}")
    return heads


def _make_llm(
    model: str,
    max_tokens: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    dtype: str,
) -> Any:
    _, _, _, _, LLM, PoolerConfig, EngineArgs = _lazy_runtime()

    def pooler() -> Any:
        candidates = (
            {"pooling_type": "LAST", "normalize": False, "task": "embed"},
            {"pooling_type": "LAST", "normalize": False},
            {"pooling_type": "LAST"},
        )
        for kwargs in candidates:
            try:
                return PoolerConfig(**kwargs)
            except (TypeError, ValueError):
                pass
        return PoolerConfig()

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
        kwargs["pooler_config"] = pooler()
    else:
        kwargs["task"] = "embed"
        kwargs["override_pooler_config"] = pooler()
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
    torch, _, _, _, _, _, _ = _lazy_runtime()
    from torch.func import functional_call, stack_module_state, vmap

    params, buffers = stack_module_state(head_modules)
    meta_model = copy.deepcopy(head_modules[0]).to("meta")

    def one(p: Any, b: Any, data: Any) -> Any:
        return functional_call(meta_model, (p, b), (data,))

    return vmap(one, in_dims=(0, 0, None)), params, buffers


def _infer_batch(
    *,
    llm: Any,
    heads: dict[str, dict[str, Any]],
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

    task = side.value
    matching = {name: info for name, info in heads.items() if info["task"] == task}
    if not matching:
        raise RuntimeError(f"No classification heads found for task={task!r}")
    names = sorted(matching)
    exemplar = matching[names[0]]
    effective_max = min(model_max_tokens, int(exemplar["max_tokens"]))
    system_prompt = exemplar["system_prompt"]
    prompts = [
        _prepare_prompt(row.text, side, tokenizer, effective_max, system_prompt)
        for row in rows
    ]

    modules = [matching[name]["head"] for name in names]
    forward, params, buffers = _build_parallel_head_forward(modules)

    started = perf_counter()
    outputs = llm.embed(prompts, use_tqdm=False)
    embeddings = torch.tensor(
        [output.outputs.embedding for output in outputs],
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        logits = forward(params, buffers, embeddings)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
    elapsed_ms = (perf_counter() - started) * 1000.0

    scores: list[dict[str, float]] = []
    for sample_index in range(len(rows)):
        item: dict[str, float] = {}
        for head_index, name in enumerate(names):
            row_probs = probabilities[head_index, sample_index]
            if row_probs.numel() < 2:
                raise RuntimeError(f"Expected binary head probabilities for {name!r}")
            item[name] = float(row_probs[1])
        scores.append(item)
    return scores, elapsed_ms


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    side = Side(args.side)
    languages = set(args.language) if args.language else None
    rows = list(
        iter_huggingface_rows(
            dataset_name=args.dataset,
            split=args.split,
            side=side,
            languages=languages,
            id_contains=args.id_contains,
            limit=args.limit,
            seed=args.seed,
        )
    )
    if not rows:
        raise RuntimeError("No benchmark rows matched the requested filters")

    torch, _, _, AutoTokenizer, _, _, _ = _lazy_runtime()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the default original SingGuard realtime benchmark")

    load_started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, truncation_side="left", use_fast=True)
    llm = _make_llm(
        args.model,
        args.max_tokens,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
        args.dtype,
    )
    model_max_tokens = min(
        int(llm.llm_engine.model_config.max_model_len),
        args.max_tokens,
    )
    heads = _load_heads(args.model, args.heads_dir, args.device)
    model_load_seconds = perf_counter() - load_started

    policy = ThresholdPolicy(default_threshold=args.threshold, review_margin=args.review_margin)

    if args.warmup > 0:
        warm_rows = rows[: min(args.warmup, len(rows))]
        _infer_batch(
            llm=llm,
            heads=heads,
            tokenizer=tokenizer,
            rows=warm_rows,
            device=args.device,
            model_max_tokens=model_max_tokens,
        )

    results: list[GuardResult] = []
    batch_latencies: list[float] = []
    sample_latencies: list[float] = []
    started = perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        batch_scores, batch_ms = _infer_batch(
            llm=llm,
            heads=heads,
            tokenizer=tokenizer,
            rows=batch,
            device=args.device,
            model_max_tokens=model_max_tokens,
        )
        batch_latencies.append(batch_ms)
        amortized = batch_ms / len(batch)
        sample_latencies.extend([amortized] * len(batch))
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
                    latency_ms=amortized,
                    model=args.model,
                )
            )
    wall_seconds = perf_counter() - started

    quality = evaluate_guard_results(rows, results, threshold=args.threshold)
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
        "hardware": {
            "device": args.device,
            "gpu_name": gpu_name,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "side": args.side,
            "languages": sorted(languages) if languages else None,
            "id_contains": args.id_contains,
            "seed": args.seed,
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
        },
        "quality": quality,
        "cold_start": {
            "model_and_head_load_seconds": model_load_seconds,
            "allocated_gpu_cost_usd": cold_start_cost,
        },
        "latency_ms": latency_summary(sample_latencies),
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
    parser.add_argument("--side", choices=["query", "response"], required=True)
    parser.add_argument("--language", action="append")
    parser.add_argument("--id-contains", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="inclusionAI/SingGuard-NSFA-0.8B")
    parser.add_argument("--heads-dir", default=None)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return main_from_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
