"""End-to-end benchmark runner for JevGuard-NSFA."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from .dataset import BenchmarkRow, iter_huggingface_rows
from .guard import AsyncJevGuard
from .metrics import evaluate_guard_results, latency_summary
from .models import GuardResult, ThresholdPolicy

DEFAULT_DATASET = "inclusionAI/NSFA_Benchmarks"
DEFAULT_JEV_INPUT_USD_PER_MILLION = 0.042


def _sha256_lines(lines: Iterable[str]) -> str:
    """SHA-256 hex digest of ``"\n".join(lines)`` encoded as UTF-8."""
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _rows_fingerprint(rows: Sequence[BenchmarkRow]) -> str:
    """Identify the attempted sample selection in emitted order."""
    return _sha256_lines(f"{row.id}|{row.label}|{row.side.value}|{row.lang}" for row in rows)


def _rows_id_digest(rows: Sequence[BenchmarkRow]) -> str:
    """Identify the row ids of a sample subset in emitted order."""
    return _sha256_lines(row.id for row in rows)


class RequestStartLimiter:
    """Simple request-start limiter to stay below a provider RPM ceiling."""

    def __init__(self, rpm: float) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            delay = max(0.0, self._next - now)
            if delay:
                await asyncio.sleep(delay)
                now = loop.time()
            self._next = max(self._next, now) + self.interval


async def _run_one(
    row: BenchmarkRow,
    guard: AsyncJevGuard,
    semaphore: asyncio.Semaphore,
    limiter: RequestStartLimiter,
) -> tuple[BenchmarkRow, GuardResult | None, str | None]:
    async with semaphore:
        await limiter.acquire()
        try:
            return row, await guard.screen(row.text, row.side), None
        except Exception as exc:
            return row, None, f"{type(exc).__name__}: {exc}"


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    languages = set(args.language) if args.language else None
    rows = list(
        iter_huggingface_rows(
            dataset_name=args.dataset,
            split=args.split,
            benchmark=args.benchmark,
            languages=languages,
            id_contains=args.id_contains,
            limit=args.limit,
            seed=args.seed,
        )
    )
    if not rows:
        raise RuntimeError("No benchmark rows matched the requested filters")

    policy = ThresholdPolicy(default_threshold=args.threshold, review_margin=args.review_margin)
    limiter = RequestStartLimiter(args.rpm)
    semaphore = asyncio.Semaphore(args.concurrency)
    retry = RetryPolicy(max_retries=args.retries)

    warmup_tokens = 0
    async with AsyncTypeSafeClient(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        retry=retry,
    ) as client:
        guard = AsyncJevGuard(policy=policy, client=client)

        for row in rows[: args.warmup]:
            await limiter.acquire()
            warm = await guard.screen(row.text, row.side)
            warmup_tokens += warm.input_tokens or 0

        started = perf_counter()
        outcomes = await asyncio.gather(
            *(_run_one(row, guard, semaphore, limiter) for row in rows)
        )
        wall_seconds = perf_counter() - started

    successful_rows: list[BenchmarkRow] = []
    results: list[GuardResult] = []
    failures: list[dict[str, str]] = []
    for row, result, error in outcomes:
        if result is None:
            failures.append({"id": row.id, "error": error or "unknown error"})
        else:
            successful_rows.append(row)
            results.append(result)

    quality = (
        evaluate_guard_results(successful_rows, results, threshold=args.threshold)
        if results
        else None
    )
    latencies = [result.latency_ms for result in results]
    token_values = [result.input_tokens for result in results]
    known_input_tokens = sum(value or 0 for value in token_values)
    usage_complete = all(value is not None for value in token_values)
    api_cost_usd = (
        known_input_tokens * args.input_price_per_million / 1_000_000.0
        if usage_complete
        else None
    )
    cost_per_1k = (
        api_cost_usd * 1000.0 / len(results)
        if api_cost_usd is not None and results
        else None
    )

    return {
        "schema_version": 1,
        "engine": "jevguard-nsfa",
        "mode": "managed-api-online",
        "model": results[0].model if results else args.model,
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "benchmark": args.benchmark,
            "languages": sorted(languages) if languages else None,
            "id_contains": args.id_contains,
            "seed": args.seed,
            "fingerprint": _rows_fingerprint(rows),
        },
        "parameters": {
            "threshold": args.threshold,
            "review_margin": args.review_margin,
            "concurrency": args.concurrency,
            "rpm": args.rpm,
            "timeout_seconds": args.timeout,
            "retries": args.retries,
            "warmup_requests": args.warmup,
            "input_price_usd_per_million": args.input_price_per_million,
        },
        "samples": {
            "attempted": len(rows),
            "successful": len(results),
            "failed": len(failures),
            "attempted_ids_sha256": _rows_id_digest(rows),
            "successful_ids_sha256": _rows_id_digest(successful_rows),
        },
        # Jev latency_ms is managed-API end-to-end latency for a single request.
        "latency_scope": "request",
        "quality": quality,
        "latency_ms": latency_summary(latencies),
        "throughput": {
            "wall_seconds": wall_seconds,
            "successful_requests_per_second": len(results) / wall_seconds if wall_seconds else None,
            "attempted_requests_per_second": len(rows) / wall_seconds if wall_seconds else None,
        },
        "usage": {
            "input_tokens": known_input_tokens,
            "input_tokens_complete": usage_complete,
            "warmup_input_tokens": warmup_tokens,
            "api_cost_usd": api_cost_usd,
            "cost_per_1000_successful_requests_usd": cost_per_1k,
        },
        "failures": failures[:100],
        "failure_records_truncated": len(failures) > 100,
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--benchmark",
        choices=["query", "response", "cross-source-query"],
        default="query",
    )
    parser.add_argument("--language", action="append", help="Language code; repeat to include multiple languages")
    parser.add_argument("--id-contains", default=None, help="Optional substring filter for dataset row ids")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--review-margin", type=float, default=0.10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rpm", type=float, default=900.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--input-price-per-million", type=float, default=DEFAULT_JEV_INPUT_USD_PER_MILLION)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/jevguard.json"))


def main_from_args(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    report = asyncio.run(run_benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Undefined metrics are None, never NaN, so the artifact must stay strict JSON.
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return main_from_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
