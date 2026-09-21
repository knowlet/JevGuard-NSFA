"""End-to-end benchmark runner for JevGuard-NSFA.

Retry ownership
---------------
The SDK client is built with ``RetryPolicy(max_retries=0)``, so one ``guard.screen`` call is
exactly one HTTP attempt. Retries are owned by this runner instead: every attempt -- including a
retry -- acquires :class:`RequestStartLimiter` immediately before it is sent, so a retried request
is visible to the provider rate limiter rather than multiplying requests behind its back. The
retryable set mirrors the SDK defaults: connection and timeout errors, plus HTTP 408, 429 and 5xx.

Operational failures stay failures: a row whose attempts are exhausted is recorded with its error,
never reinterpreted as a safe verdict and never turned into a System-Two fallback (see guard.py).
``samples.request_attempts`` and ``samples.retried_requests`` count the measured rows only, and a
row holds its ``--concurrency`` slot across its attempts, so concurrency keeps bounding in-flight
rows while the attempt budget bounds how long one row can hold a slot.

``--timeout`` is the per-request HTTP timeout handed to the SDK. ``--row-budget`` is the total
budget for one row's attempt sequence and defaults to ``--timeout``: with that default one
timed-out attempt spends the whole budget and the row fails without a retry, so a full-corpus run
that has to survive a rare transport hang sets ``--row-budget`` above ``--timeout``. A retry is not
started when its wait would push the row past that budget, so a row cannot outlive the deadline the
operator configured. The per-attempt wait is ``Retry-After`` when the failure carries it, otherwise
exponential backoff from 0.5s with jitter, capped at the local ``_RETRY_BACKOFF_CAP_SECONDS``.
A server-provided ``Retry-After`` value is kept as requested; the total row budget decides whether
another attempt can begin.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import replace
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import monotonic, perf_counter, time
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
)

from .dataset import BenchmarkRow, iter_huggingface_rows
from .guard import AsyncJevGuard
from .metrics import evaluate_guard_results, latency_summary
from .models import GuardResult, ThresholdPolicy

DEFAULT_DATASET = "inclusionAI/NSFA_Benchmarks"
DEFAULT_JEV_INPUT_USD_PER_MILLION = 0.042

# Mirrors ``typesafe_sdk.RetryPolicy``'s default retryable HTTP statuses.
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429})
_RETRY_BACKOFF_INITIAL_SECONDS = 0.5
_RETRY_BACKOFF_CAP_SECONDS = 8.0
_RETRY_BACKOFF_JITTER = 0.25


def _sha256_lines(lines: Iterable[str]) -> str:
    """SHA-256 hex digest of ``"\n".join(lines)`` encoded as UTF-8."""
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _sample_identity(row: BenchmarkRow) -> str:
    """Canonical JSON identity of one sample, byte-identical to ``benchmark_singguard._sample_identity``.

    The fingerprint must cover everything that changes what was measured, not just the row id:
    two runs over the same ids with different texts, labels, sides, L1 domains or languages are
    different selections and must not be reported as aligned.
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


def _is_retryable(error: BaseException) -> bool:
    """Classify a failure exactly like the SDK's default retry predicate.

    Transport-level failures (connection, timeout) and HTTP 408/429/5xx are retryable; everything
    else -- authentication, permission, bad request, an unparseable 200 response -- fails the row
    immediately, because retrying it would only repeat a non-transient mistake.
    """
    if isinstance(error, (TypeSafeAPITimeoutError, TypeSafeAPIConnectionError)):
        return True
    if isinstance(error, TypeSafeAPIError):
        return error.status in _RETRYABLE_HTTP_STATUSES or 500 <= error.status <= 599
    return False


def _retry_after_header_seconds(value: str, *, milliseconds: bool) -> float | None:
    """Parse one ``Retry-After`` style header value, or ``None`` when it is unusable."""
    try:
        numeric = float(value.strip())
    except (AttributeError, TypeError, ValueError):
        numeric = None
    if numeric is not None:
        if not math.isfinite(numeric) or numeric < 0:
            return None
        return numeric / 1000.0 if milliseconds else numeric
    if milliseconds:
        return None
    try:
        delta = parsedate_to_datetime(value).timestamp() - time()
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, delta) if math.isfinite(delta) else None


def _retry_after_seconds(error: BaseException) -> float | None:
    """The server's requested wait in seconds, from the SDK attribute or the response headers."""
    retry_after_ms = getattr(error, "retry_after_ms", None)
    if isinstance(retry_after_ms, (int, float)) and not isinstance(retry_after_ms, bool):
        if math.isfinite(retry_after_ms) and retry_after_ms >= 0:
            return float(retry_after_ms) / 1000.0
        return None

    headers = getattr(error, "headers", None)
    get_header = getattr(headers, "get", None)
    if not callable(get_header):
        return None
    raw_ms = get_header("retry-after-ms")
    if raw_ms is not None:
        return _retry_after_header_seconds(raw_ms, milliseconds=True)
    raw = get_header("retry-after")
    if raw is not None:
        return _retry_after_header_seconds(raw, milliseconds=False)
    return None


def retry_wait_seconds(attempt: int, error: BaseException) -> float:
    """Delay in seconds before the next attempt, for the 1-based ``attempt`` that just failed.

    ``Retry-After`` wins when the failure exposes one. Server-requested waits are not shortened;
    the row budget decides whether another attempt is allowed. Without that header, use local
    exponential backoff from 0.5s with jitter and a local cap.
    """
    delay = _retry_after_seconds(error)
    if delay is None:
        delay = min(_RETRY_BACKOFF_INITIAL_SECONDS * 2 ** (attempt - 1), _RETRY_BACKOFF_CAP_SECONDS)
        delay *= 1.0 - _RETRY_BACKOFF_JITTER * random.random()
    return delay


async def _sleep(seconds: float) -> None:
    """Default inter-attempt sleep; tests inject their own so no test ever waits."""
    if seconds > 0:
        await asyncio.sleep(seconds)


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


async def _screen_with_attempts(
    row: BenchmarkRow,
    guard: AsyncJevGuard,
    limiter: RequestStartLimiter,
    *,
    retries: int,
    timeout: float | None,
    budget: float | None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[GuardResult | None, str | None, int]:
    """Screen one row, retrying only what the SDK itself would retry.

    With ``RetryPolicy(max_retries=0)`` each ``guard.screen`` call is exactly one real HTTP
    attempt, so the limiter is acquired immediately before every attempt and the provider sees the
    true request rate. A failure is reported as a failure; it is never converted into a verdict.

    ``timeout`` bounds a single HTTP attempt while ``budget`` bounds the whole row, so a row that
    hits a rare transport hang can still use its retries. When the two are equal, a timed-out
    attempt spends the entire budget and the row fails without a second attempt.
    """
    sleeper = sleep or _sleep
    now = clock or monotonic
    started = 0.0
    attempts = 0
    while True:
        attempt_timeout = timeout
        if budget is not None and attempts > 0:
            # The budget gate runs before the limiter so a row that cannot start another
            # attempt never reserves a request slot it will not use.
            remaining = budget - (now() - started)
            if remaining <= 0:
                return None, "retry budget exhausted before attempt", attempts
            attempt_timeout = remaining if timeout is None else min(timeout, remaining)
        await limiter.acquire()
        if attempts == 0:
            # Start the logical-request clock immediately before the first attempt.
            started = now()
        attempts += 1
        try:
            if attempt_timeout is None:
                result = await guard.screen(row.text, row.side)
            else:
                result = await guard.screen(row.text, row.side, timeout=attempt_timeout)
            return replace(result, latency_ms=max(0.0, now() - started) * 1000.0), None, attempts
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempts > retries or not _is_retryable(exc):
                return None, error, attempts
            delay = retry_wait_seconds(attempts, exc)
            if budget is not None and now() - started + delay >= budget:
                return None, error, attempts
            await sleeper(delay)


async def _run_one(
    row: BenchmarkRow,
    guard: AsyncJevGuard,
    semaphore: asyncio.Semaphore,
    limiter: RequestStartLimiter,
    *,
    retries: int = 0,
    timeout: float | None = None,
    budget: float | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[BenchmarkRow, GuardResult | None, str | None, int]:
    """Run one row under the concurrency semaphore; return its result (or error) and attempt count."""
    async with semaphore:
        result, error, attempts = await _screen_with_attempts(
            row,
            guard,
            limiter,
            retries=retries,
            timeout=timeout,
            budget=budget,
            sleep=sleep,
            clock=clock,
        )
    return row, result, error, attempts


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    languages = set(args.language) if args.language else None
    dataset_revision = getattr(args, "dataset_revision", None)
    full_dataset = bool(getattr(args, "full", False))
    selected_limit = None if full_dataset else args.limit
    rows = list(
        iter_huggingface_rows(
            dataset_name=args.dataset,
            split=args.split,
            benchmark=args.benchmark,
            languages=languages,
            id_contains=args.id_contains,
            limit=selected_limit,
            seed=args.seed,
            revision=dataset_revision,
        )
    )
    if not rows:
        raise RuntimeError("No benchmark rows matched the requested filters")

    policy = ThresholdPolicy(default_threshold=args.threshold, review_margin=args.review_margin)
    limiter = RequestStartLimiter(args.rpm)
    semaphore = asyncio.Semaphore(args.concurrency)
    # SDK retries are disabled on purpose: this runner retries so that every real HTTP attempt is
    # acquired from the limiter and counted in samples.request_attempts.
    retry = RetryPolicy(max_retries=0)
    request_timeout = float(args.timeout) if args.timeout and args.timeout > 0 else None
    row_budget = getattr(args, "row_budget", None)
    # Default to the historical single-value deadline: one row may not outlive ``--timeout``
    # unless the operator widens the retry window with ``--row-budget``.
    retry_budget = (
        request_timeout
        if row_budget is None
        else (float(row_budget) if row_budget and row_budget > 0 else None)
    )

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
            *(
                _run_one(
                    row,
                    guard,
                    semaphore,
                    limiter,
                    retries=args.retries,
                    timeout=request_timeout,
                    budget=retry_budget,
                )
                for row in rows
            )
        )
        wall_seconds = perf_counter() - started

    successful_rows: list[BenchmarkRow] = []
    results: list[GuardResult] = []
    failures: list[dict[str, Any]] = []
    request_attempts = 0
    retried_requests = 0
    for row, result, error, attempts in outcomes:
        request_attempts += attempts
        if attempts > 1:
            retried_requests += 1
        if result is None:
            failures.append({"id": row.id, "error": error or "unknown error", "attempts": attempts})
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
            # The revision actually requested, or null when the hub's default revision was used.
            "revision": dataset_revision,
            "fingerprint": _rows_fingerprint(rows),
        },
        "parameters": {
            "threshold": args.threshold,
            "review_margin": args.review_margin,
            "concurrency": args.concurrency,
            "rpm": args.rpm,
            "timeout_seconds": args.timeout,
            "row_budget_seconds": retry_budget,
            "retries": args.retries,
            "warmup_requests": args.warmup,
            "input_price_usd_per_million": args.input_price_per_million,
            "requested_limit": None if full_dataset else args.limit,
            "full_dataset": full_dataset,
        },
        "samples": {
            "attempted": len(rows),
            "successful": len(results),
            "failed": len(failures),
            # Diagnostic only, and deliberately id-only: the comparator gates on the
            # content-covering fingerprints, never on this digest.
            "attempted_ids_sha256": _rows_id_digest(rows),
            # Covers the rows that actually produced a scored result.
            "successful_sha256": _rows_fingerprint(successful_rows),
            # Screening attempts including runner-managed retries; attempted/successful/failed
            # above keep counting rows.
            "request_attempts": request_attempts,
            "retried_requests": retried_requests,
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
        "--dataset-revision",
        default=None,
        help="Dataset revision (branch, tag or commit) to load; omitted means the hub default",
    )
    parser.add_argument(
        "--benchmark",
        choices=["query", "response", "cross-source-query"],
        default="query",
    )
    parser.add_argument("--language", action="append", help="Language code; repeat to include multiple languages")
    parser.add_argument("--id-contains", default=None, help="Optional substring filter for dataset row ids")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--full", action="store_true", help="Ignore --limit and score the full selected benchmark subset")
    parser.add_argument("--seed", type=int, default=42)
    # Left unset so the SDK resolves the model as explicit value -> TYPESAFE_DEFAULT_MODEL -> SDK
    # default; a hardcoded default here would make the environment variable unreachable.
    parser.add_argument(
        "--model",
        default=None,
        help="TypeSafe model name; omitted means TYPESAFE_DEFAULT_MODEL or the SDK default",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--review-margin", type=float, default=0.10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rpm", type=float, default=900.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-request HTTP timeout in seconds; the default total attempt budget for one row",
    )
    parser.add_argument(
        "--row-budget",
        type=float,
        default=None,
        help=(
            "Total seconds one row may spend across all of its attempts, including retries; "
            "defaults to --timeout, which means one timed-out attempt exhausts the budget"
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Runner-managed retries per row; each attempt is acquired from the request limiter",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--input-price-per-million", type=float, default=DEFAULT_JEV_INPUT_USD_PER_MILLION)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/jevguard.json"))


def main_from_args(args: argparse.Namespace) -> int:
    if not getattr(args, "full", False) and (args.limit is None or args.limit <= 0):
        raise ValueError("--limit must be positive unless --full is set")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    if getattr(args, "row_budget", None) is not None and args.row_budget <= 0:
        raise ValueError("--row-budget must be positive when set")
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
