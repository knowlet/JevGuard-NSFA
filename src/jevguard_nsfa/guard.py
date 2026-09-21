"""Jev-backed NSFA guard.

Transport/provider failures are deliberately not converted into System-Two
fallbacks. A fallback may run only when a successful Jev decision lands in the
configured semantic review band.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, TypeVar

from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

from .models import Decision, GuardResult, Side, ThresholdPolicy
from .taxonomy import questions_for

T = TypeVar("T")


def _result_from_response(
    *,
    response: Any,
    side: Side,
    policy: ThresholdPolicy,
    latency_ms: float,
) -> GuardResult:
    expected = tuple(questions_for(side))
    scores: dict[str, float] = {}
    for name in expected:
        answer = response.nouls.get(name)
        if answer is None:
            raise ValueError(f"Jev response is missing Noul answer {name!r}")
        score = float(answer.noul)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Jev returned out-of-range probability for {name!r}")
        scores[name] = score

    predicted_domain = max(scores, key=scores.get) if scores else None
    max_risk = scores[predicted_domain] if predicted_domain is not None else 0.0
    usage = getattr(response, "usage", None)
    return GuardResult(
        side=side,
        scores=scores,
        unsafe=policy.is_unsafe(scores),
        decision=policy.decision(scores),
        predicted_domain=predicted_domain,
        max_risk=max_risk,
        latency_ms=latency_ms,
        model=str(getattr(response, "model", "unknown")),
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


class JevGuard:
    def __init__(
        self,
        *,
        policy: ThresholdPolicy | None = None,
        client: TypeSafeClient | Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.policy = policy or ThresholdPolicy()
        self._owns_client = client is None
        self.client = client or TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

    def screen(self, text: str, side: Side | str, *, timeout: float | None = None) -> GuardResult:
        side = Side(side)
        started = perf_counter()
        request = {"state": {"untrusted_text": text}, "questions": questions_for(side)}
        if timeout is not None:
            request["timeout"] = timeout
        response = self.client.system_one(**request)
        return _result_from_response(
            response=response,
            side=side,
            policy=self.policy,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    def screen_or_fallback(
        self,
        text: str,
        side: Side | str,
        fallback: Callable[[str, Side, GuardResult], T],
    ) -> GuardResult | T:
        """Use System Two only for semantic ambiguity.

        Exceptions from Jev propagate unchanged. This intentionally prevents
        authentication, quota, timeout, malformed-response, or transport
        failures from silently changing the security decision path.
        """

        resolved_side = Side(side)
        result = self.screen(text, resolved_side)
        if result.decision is Decision.REVIEW:
            return fallback(text, resolved_side, result)
        return result

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "JevGuard":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncJevGuard:
    def __init__(
        self,
        *,
        policy: ThresholdPolicy | None = None,
        client: AsyncTypeSafeClient | Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.policy = policy or ThresholdPolicy()
        self._owns_client = client is None
        self.client = client or AsyncTypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

    async def screen(self, text: str, side: Side | str, *, timeout: float | None = None) -> GuardResult:
        side = Side(side)
        started = perf_counter()
        request = {"state": {"untrusted_text": text}, "questions": questions_for(side)}
        if timeout is not None:
            request["timeout"] = timeout
        response = await self.client.system_one(**request)
        return _result_from_response(
            response=response,
            side=side,
            policy=self.policy,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    async def screen_or_fallback(
        self,
        text: str,
        side: Side | str,
        fallback: Callable[[str, Side, GuardResult], Any],
    ) -> GuardResult | Any:
        resolved_side = Side(side)
        result = await self.screen(text, resolved_side)
        if result.decision is Decision.REVIEW:
            value = fallback(text, resolved_side, result)
            if hasattr(value, "__await__"):
                return await value
            return value
        return result

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> "AsyncJevGuard":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def scores_to_binary(
    scores: Mapping[str, float],
    policy: ThresholdPolicy,
) -> bool:
    return policy.is_unsafe(scores)
