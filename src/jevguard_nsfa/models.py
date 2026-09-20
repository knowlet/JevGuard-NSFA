"""Core result and policy types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class Side(str, Enum):
    QUERY = "query"
    RESPONSE = "response"


class Decision(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Convert independent domain probabilities into an operational decision.

    The binary benchmark label always uses each domain's threshold directly.
    The review band is an operational policy around those thresholds and is not
    used when computing apples-to-apples benchmark F1.
    """

    default_threshold: float = 0.5
    domain_thresholds: Mapping[str, float] = field(default_factory=dict)
    review_margin: float = 0.10

    def __post_init__(self) -> None:
        values = [self.default_threshold, *self.domain_thresholds.values()]
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("thresholds must be in [0, 1]")
        if not 0.0 <= self.review_margin <= 0.5:
            raise ValueError("review_margin must be in [0, 0.5]")

    def threshold_for(self, domain: str) -> float:
        return float(self.domain_thresholds.get(domain, self.default_threshold))

    def is_unsafe(self, scores: Mapping[str, float]) -> bool:
        return any(score > self.threshold_for(domain) for domain, score in scores.items())

    def decision(self, scores: Mapping[str, float]) -> Decision:
        if not scores:
            raise ValueError("scores must not be empty")
        best_margin = max(score - self.threshold_for(domain) for domain, score in scores.items())
        if best_margin >= self.review_margin:
            return Decision.BLOCK
        if best_margin <= -self.review_margin:
            return Decision.ALLOW
        return Decision.REVIEW


@dataclass(frozen=True)
class GuardResult:
    side: Side
    scores: Mapping[str, float]
    unsafe: bool
    decision: Decision
    predicted_domain: str | None
    max_risk: float
    latency_ms: float
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side.value,
            "scores": dict(self.scores),
            "unsafe": self.unsafe,
            "decision": self.decision.value,
            "predicted_domain": self.predicted_domain,
            "max_risk": self.max_risk,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
