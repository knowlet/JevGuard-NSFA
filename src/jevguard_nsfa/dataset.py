"""Adapters for the public inclusionAI/NSFA_Benchmarks dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import Side
from .taxonomy import DOMAINS


@dataclass(frozen=True)
class BenchmarkRow:
    id: str
    text: str
    label: int
    side: Side
    domain: str | None
    lang: str


def _slug(value: str) -> str:
    value = value.strip().lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


_DOMAIN_ALIASES: dict[str, str] = {}
for _domain in DOMAINS:
    for _alias in (_domain.id, _domain.title, _slug(_domain.title)):
        _DOMAIN_ALIASES[_slug(_alias)] = _domain.id

# Common short names used by model cards / head files.
_DOMAIN_ALIASES.update(
    {
        "sensitive_info_stealing": "sensitive_information_stealing",
        "danger_ops_and_tool_abuse": "dangerous_operations_and_tool_abuse",
        "hazardous_action": "hazardous_action_generation",
        "sensitive_info_leakage": "sensitive_information_leakage",
    }
)

_QUERY_IDS = {domain.id for domain in DOMAINS if domain.side is Side.QUERY}
_RESPONSE_IDS = {domain.id for domain in DOMAINS if domain.side is Side.RESPONSE}


def canonical_domain(value: Any) -> str | None:
    if value is None:
        return None
    key = _slug(str(value))
    if not key or key in {"no_risk", "safe", "none", "nan"}:
        return None
    return _DOMAIN_ALIASES.get(key)


def infer_side(row: dict[str, Any], forced: Side | None = None) -> Side:
    if forced is not None:
        return forced

    domain = canonical_domain(row.get("L1-Risk") or row.get("l1_risk") or row.get("domain"))
    if domain in _QUERY_IDS:
        return Side.QUERY
    if domain in _RESPONSE_IDS:
        return Side.RESPONSE

    explicit = str(row.get("side", "")).strip().lower()
    if explicit in {"query", "input"}:
        return Side.QUERY
    if explicit in {"response", "output"}:
        return Side.RESPONSE

    row_id = str(row.get("id", "")).lower()
    if "response" in row_id or "output" in row_id:
        return Side.RESPONSE
    if "query" in row_id or "input" in row_id or "crosssource" in row_id or "cross_source" in row_id:
        return Side.QUERY

    raise ValueError(
        f"Cannot infer query/response side for row {row.get('id')!r}; "
        "pass --side query or --side response for this dataset subset."
    )


def row_from_mapping(row: dict[str, Any], forced_side: Side | None = None) -> BenchmarkRow:
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(f"Expected binary label 0/1, got {label!r}")
    domain = canonical_domain(row.get("L1-Risk") or row.get("l1_risk") or row.get("domain"))
    return BenchmarkRow(
        id=str(row.get("id", "")),
        text=str(row["text"]),
        label=label,
        side=infer_side(row, forced_side),
        domain=domain,
        lang=str(row.get("lang", "")),
    )


def iter_huggingface_rows(
    *,
    dataset_name: str = "inclusionAI/NSFA_Benchmarks",
    split: str = "train",
    side: Side | None = None,
    languages: set[str] | None = None,
    id_contains: str | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> Iterable[BenchmarkRow]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("Install the benchmark extra: pip install -e '.[benchmark]'") from exc

    dataset = load_dataset(dataset_name, split=split)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    emitted = 0
    for raw in dataset:
        row = dict(raw)
        if id_contains and id_contains.lower() not in str(row.get("id", "")).lower():
            continue
        if languages and str(row.get("lang", "")) not in languages:
            continue
        try:
            parsed = row_from_mapping(row, forced_side=None)
        except ValueError:
            if side is None:
                raise
            parsed = row_from_mapping(row, forced_side=side)
        if side is not None and parsed.side is not side:
            continue
        yield parsed
        emitted += 1
        if limit is not None and emitted >= limit:
            break
