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
    domains: tuple[str, ...]
    lang: str

    @property
    def domain(self) -> str | None:
        return self.domains[0] if self.domains else None


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

BENCHMARK_FILES: dict[str, tuple[str, Side]] = {
    "query": ("NSFA_Query_Multilingual.parquet", Side.QUERY),
    "response": ("NSFA_Response_Multilingual.parquet", Side.RESPONSE),
    "cross-source-query": ("NSFA_CrossSource_Query_Multilingual.parquet", Side.QUERY),
}


def canonical_domain(value: Any) -> str | None:
    domains = canonical_domains(value)
    return domains[0] if domains else None


def canonical_domains(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = str(value).strip()
    if not raw:
        return ()
    resolved: list[str] = []
    for part in raw.split(";"):
        key = _slug(part)
        if not key or key in {"no_risk", "safe", "none", "nan"}:
            continue
        domain = _DOMAIN_ALIASES.get(key)
        if domain is None:
            raise ValueError(f"Unknown NSFA L1 risk domain: {part!r}")
        if domain not in resolved:
            resolved.append(domain)
    return tuple(resolved)


def infer_side(row: dict[str, Any], forced: Side | None = None) -> Side:
    if forced is not None:
        return forced

    domains = canonical_domains(row.get("L1-Risk") or row.get("l1_risk") or row.get("domain"))
    if domains and all(domain in _QUERY_IDS for domain in domains):
        return Side.QUERY
    if domains and all(domain in _RESPONSE_IDS for domain in domains):
        return Side.RESPONSE
    if domains:
        raise ValueError(f"Row mixes query-side and response-side L1 domains: {domains!r}")

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
    domains = canonical_domains(row.get("L1-Risk") or row.get("l1_risk") or row.get("domain"))
    return BenchmarkRow(
        id=str(row.get("id", "")),
        text=str(row["text"]),
        label=label,
        side=infer_side(row, forced_side),
        domains=domains,
        lang=str(row.get("lang", "")),
    )


def iter_huggingface_rows(
    *,
    dataset_name: str = "inclusionAI/NSFA_Benchmarks",
    split: str = "train",
    benchmark: str | None = None,
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

    forced_side = side
    if benchmark is not None:
        try:
            filename, benchmark_side = BENCHMARK_FILES[benchmark]
        except KeyError as exc:
            raise ValueError(f"Unknown NSFA benchmark {benchmark!r}") from exc
        if side is not None and side is not benchmark_side:
            raise ValueError(f"Benchmark {benchmark!r} is {benchmark_side.value}-side, not {side.value}-side")
        forced_side = benchmark_side
        source = f"hf://datasets/{dataset_name}/{filename}"
        dataset = load_dataset("parquet", data_files={"train": source}, split=split)
    else:
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
            parsed = row_from_mapping(row, forced_side=forced_side)
        except ValueError:
            if forced_side is None:
                raise
            parsed = row_from_mapping(row, forced_side=forced_side)
        if forced_side is not None and parsed.side is not forced_side:
            continue
        yield parsed
        emitted += 1
        if limit is not None and emitted >= limit:
            break
