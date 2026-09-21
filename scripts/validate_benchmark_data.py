"""Validate the public NSFA benchmark files and emit strict JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jevguard_nsfa.dataset import iter_huggingface_rows


def _content_digest(rows: list[Any]) -> str:
    lines = (
        json.dumps(
            [row.id, row.text, row.label, row.side.value, list(row.domains), row.lang],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in rows
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def validate_benchmark(
    benchmark: str,
    *,
    dataset: str,
    split: str,
    revision: str | None,
    limit: int | None,
) -> dict[str, Any]:
    rows = list(
        iter_huggingface_rows(
            dataset_name=dataset,
            split=split,
            benchmark=benchmark,
            revision=revision,
            limit=limit,
        )
    )
    if not rows:
        raise ValueError(f"No rows found for benchmark={benchmark!r}")

    ids = [row.id for row in rows]
    nonblank_ids = [row_id for row_id in ids if row_id.strip()]
    labels = Counter(row.label for row in rows)
    languages = Counter(row.lang for row in rows)
    domains = Counter(domain for row in rows for domain in row.domains)
    return {
        "benchmark": benchmark,
        "rows": len(rows),
        "labels": dict(sorted(labels.items())),
        "languages": {
            "count": len(languages),
            "top10": languages.most_common(10),
        },
        "domains": dict(sorted(domains.items())),
        "empty_text": sum(not row.text.strip() for row in rows),
        "invalid_labels": sum(row.label not in (0, 1) for row in rows),
        "blank_ids": len(rows) - len(nonblank_ids),
        "duplicate_nonblank_ids": len(nonblank_ids) - len(set(nonblank_ids)),
        "content_digest": _content_digest(rows),
        "max_chars": max(len(row.text) for row in rows),
        "mean_chars": sum(len(row.text) for row in rows) / len(rows),
        "dataset": {
            "name": dataset,
            "split": split,
            "revision": revision,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="inclusionAI/NSFA_Benchmarks")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=["query", "response", "cross-source-query"],
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    benchmarks = args.benchmark or ["query", "response", "cross-source-query"]
    reports = {
        benchmark: validate_benchmark(
            benchmark,
            dataset=args.dataset,
            split=args.split,
            revision=args.dataset_revision,
            limit=args.limit,
        )
        for benchmark in benchmarks
    }
    rendered = json.dumps(
        {
            "benchmarks": reports,
            "total_rows": sum(report["rows"] for report in reports.values()),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
