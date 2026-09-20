"""JevGuard-NSFA command line interface."""

from __future__ import annotations

import argparse
import json

from .guard import JevGuard
from .models import Side, ThresholdPolicy


def _screen(args: argparse.Namespace) -> int:
    policy = ThresholdPolicy(default_threshold=args.threshold, review_margin=args.review_margin)
    with JevGuard(
        policy=policy,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
    ) as guard:
        result = guard.screen(args.text, Side(args.side))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="jevguard-nsfa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen", help="Screen one query or response with Jev")
    screen.add_argument("text")
    screen.add_argument("--side", choices=["query", "response"], required=True)
    screen.add_argument("--model", default="jev-latest")
    screen.add_argument("--base-url", default=None)
    screen.add_argument("--threshold", type=float, default=0.5)
    screen.add_argument("--review-margin", type=float, default=0.10)
    screen.add_argument("--timeout", type=float, default=15.0)
    screen.set_defaults(handler=_screen)

    from . import benchmark_jev, benchmark_singguard, compare

    bench_jev = sub.add_parser("bench-jev", help="Benchmark JevGuard on the public NSFA benchmark")
    benchmark_jev.add_arguments(bench_jev)
    bench_jev.set_defaults(handler=benchmark_jev.main_from_args)

    bench_sing = sub.add_parser("bench-singguard", help="Benchmark original SingGuard-NSFA locally")
    benchmark_singguard.add_arguments(bench_sing)
    bench_sing.set_defaults(handler=benchmark_singguard.main_from_args)

    comp = sub.add_parser("compare", help="Render a side-by-side benchmark comparison")
    compare.add_arguments(comp)
    comp.set_defaults(handler=compare.main_from_args)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
