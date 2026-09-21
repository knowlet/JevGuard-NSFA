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


def build_parser() -> argparse.ArgumentParser:
    """Build the full command line parser, including every subcommand's own arguments."""
    parser = argparse.ArgumentParser(prog="jevguard-nsfa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen", help="Screen one query or response with Jev")
    screen.add_argument("text")
    screen.add_argument("--side", choices=["query", "response"], required=True)
    # Left unset so the SDK resolves the model as explicit value -> TYPESAFE_DEFAULT_MODEL -> SDK
    # default; a hardcoded default here would make the environment variable unreachable.
    screen.add_argument(
        "--model",
        default=None,
        help="TypeSafe model name; omitted means TYPESAFE_DEFAULT_MODEL or the SDK default",
    )
    screen.add_argument("--base-url", default=None)
    screen.add_argument("--threshold", type=float, default=0.5)
    screen.add_argument("--review-margin", type=float, default=0.10)
    screen.add_argument("--timeout", type=float, default=15.0)
    screen.set_defaults(handler=_screen)

    from . import benchmark_jev, benchmark_open, benchmark_singguard, compare, matrix

    bench_jev = sub.add_parser("bench-jev", help="Benchmark JevGuard on the public NSFA benchmark")
    benchmark_jev.add_arguments(bench_jev)
    bench_jev.set_defaults(handler=benchmark_jev.main_from_args)

    bench_sing = sub.add_parser("bench-singguard", help="Benchmark original SingGuard-NSFA locally")
    benchmark_singguard.add_arguments(bench_sing)
    bench_sing.set_defaults(handler=benchmark_singguard.main_from_args)

    bench_open = sub.add_parser(
        "bench-open",
        help="Benchmark Kev, Laya, Decider, or RLCD on the public NSFA benchmark",
    )
    benchmark_open.add_arguments(bench_open)
    bench_open.set_defaults(handler=benchmark_open.main_from_args)

    comp = sub.add_parser("compare", help="Render a side-by-side benchmark comparison")
    compare.add_arguments(comp)
    comp.set_defaults(handler=compare.main_from_args)

    matrix_parser = sub.add_parser("matrix", help="Render an N-way benchmark comparison matrix")
    matrix.add_arguments(matrix_parser)
    matrix_parser.set_defaults(handler=matrix.main_from_args)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
