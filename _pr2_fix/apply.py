"""Apply reviewed, exact-anchor fixes to PR #2's pinned source tree."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def edit(path, old, new):
    target = Path(path)
    text = target.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one anchor in {path}: {old[:90]!r}")
    target.write_text(text.replace(old, new))


p = 'src/jevguard_nsfa/benchmark_jev.py'
text = Path(p).read_text()
start = text.index('async def _screen_with_attempts(')
end = text.index('async def _run_one(', start)
Path(p).write_text(text[:start] + (HERE / 'attempts.txt').read_text().rstrip() + '\n\n\n' + text[end:])
edit(p, 'rows while the attempt budget bounds how long one row can hold a slot.',
     "rows. The attempt budget starts after initial RPM admission and bounds the attempt sequence,\n"
     "including subsequent backoff and limiter waits; initial semaphore/RPM queueing is excluded\n"
     "from both the row budget and the reported logical-request latency. Cancellation is cooperative.")
edit(p, '        if rpm <= 0:\n            raise ValueError("rpm must be positive")',
     '        if not math.isfinite(rpm) or rpm <= 0:\n            raise ValueError("rpm must be finite and positive")')
edit(p, 'async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:',
     (HERE / 'pacing.txt').read_text() + 'async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:')
edit(p, '    policy = ThresholdPolicy(default_threshold=args.threshold, review_margin=args.review_margin)',
     '    minimum_pacing_seconds = check_pacing_budget(\n'
     '        len(rows), min(args.warmup, len(rows)), args.rpm,\n'
     '        getattr(args, "pacing_budget_seconds", None),\n'
     '    )\n\n'
     '    policy = ThresholdPolicy(default_threshold=args.threshold, review_margin=args.review_margin)')
edit(p, '        "mode": "managed-api-online",',
     '        "mode": "managed-api-online",\n'
     '        "execution_plan": {\n'
     '            "minimum_pacing_seconds": minimum_pacing_seconds,\n'
     '            "pacing_budget_seconds": getattr(args, "pacing_budget_seconds", None),\n'
     '            "note": "No-retry request-start lower bound, including warmup; not a runtime estimate.",\n'
     '        },')
edit(p, '    parser.add_argument("--rpm", type=float, default=900.0)',
     '    parser.add_argument("--rpm", type=float, default=900.0)\n'
     '    parser.add_argument(\n'
     '        "--pacing-budget-seconds", type=float, default=None,\n'
     '        help="Reject impossible no-retry request-start schedules before API use; not a runtime timeout",\n'
     '    )')
edit(p, '"Total seconds one row may spend across all of its attempts, including retries; "',
     '"Seconds after initial RPM admission, including attempts, retry backoff and retry RPM waits; "')
edit(p, '    if getattr(args, "row_budget", None) is not None and args.row_budget <= 0:\n'
        '        raise ValueError("--row-budget must be positive when set")',
     '    if args.warmup < 0:\n'
     '        raise ValueError("--warmup must be non-negative")\n'
     '    if args.timeout is not None and not math.isfinite(args.timeout):\n'
     '        raise ValueError("--timeout must be finite")\n'
     '    if getattr(args, "row_budget", None) is not None and (\n'
     '        not math.isfinite(args.row_budget) or args.row_budget <= 0\n'
     '    ):\n'
     '        raise ValueError("--row-budget must be positive and finite when set")\n'
     '    check_pacing_budget(0, 0, args.rpm, getattr(args, "pacing_budget_seconds", None))')
# Existing fake-clock script now observes the additional post-limiter clock read.
edit('tests/test_reporting.py', 'ticks = iter([0.0, 10.0, 10.0, 11.0])',
     'ticks = iter([0.0, 10.0, 10.0, 11.0, 11.0])')

p = 'src/jevguard_nsfa/benchmark_open.py'
edit(p, '    try:\n        for row in rows[: args.warmup]:\n            backend.screen(row, policy)',
     '    warmup_rows = rows[: args.warmup]\n'
     '    warmup_failures: list[dict[str, Any]] = []\n'
     '    try:\n'
     '        for row in warmup_rows:\n'
     '            try:\n'
     '                backend.screen(row, policy)\n'
     '            except Exception as exc:\n'
     '                warmup_failures.append({"id": row.id, "error": f"{type(exc).__name__}: {exc}"})\n'
     '                if args.fail_fast:\n'
     '                    raise')
edit(p, '        "latency_scope": "request",',
     '        "warmup": {\n'
     '            "requested": args.warmup,\n'
     '            "attempted": len(warmup_rows),\n'
     '            "successful": len(warmup_rows) - len(warmup_failures),\n'
     '            "failed": len(warmup_failures),\n'
     '            "failures": warmup_failures[:100],\n'
     '            "failure_records_truncated": len(warmup_failures) > 100,\n'
     '        },\n'
     '        "latency_scope": "request",')

p = 'scripts/validate_benchmark_results.py'
text = Path(p).read_text()
start = text.index('def _close(')
end = text.index('def validate_report(', start)
Path(p).write_text(text[:start] + '''def _finite_number(value: Any) -> float | None:
    """A JSON number, excluding booleans, that fits in a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _close(actual: float, reported: Any) -> bool:
    number = _finite_number(reported)
    return number is not None and math.isclose(actual, number, rel_tol=1e-10, abs_tol=1e-12)


''' + text[end:])
text = Path(p).read_text()
start = text.index('            for name in ("brier", "log_loss"):')
end = text.index('    if data.get("engine") == "singguard-nsfa":', start)
Path(p).write_text(text[:start] + '''            for name in ("brier", "log_loss"):
                value = binary.get(name)
                number = _finite_number(value)
                if value is not None and (number is None or number < 0):
                    errors.append(f"quality.binary.{name} must be a finite non-negative number")
            ece = binary.get("expected_calibration_error")
            number = _finite_number(ece)
            if ece is not None and (number is None or not 0.0 <= number <= 1.0):
                errors.append("quality.binary.expected_calibration_error must be in [0, 1]")

''' + text[end:])

Path('.github/workflows/benchmark-jev.yml').write_text((HERE / 'benchmark-jev.yml').read_text())
edit('pyproject.toml', 'dev = [\n', 'dev = [\n  "PyYAML>=6.0",\n')
Path('tests/test_pr2_regressions.py').write_text((HERE / 'test_pr2_regressions.py').read_text())

edit('src/jevguard_nsfa/compare.py', '        "## Method notes",\n        "",',
     '        "## Method notes",\n        "",\n'
     '        "- Deltas use unrounded report values; operands and deltas are rounded independently for display.",')

p = 'README.md'
edit(p, '下一輪擴大樣本與多模型比較（Kev、Laya、Decider-2B、Qwen RLCD）的固定測試矩陣、runtime adapter 與重現命令，請參考 [BENCHMARK_MATRIX.md](BENCHMARK_MATRIX.md)。',
     '作者回報已完成 JevGuard／SingGuard 的 query、response、cross-source-query 全量測試，以及 Decider-2b 的 cross-source-query 全量測試；原始七份 JSON 未隨 PR 附上，仍待獨立 artifact／fingerprint 複核，詳見 [BENCHMARK_VALIDATION.md](BENCHMARK_VALIDATION.md)。Kev、Laya、Qwen RLCD 仍待實測；固定測試矩陣與重現命令見 [BENCHMARK_MATRIX.md](BENCHMARK_MATRIX.md)。')
edit(p, 'The first full-set round (query 63,431, response 29,972, cross-source-query 3,435) is recorded in [BENCHMARK_VALIDATION.md](BENCHMARK_VALIDATION.md): all three subsets are zero-failure pairs, and it includes a real Decider-2b cross-source run.',
     'Author-reported full-set measurements are recorded in [BENCHMARK_VALIDATION.md](BENCHMARK_VALIDATION.md): JevGuard and SingGuard on query 63,431, response 29,972 and cross-source-query 3,435, plus Decider-2b on cross-source-query only. The seven raw JSON reports are not included in this PR, so zero-failure/alignment claims remain pending independent artifact and fingerprint verification. Kev, Laya and Qwen RLCD remain unmeasured.')

p = 'BENCHMARK_VALIDATION.md'
edit(p, '## Full-set benchmark round (2026-09-21)',
     '## Full-set benchmark round (2026-09-21; author-reported)\n\n'
     '> **Evidence status / 證據狀態：待獨立複核。** 下列為作者回報的執行結果，不是本 PR 已提供可獨立核驗的 artifacts。七份原始 JSON 位於被 gitignore 排除的 `benchmark-results/`，未隨本 PR 提供；因此零失敗、fingerprint／alignment 與效能結論都仍待最終 artifact 複核。範圍是 JevGuard／SingGuard 各三個 subset，加上 Decider-2b 的 cross-source-query，並非 Decider 的三個 subset 都已完成。\n\n'
     '要升級為可核驗結果，須發布這七份 JSON、各檔 SHA-256、source commit、完整執行參數、runtime／model／dataset revision 及硬體紀錄，再重跑下列 validator 與 comparison。Validator 可從 confusion matrix 重算分類指標，但 calibration 指標目前僅做數值／值域檢查；重新計算 Brier／log loss／ECE 仍需逐筆 label／probability。\n\n'
     '**Rounding rule:** comparison deltas are computed from unrounded JSON values; operands and deltas are rounded independently for display. Subtracting displayed four-decimal operands can differ by one unit in the last digit. The historical table has not been regenerated without its raw reports; its numerical claims remain pending artifact verification. Do not replace its deltas with differences of already-rounded operands.')
edit(p, '上方 100 筆的結果保留為歷史驗證，這一輪才是目前的主要結論。',
     '上方 100 筆的結果保留為歷史驗證；這一輪是作者回報的主要結果，獨立結論須等原始 artifacts 複核後才能確認。')
# Scope the later verification claims explicitly without altering the historical numbers.
edit(p, '### 驗證\n\n- 七份納入結果表的報告都通過',
     '### 作者回報的驗證（待原始 artifacts 獨立複核）\n\n- 作者回報七份納入結果表的報告都通過')

p = 'BENCHMARK_MATRIX.md'
edit(p, 'This document defines the next comparison round for JevGuard-NSFA.',
     'This document defines the expanded comparison protocol and remaining follow-up for JevGuard-NSFA.')
edit(p, 'The first full-set round has now executed on this machine:',
     'The author reports the following full-set runs. The seven original JSON artifacts are not included in this PR; zero-failure, fingerprint/alignment and performance claims remain pending independent verification. Decider-2b was run on cross-source-query only, not query/response:')
with Path(p).open('a') as out:
    out.write('''

## Workflow and deadline semantics

The manual Jev workflow allocates 120 minutes for full runs (110-minute benchmark step),
otherwise 60 minutes (50-minute step). It passes a 6,000/2,400-second no-retry pacing
budget to `bench-jev`. After selecting the actual dataset rows, the runner rejects
impossible schedules before constructing the API client. This is a necessary-condition
check, not a guarantee against slow inference, retries, dataset downloads or setup time.
The remaining time is reserved for long tails, report validation and upload; a cancelled
job can still lose an unfinished report because checkpoint/resume is not implemented.
Full runs explicitly use `--timeout 180 --row-budget 600 --retries 2`, matching the
reported full-set protocol. Smaller runs retain the 15-second/no-retry defaults.

`--row-budget` starts after initial RPM admission. It includes the first attempt,
all retry attempts, backoff and retry RPM waits, but excludes the initial semaphore/RPM
queue. Each attempt's HTTP timeout is clamped to the remaining budget after admission.
An outer cooperative asyncio deadline cancels an expired request or limiter wait.
External cancellation propagates; operational errors never become safe verdicts.

Open-engine warmup failures follow `--fail-fast` and are recorded separately in
`warmup`, outside measured sample counts, fingerprints, quality and latency. Model
initialization failures still abort. A valid JSON report is not sufficient for publishing
zero-failure results: the workflow also requires zero failed measured rows and treats
missing artifacts as errors.
''')
print('Applied PR #2 review fixes; original benchmark numbers were not changed.')
