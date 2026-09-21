# Expanded NSFA benchmark matrix

This document defines the next comparison round for JevGuard-NSFA. It expands the original JevGuard vs SingGuard validation to larger sample sizes and additional typed-decision engines without treating unlike runtime boundaries as interchangeable.

## Engines

| Engine | Integration | Primary target | Important caveat |
|---|---|---|---|
| JevGuard | managed TypeSafe API | current configured Jev model | includes network/service latency and API cost |
| SingGuard-NSFA | local embedding + NSFA heads | inclusionAI/SingGuard-NSFA-0.8B initially | specialised NSFA classifier; batch throughput is separate from online latency |
| Kev | TypeSafe-compatible HTTP | jaredpalmer/kev-4b | current Kev family also includes 0.8B and 9B; pin the served checkpoint |
| Laya | in-process Python | routed English/multilingual checkpoints | routed mode is the fair multilingual default; explicit checkpoints are diagnostic slices |
| Decider | TypeSafe-compatible HTTP | Mapika/decider-2b | model card is English-only, so report English and multilingual runs separately |
| Qwen RLCD | parallel-constrained HTTP | harshatheg/Qwen-2.5-1B-RLCD | different decoding architecture and Apple-Silicon-oriented runtime; do not treat its latency as same-hardware GB10 latency |

Kev and Decider use the same System-One request shape as Jev. Laya answers the same NSFA Noul questions directly. Qwen RLCD receives one boolean schema field per NSFA Level-1 domain. All adapters must produce a complete probability vector for every domain on the benchmarked side; a missing field is a failed sample, never an implicit probability of zero.

## Dataset and sample schedule

Pin the NSFA dataset revision for every comparable run. The revision used by the existing validation is:

    54b390c5b9c26ec40ce7f660e278d909f4dad8dc

The official subsets currently contain:

| Subset | Rows |
|---|---:|
| query | 63,431 |
| response | 29,972 |
| cross-source-query | 3,435 |

Run the following stages with seed 42 as the primary paired comparison:

| Stage | Query | Response | Cross-source-query | Purpose |
|---|---:|---:|---:|---|
| smoke | 100 | 100 | 100 | adapter/runtime validation |
| medium | 500 | 500 | 500 | detect large regressions cheaply |
| primary | 1,000 | 1,000 | 1,000 | main expanded comparison |
| full cross-source | - | - | 3,435 | mandatory follow-up to the current Jev cross-source lead |
| full corpus | 63,431 | 29,972 | 3,435 | optional when API/GPU cost is acceptable |

After the seed-42 primary runs, repeat the 1,000-row runs with seeds 7 and 1337 for sampling-stability diagnostics. Keep each seed as a separate aligned matrix; do not merge different content fingerprints into one paired delta.

The new `--full` flag is the canonical way to score an entire selected subset:

    jevguard-nsfa bench-jev \
      --benchmark cross-source-query \
      --dataset-revision 54b390c5b9c26ec40ce7f660e278d909f4dad8dc \
      --full

    jevguard-nsfa bench-singguard \
      --benchmark cross-source-query \
      --dataset-revision 54b390c5b9c26ec40ce7f660e278d909f4dad8dc \
      --full \
      --batch-size 1

## Language slices

NSFA is multilingual, so one aggregate number is not sufficient for engines with different language coverage.

For every primary engine run, keep:

1. the full multilingual subset;
2. an English-only slice with `--language en`;
3. additional per-language diagnostics only after the data-quality report confirms the exact language code and enough positive/negative examples.

Laya should use routed mode as its primary multilingual configuration because its upstream Router selects English vs multilingual checkpoints before inference. Also run the explicit English and multilingual checkpoints as diagnostics.

Decider-2B documents English as its supported language. Its multilingual score is still useful as an out-of-distribution measurement, but it must not be described as supported-language performance.

## Metrics

Every successful backend is normalized to the same report schema and should include:

- binary accuracy, precision, recall and F1;
- false-positive and false-negative rates;
- Brier score;
- log loss;
- expected calibration error (10 equal-width probability bins);
- positive-sample Level-1 top-domain accuracy;
- per-domain one-vs-rest quality;
- request latency p50/p95/p99 where the runtime boundary is meaningful;
- throughput with an explicit note about sequential, concurrent, or batched execution;
- cost per 1,000 requests when an explicit API price or hardware hourly price is supplied;
- attempted/successful/failed sample counts;
- dataset content fingerprint and dataset revision;
- requested model revision and, when the backend can prove it, resolved artifact revision.

Do not tune thresholds on the benchmark rows. Any threshold calibration must use a separate calibration split.

## Runtime adapters

### Kev

Start a pinned Kev server using the upstream repository. Kev exposes `POST /v1/systemone` in the TypeSafe request format.

Primary run:

    jevguard-nsfa bench-open \
      --backend systemone-http \
      --engine kev-4b \
      --base-url http://127.0.0.1:8009 \
      --model kev-latest \
      --model-revision <pinned-kev-commit-or-release> \
      --benchmark cross-source-query \
      --dataset-revision 54b390c5b9c26ec40ce7f660e278d909f4dad8dc \
      --limit 1000 \
      --output benchmark-results/kev-4b-cross-1000.json

Use Kev-4B as the primary comparison. Kev-0.8B and Kev-9B can be run as a size/performance frontier after the primary matrix is complete.

### Laya

Install Laya in the benchmark environment, then use routed mode for the multilingual primary:

    jevguard-nsfa bench-open \
      --backend laya \
      --engine laya-routed \
      --model routed \
      --device cuda \
      --benchmark cross-source-query \
      --dataset-revision 54b390c5b9c26ec40ce7f660e278d909f4dad8dc \
      --limit 1000 \
      --output benchmark-results/laya-routed-cross-1000.json

The adapter preloads English and multilingual checkpoints before measurement so language switches do not include model reload time.

Diagnostic explicit checkpoints:

    --model english
    --model multilingual
    --model typed-decisions

### Decider-2B

Serve `Mapika/decider-2b` using its upstream TypeSafe-compatible server and pin the model artifact used by that server:

    jevguard-nsfa bench-open \
      --backend systemone-http \
      --engine decider-2b \
      --base-url http://127.0.0.1:8000 \
      --model Mapika/decider-2b \
      --model-revision <pinned-hugging-face-commit> \
      --benchmark query \
      --language en \
      --dataset-revision 54b390c5b9c26ec40ce7f660e278d909f4dad8dc \
      --limit 1000 \
      --output benchmark-results/decider-2b-query-en-1000.json

Run a full multilingual counterpart with the same sample size and seed, but label it explicitly as an out-of-distribution multilingual measurement.

### Qwen-2.5-1B-RLCD

The upstream server exposes parallel constrained decoding at `/api/run-parallel`. The adapter translates each NSFA Level-1 Noul into one boolean schema field and extracts the probability of `true`.

    jevguard-nsfa bench-open \
      --backend rlcd-http \
      --engine qwen-2.5-1b-rlcd \
      --base-url http://127.0.0.1:8000 \
      --model harshatheg/Qwen-2.5-1B-RLCD \
      --model-revision <pinned-hugging-face-commit> \
      --benchmark query \
      --dataset-revision 54b390c5b9c26ec40ce7f660e278d909f4dad8dc \
      --limit 1000 \
      --output benchmark-results/rlcd-query-1000.json

The repository name says "1B", while its model card identifies a Qwen2.5-1.5B-Instruct base and an Apple-Silicon/MLX deployment target. Record the exact upstream revision and runtime instead of inferring parameter count or hardware comparability from the repository name.

## Comparison matrix

Render any number of aligned reports:

    jevguard-nsfa matrix \
      --report Jev=benchmark-results/jev-cross-1000.json \
      --report SingGuard=benchmark-results/singguard-cross-1000.json \
      --report Kev=benchmark-results/kev-4b-cross-1000.json \
      --report Laya=benchmark-results/laya-routed-cross-1000.json \
      --report Decider=benchmark-results/decider-2b-cross-1000.json \
      --report RLCD=benchmark-results/rlcd-cross-1000.json \
      --output benchmark-results/cross-1000-matrix.md

The matrix reports whether quality and latency are actually comparable. Quality requires matching benchmark, attempted content fingerprint, threshold, non-conflicting pinned dataset revisions, successful sample count, and successful-sample digest. Latency additionally requires request-level timing and no batch size greater than one.

A matching F1 table does not by itself make latency an apples-to-apples model comparison: managed API, in-process Python, local HTTP, local GPU classification, and Apple MLX include different runtime boundaries.

## Throughput protocol

The generic `bench-open` runner is intentionally sequential. Its throughput is a sequential service-rate observation, not a maximum-capacity load test.

For final throughput numbers:

- Jev: use explicit request concurrency and provider RPM limits;
- SingGuard: publish batch-1 online latency separately from larger-batch throughput;
- Kev/Decider: add a native concurrent HTTP load run after correctness is validated;
- Laya: measure multi-question single-forward-pass behavior separately from request concurrency;
- RLCD: report the upstream parallel-constrained request latency and any separate concurrency test on the hardware that actually supports its runtime.

Do not put these different throughput modes in a single "winner" column without the runtime note.

## Validation gates

Before accepting a result into the published matrix:

1. dataset fingerprint must be the expected selection;
2. zero-failure runs must have `dataset.fingerprint == samples.successful_sha256`;
3. TP/FP/TN/FN must independently reproduce accuracy, precision, recall and F1;
4. Brier/log-loss must be finite and non-negative; ECE must be in [0, 1];
5. every backend must emit all expected NSFA domains for the selected side;
6. model/dataset revision provenance must be recorded as far as the runtime can actually verify it;
7. local and remote failures remain failures; no safety verdict is invented as a fallback;
8. batch throughput latency must not be compared with online request latency.

Use:

    PYTHONPATH=src python scripts/validate_benchmark_results.py \
      --report benchmark-results/<report>.json

## Current status

This PR adds the reusable benchmark adapters, larger/full-dataset selection, N-way matrix comparator, calibration metrics, and unit tests. The first full-set round has now executed on this machine:

- cross-source-query, full 3,435 rows: JevGuard, SingGuard and Decider-2b all zero-failure with matching fingerprints and successful-sample digests, so quality and latency deltas are both comparable;
- query, full 63,431 rows, and response, full 29,972 rows: both engines zero-failure and fully paired, so the comparator reports deltas for both;
- the JevGuard runs needed a runner fix first: `--timeout` was both the per-attempt timeout and the whole row budget, so a single timeout consumed the retry budget. `--row-budget` now separates them (defaulting to `--timeout`); the reruns finished 63,431/63,431 and 29,972/29,972 with 2 and 5 retried rows;
- Decider-2b: served locally from the pinned `Mapika/decider-2b` snapshot through `POST /v1/systemone`, so the `systemone-http` adapter now has real-runtime evidence instead of only fake-runtime tests.

Concrete numbers, alignment output and the remaining gaps are in [BENCHMARK_VALIDATION.md](BENCHMARK_VALIDATION.md). Kev, Laya and Qwen-2.5-1B-RLCD still have no pinned runtime on this machine, so no accuracy numbers are claimed for them. The 100-row JevGuard/SingGuard measurements remain historical validation evidence.
