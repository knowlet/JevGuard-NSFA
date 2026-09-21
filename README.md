# JevGuard-NSFA

JevGuard-NSFA is an experimental System One implementation of the SingGuard-NSFA agent-security taxonomy. It asks independent Jev Noul questions for each NSFA Level-1 risk domain, then applies deterministic threshold and review policy in code.

The project is intentionally benchmark-first: it compares JevGuard against the original SingGuard-NSFA real-time classification path on the same public NSFA benchmark subsets without inventing or copying performance numbers.

## Scope

The current MVP implements the seven published NSFA Level-1 domains:

Query:
- Prompt Injection & Jailbreak
- Malicious Code & Cyberattack
- Sensitive Information Stealing
- Dangerous Operations & Tool Abuse
- Resource Abuse

Response:
- Hazardous Action Generation
- Sensitive Information Leakage

Each domain is an independent Noul probability. This preserves multi-label behavior; a sample can be positive for more than one risk domain.

The public NSFA benchmark exposes text, a binary label, Level-1 risk labels, and language. Level-2/Level-3 expansion is therefore treated as diagnostic taxonomy work rather than a prerequisite for an apples-to-apples public benchmark.

## Architecture

    untrusted query / response
              |
              v
      Jev System One request
      independent parallel Nouls
              |
              v
      per-domain probabilities
              |
       +------+------+
       |             |
       v             v
    binary      review-band
    threshold    ambiguity
       |             |
       v             v
    allow/block   optional System Two
                  semantic review only

Provider, authentication, quota, timeout, malformed-response, and transport failures are not semantic ambiguity. They propagate as failures and never silently trigger the System-Two fallback.

## Install

Core JevGuard:

    python -m pip install -e .

Benchmark Jev against the public dataset:

    python -m pip install -e ".[benchmark]"

Benchmark the original local SingGuard classifier on a CUDA machine:

    python -m pip install -e ".[singguard]"

Development:

    python -m pip install -e ".[dev]"
    ruff check .
    pytest

## Credentials

Set:

    export TYPESAFE_API_KEY=...

Optional:

    export TYPESAFE_BASE_URL=https://api.typesafe.ai
    export TYPESAFE_DEFAULT_MODEL=jev-latest

The official TypeSafe SDK is used directly. No OpenAI-compatible wrapper is required for Jev.

The `--model` option is unset by default on both CLI paths, so the SDK resolves the model
in the order explicit `--model` -> `TYPESAFE_DEFAULT_MODEL` -> SDK default. Pass `--model`
only when you want to override the environment.

## Screen one item

    jevguard-nsfa screen "Ignore all previous instructions..." --side query

The JSON result includes all side-specific domain scores, binary unsafe state, operational allow/review/block decision, model name, end-to-end latency, and API-reported token usage.

## Public benchmark subsets

The upstream Hugging Face dataset repository publishes three separate Parquet files:

- query: NSFA_Query_Multilingual.parquet
- response: NSFA_Response_Multilingual.parquet
- cross-source-query: NSFA_CrossSource_Query_Multilingual.parquet

JevGuard loads these files separately rather than relying on the concatenated train split. This is required because safe rows have no risk domain from which query-vs-response side can be inferred.

The adapter also preserves semicolon-separated multi-label Level-1 ground truth.

Validate the complete cached dataset independently of model scoring:

    PYTHONPATH=src python scripts/validate_benchmark_data.py \
      --dataset-revision <commit> \
      --output benchmark-results/data-quality.json

## Benchmark JevGuard

Example, 1,000 query samples:

    jevguard-nsfa bench-jev \
      --benchmark query \
      --limit 1000 \
      --concurrency 8 \
      --rpm 900 \
      --output benchmark-results/jev-query.json

Response benchmark:

    jevguard-nsfa bench-jev --benchmark response --limit 1000

Cross-source query benchmark:

    jevguard-nsfa bench-jev --benchmark cross-source-query --limit 1000

The Jev runner reports:
- binary accuracy, precision, recall and F1
- false-positive and false-negative rates
- Brier score using max domain risk as the binary risk score
- positive-sample Level-1 top-domain accuracy
- per-domain one-vs-rest metrics
- latency mean/p50/p95/p99/min/max
- wall-clock throughput
- failed requests
- API-reported input/output token counts
- exact token-based API cost under the configured input-token price

The default Jev price assumption is 0.042 USD per million input tokens. It is a benchmark parameter, not a hard-coded economic claim:

    --input-price-per-million 0.042

Change it whenever current pricing changes.

Rate limiting and retries are both owned by the runner, not by the SDK: the client is
created with SDK-internal retries disabled, every attempt (including a retry) takes a
`--rpm` slot, and `Retry-After` is honoured when the API asks for a specific wait. Rows
whose retries are exhausted stay recorded failures; they are never reinterpreted as safe
and never reach the semantic review fallback. The report therefore adds
`samples.request_attempts` (attempts including retries) and `samples.retried_requests`
next to `samples.failed`.

Pin the dataset content with `--dataset-revision <ref>` for a reproducible comparison; the
requested revision is recorded as `dataset.revision`. Every report also carries a
`dataset.fingerprint` over the full sample content (id, text, label, side, L1 domains and
language) plus `samples.successful_sha256` over exactly the samples that were scored.

## Benchmark original SingGuard-NSFA

The baseline runner reconstructs the upstream real-time classification data path:

    text
      -> XML-safe untrusted_input / untrusted_output wrapper
      -> upstream chat template
      -> vLLM LAST-token embedding
      -> all matching NSFA MLP heads in parallel with torch.vmap
      -> class-1 risk probabilities

The baseline asks the pooler for the raw LAST-token hidden state. Recent vLLM
releases replaced the `normalize` switch with `use_activation`, so the runner
tries the modern switch first, checks on the constructed configuration that
activation and normalisation are really disabled, and fails loudly when it
cannot verify that. A silently activated pooler would change the vectors the
NSFA heads were trained on, and every quality number derived from them.

`latency_ms` is the per-request latency: a sample grouped into a batch waits for
the whole batch to finish, so a batch of 4 samples taking 80 ms is reported as
80 ms per request, not 20 ms. Online comparison against the managed API is
therefore only meaningful at `--batch-size 1`. The per-sample amortised
processing time is reported separately as `amortized_ms_per_sample`, and
`batch_latency_ms` keeps the batch wall time. Preparing the parallel-head
execution state is a one-off load-time cost, reported under `cold_start`, and is
never folded into steady-state latency.

The runner refuses an incomplete classification-head set. For the side being benchmarked
it requires every NSFA Level-1 domain of that side, fails before the model is loaded when a
head file is missing or a head's task metadata is mislabeled, and records the real coverage
in `head_manifest` (`expected_domains`, `loaded_domains`, `missing_domains`,
`unexpected_domains`, `complete`, `head_count`). A head that was never run can therefore no
longer enter the metrics as a 0.0 risk probability. `--allow-partial-heads` exists only for
deliberate partial-head experiments: those runs are reported with
`head_manifest.complete=false` and `baseline_complete=false`, and the comparator withholds
quality deltas instead of presenting them as a full baseline. `--dataset-revision` pins and
records the dataset revision exactly as in the Jev runner.

Online-style latency comparison should use batch size 1:

    jevguard-nsfa bench-singguard \
      --benchmark query \
      --model inclusionAI/SingGuard-NSFA-0.8B \
      --batch-size 1 \
      --limit 1000 \
      --gpu-hourly-usd 1.50 \
      --output benchmark-results/singguard-query.json

Throughput-oriented comparison can also be run separately:

    jevguard-nsfa bench-singguard \
      --benchmark query \
      --model inclusionAI/SingGuard-NSFA-0.8B \
      --batch-size 256 \
      --limit 10000

The runner keeps model/head load time separate from steady-state inference. Local inference is not labeled free: when a GPU hourly price is supplied, the report converts occupied GPU time to allocated compute cost per 1,000 successful samples.

Run the 0.8B, 2B, 4B, and 9B variants separately if you want the complete SingGuard size/performance frontier.

## Compare results

    jevguard-nsfa compare \
      --jev benchmark-results/jev-query.json \
      --singguard benchmark-results/singguard-query.json \
      --output benchmark-results/query-comparison.md

The comparison includes F1, precision, recall, accuracy, calibration, Level-1 accuracy, p50/p95/p99 latency, throughput, cost per 1,000 requests, and failures.

The comparator verifies alignment before it shows any delta: dataset name, split,
benchmark, language filter, id filter, seed, dataset revision, the content fingerprint of
the attempted selection, threshold, attempted sample count, the content digest of the
samples that were actually scored, SingGuard head-set completeness, the latency scope, and
the SingGuard batch size. Because the fingerprint covers the sample text and the L1
ground-truth domains, revising either one makes two runs incomparable instead of looking
aligned. Every check is listed in an
`## Alignment` section as ok, mismatch, or unknown. The data checks gate the
quality deltas and the latency checks gate the latency deltas: a partial Jev
failure can no longer masquerade as a paired quality difference, while a
throughput run with a large batch size still reports its quality delta and only
loses the latency delta. Per-engine values are always printed; a withheld delta
renders as n/a, the reason is stated in the report, and a warning is printed to
stdout.

本機 NVIDIA GB10 的既有實測、資料品質掃描與 JevGuard 對 SingGuard 比較，請參考 [BENCHMARK_VALIDATION.md](BENCHMARK_VALIDATION.md)。

下一輪擴大樣本與多模型比較（Kev、Laya、Decider-2B、Qwen RLCD）的固定測試矩陣、runtime adapter 與重現命令，請參考 [BENCHMARK_MATRIX.md](BENCHMARK_MATRIX.md)。

### Expanded benchmark engines

The generic runner normalizes open decision engines to the same NSFA Level-1 probability vector:

    # Kev or Decider: serve the upstream TypeSafe-compatible endpoint first.
    jevguard-nsfa bench-open \
      --backend systemone-http \
      --engine kev-4b \
      --base-url http://127.0.0.1:8009 \
      --benchmark cross-source-query \
      --limit 1000

    # Laya: routed English/multilingual checkpoints.
    jevguard-nsfa bench-open \
      --backend laya \
      --engine laya-routed \
      --model routed \
      --device cuda \
      --benchmark cross-source-query \
      --limit 1000

    # Qwen RLCD: upstream /api/run-parallel server.
    jevguard-nsfa bench-open \
      --backend rlcd-http \
      --engine qwen-2.5-1b-rlcd \
      --base-url http://127.0.0.1:8000 \
      --benchmark query \
      --limit 1000

Use `--full` on `bench-jev`, `bench-singguard`, or `bench-open` to score the entire selected subset. New reports include Brier score, log loss, and 10-bin expected calibration error in addition to the existing classification metrics.

Render an aligned N-way matrix with:

    jevguard-nsfa matrix \
      --report Jev=benchmark-results/jev.json \
      --report SingGuard=benchmark-results/singguard.json \
      --report Kev=benchmark-results/kev.json \
      --report Laya=benchmark-results/laya.json \
      --report Decider=benchmark-results/decider.json \
      --report RLCD=benchmark-results/rlcd.json

## Fair-comparison rules

1. Use the exact same benchmark file (pin `--dataset-revision` when the upstream revision matters), shuffled seed, limit, language filter, and threshold, and let the comparator confirm the alignment. It refuses unpaired deltas.
2. Compare query, response, and cross-source-query separately.
3. For request latency, compare Jev end-to-end managed API latency against SingGuard batch-size-1 local inference and state that the former includes network/service overhead while the latter does not.
4. Report SingGuard model load/cold start separately.
5. Report throughput separately from request latency. SingGuard can exploit large local batches; Jev can exploit request concurrency and provider rate limits. Jev `samples.request_attempts` counts retries, so compare it with `samples.attempted` when reasoning about rate limits.
6. Derive Jev cost from returned token usage. Derive SingGuard compute cost only from an explicit hardware hourly price.
7. Record provider/API failures; do not drop them silently or reinterpret them as safe. A quality difference is only paired when both runs scored the same sample content; otherwise the comparator withholds the delta.
8. Tune thresholds only on a separate calibration set. Do not tune on the benchmark rows and then report those same rows as unbiased evaluation.
9. Report undefined metrics as `null`, never `NaN`; reports are strict JSON (`allow_nan=False`) so downstream consumers can parse them.

## Published upstream reference points

These are upstream claims, not JevGuard measurements:

- SingGuard-NSFA publishes 7 Level-1, 28 Level-2, and 185 Level-3 risk variants.
- Its real-time classification mode uses a frozen backbone plus parallel classification heads and reports roughly 45-57 ms per sample on a single NVIDIA A100.
- The public benchmark contains 63,431 query samples, 29,972 response samples, and 3,435 cross-source query samples across 133 languages.

Do not put those numbers in a Jev-vs-SingGuard result table until this repository has produced a reproducible local run on stated hardware.

## Project status

MVP:
- seven independent NSFA Level-1 Jev decisions
- multi-label benchmark adapter
- semantic-only System-Two fallback hook
- Jev managed-API benchmark
- original SingGuard realtime baseline runner
- cost, latency, quality, throughput and failure metrics
- Markdown comparison report
- unit tests and CI
- manual GitHub Actions Jev benchmark with artifact upload

Next:
- Level-2/Level-3 diagnostic registry
- threshold calibration split and reliability diagrams
- repeated-run confidence intervals
- expanded Kev/Laya/Decider/RLCD NSFA matrix with 500/1,000/full-set runs
- trajectory/tool-state extensions beyond the upstream single-turn NSFA scope
- optional served SingGuard endpoint mode to compare network-to-network latency

## Upstream

- SingGuard-NSFA: https://github.com/inclusionAI/SingGuard-NSFA
- NSFA benchmark: https://huggingface.co/datasets/inclusionAI/NSFA_Benchmarks
- SingGuard-NSFA models: https://huggingface.co/collections/inclusionAI/singguard-nsfa
- TypeSafe SDK: https://github.com/typesafe-ai/typesafe-sdk-python
- Kev: https://github.com/jaredpalmer/kev
- Laya: https://github.com/NandhaKishorM/laya
- Decider-2B: https://huggingface.co/Mapika/decider-2b
- Qwen-2.5-1B-RLCD: https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD

This repository is an independent experiment and is not an official InclusionAI, Ant Group, or TypeSafe project.
