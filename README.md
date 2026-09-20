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

## Benchmark JevGuard

Important: TypeSafe's current Master Customer Agreement includes a restriction on publishing benchmark or performance information about the service. Run Jev measurements locally or in a private repository unless your agreement explicitly permits publication. The included GitHub Actions benchmark job is intentionally disabled when the repository is public.

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

## Benchmark original SingGuard-NSFA

The baseline runner reconstructs the upstream real-time classification data path:

    text
      -> XML-safe untrusted_input / untrusted_output wrapper
      -> upstream chat template
      -> vLLM LAST-token embedding
      -> all matching NSFA MLP heads in parallel with torch.vmap
      -> class-1 risk probabilities

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

## Fair-comparison rules

1. Use the exact same benchmark file, shuffled seed, limit, language filter, and threshold.
2. Compare query, response, and cross-source-query separately.
3. For request latency, compare Jev end-to-end managed API latency against SingGuard batch-size-1 local inference and state that the former includes network/service overhead while the latter does not.
4. Report SingGuard model load/cold start separately.
5. Report throughput separately from request latency. SingGuard can exploit large local batches; Jev can exploit request concurrency and provider rate limits.
6. Derive Jev cost from returned token usage. Derive SingGuard compute cost only from an explicit hardware hourly price.
7. Record provider/API failures; do not drop them silently or reinterpret them as safe.
8. Tune thresholds only on a separate calibration set. Do not tune on the benchmark rows and then report those same rows as unbiased evaluation.

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
- private-repository-only GitHub Actions Jev benchmark with artifact upload

Next:
- Level-2/Level-3 diagnostic registry
- threshold calibration split and reliability diagrams
- repeated-run confidence intervals
- trajectory/tool-state extensions beyond the upstream single-turn NSFA scope
- optional served SingGuard endpoint mode to compare network-to-network latency

## Upstream

- SingGuard-NSFA: https://github.com/inclusionAI/SingGuard-NSFA
- NSFA benchmark: https://huggingface.co/datasets/inclusionAI/NSFA_Benchmarks
- SingGuard-NSFA models: https://huggingface.co/collections/inclusionAI/singguard-nsfa
- TypeSafe SDK: https://github.com/typesafe-ai/typesafe-sdk-python

This repository is an independent experiment and is not an official InclusionAI, Ant Group, or TypeSafe project.
