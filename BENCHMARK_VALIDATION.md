# GPU benchmark validation

驗證日期：2026 年 9 月 21 日。這份報告記錄 JevGuard-NSFA 與 SingGuard-NSFA 0.8B 在同一批 NSFA benchmark sample 上的 GPU 實測，以及三個官方 Parquet 的資料品質掃描。

本次結果在「SingGuard 0.8B、每個子集 100 筆」範圍內可用。2B、4B、9B 尚未執行，因此不代表完整的 SingGuard model-size frontier。

## Hardware and runtime

主機與容器均由 nvidia-smi、Docker 與 CUDA smoke test 驗證：

| 項目 | 實際值 |
|---|---|
| GPU | NVIDIA GB10 |
| NVIDIA driver | 580.173.02 |
| CUDA | 13.0 |
| Host architecture | aarch64 |
| Docker | 29.2.1 |
| Docker image | jevguard-vllm:0.24.0-cu130-arm64 |
| Image digest | sha256:2669f5fde7896fc8f4cf154656a3e7901963c5ae811c40558514ba80837915de |
| vLLM | 0.24.0 |
| PyTorch | 2.11.0+cu130 |
| Model revision | 455a72e4331b9ef37ae49154eff2a3715642c17a |
| Model weight SHA-256 | 07b9d21d13bb064d5187dfa73ab187e749ebac6cdb4d4b16058eca80ef002559 |
| Embedding size | 1024 |
| Pooler | raw LAST-token embedding; use_activation=false |

Container smoke test 回報 torch.cuda.is_available() = True、GPU = NVIDIA GB10、embedding length = 1024，並輸出 smoke_ok。正式 benchmark 期間 nvidia-smi 看見 vLLM EngineCore 約使用 17.5 GiB GPU memory，GPU utilization 曾達約 96%。

所有 benchmark 使用 seed=42、threshold=0.5、warmup=8、max_tokens=1024、gpu_memory_utilization=0.8，以及完整官方 heads。SingGuard 成本以明示的 1.50 USD/GPU-hour 計算，這是成本情境假設。

## Dataset validation

在同一個 Docker 與 Hugging Face cache 環境完成全量掃描，共 96,838 筆：

| Parquet | Rows | Label 0 / 1 | Languages | Level-1 domains |
|---|---:|---:|---:|---|
| query | 63,431 | 33,957 / 29,474 | 133 | 5 個 query domains |
| response | 29,972 | 15,658 / 14,314 | 133 | 2 個 response domains |
| cross-source-query | 3,435 | 1,120 / 2,315 | 133 | 5 個 query domains |

三個檔案均為 0 個空文字、0 個非 0/1 label。query 與 response Parquet 沒有 id 欄位；cross-source-query 有重複 ID。因此比較器以 text、label、side、Level-1 domains、language 組成的 content fingerprint 作為主要對齊條件，ID digest 僅作診斷。

全量資料 content digest 與本次 100 筆比較 sample fingerprint 如下：

| Benchmark | Full-file content digest | 100-row fingerprint |
|---|---|---|
| query | 80d795d1af869753aaa8971be2ead4cb4f3a15501b059fa81ca28490245b275a | bb98428f6d42904613625d4f1b712e9c29d33e73622fe428df008a35abad02b0 |
| response | 23103d84810ea8fec144cfc27770b5c03fbd7c6c62b74a32f03b1ef01f1b6baa | 568243a67649a2a1b19200788a0a9ca634e7353a6c902225b694858080fb9620 |
| cross-source-query | f223d948ba8b718cfa0b3677158f21f83e5adc649665559776690f1974461694 | 30b07270a21834a2458194e9bc9d573d94c867c8574fc436e443ff81798c0c13 |

The source shape also explains the ID limitation: query has 63,430 missing or blank IDs, response has 29,971 missing or blank IDs, and cross-source-query has 2,873 repeated IDs.

dataset cache 的 main reference 為 54b390c5b9c26ec40ce7f660e278d909f4dad8dc。這次 benchmark command 沒有傳入 --dataset-revision，所以 JSON 的 dataset.revision 是 null；每個報告仍保存完整 content fingerprint，且 JevGuard/SingGuard 完全一致。正式發布時應傳入固定 revision。

為支援官方資料與官方 head metadata，dataset adapter 補上 hazardous_action_output、sensitive_info_output、Dangerous_Operations_Tool_Abuse aliases，並加入回歸測試。

## Batch-1 comparison

JevGuard latency 是 managed API end-to-end latency；SingGuard latency 是本機 GPU inference latency，部署邊界不同。

| Benchmark | Jev F1 / accuracy | SingGuard F1 / accuracy | Jev p50 / p95 ms | SingGuard p50 / p95 ms | Jev req/s | SingGuard req/s |
|---|---:|---:|---:|---:|---:|---:|
| query | 0.8537 / 0.88 | 0.9647 / 0.97 | 313.97 / 728.13 | 53.93 / 65.29 | 7.83 | 16.79 |
| response | 0.9512 / 0.96 | 0.9744 / 0.98 | 324.98 / 801.52 | 45.57 / 49.52 | 7.81 | 20.08 |
| cross-source-query | 0.8955 / 0.86 | 0.8609 / 0.79 | 309.39 / 781.35 | 49.33 / 59.90 | 7.82 | 17.34 |

在 1.50 USD/GPU-hour 假設下，SingGuard steady-state cost 每 1,000 successful requests 為 query $0.02482、response $0.02075、cross-source-query $0.02403。JevGuard 報告的對應成本為 $0.04008、$0.02431、$0.04500。

SingGuard model/head cold start 為 265.1–282.2 秒，已獨立放在 cold_start，沒有混入 steady-state latency 或 throughput。

## Batch-16 throughput comparison

batch-16 是 throughput test。latency_ms 對 batch 中每個 sample 計入完整 batch completion time，因此 comparator 會隱藏它與 JevGuard 的 per-request latency delta；amortized_ms_per_sample 才是批次處理效率。

| Benchmark | Batch-1 req/s | Batch-16 req/s | Amortized p50 / p95 ms | Batch-16 cost / 1k |
|---|---:|---:|---:|---:|
| query | 16.79 | 123.04 | 7.64 / 8.05 | $0.00339 |
| response | 20.08 | 117.53 | 8.30 / 8.78 | $0.00355 |
| cross-source-query | 17.34 | 51.59 | 11.98 / 57.07 | $0.00808 |

cross-source-query 的長文本造成 batch tail：第一次 batch latency p95 為 722.59 ms，amortized p95 為 57.07 ms。相同配置重跑得到 51.42 req/s、amortized p50/p95 = 12.89/57.75 ms，且 fingerprint 相同；F1 由 0.8477 變為 0.8533。這是 100 筆 sample 的品質波動，沒有解讀成模型品質改善或退化。

## Artifacts and checks

原始 JSON 與 Markdown 報告保存在被 .gitignore 排除的 benchmark-results/：三份 JevGuard JSON、六份 SingGuard JSON、六份 comparison Markdown，以及 cross-source-query batch-16 rerun JSON。全量資料掃描可由 scripts/validate_benchmark_data.py 重現。

比較器會驗證 dataset name、split、benchmark、seed、threshold、content fingerprint、successful sample digest、head completeness、latency scope 與 batch size。三個 batch-1 comparison 的 alignment 全部為 ok；三個 batch-16 comparison 的 data alignment 為 ok，latency delta 因 batch_size=16 被正確 withheld。

通過的程式驗證：git diff --check、ruff check .、pytest；結果為 105 tests passed。

目前限制：本次歷史報告未在 command line 固定 dataset revision、每個子集只有 100 筆、沒有 confidence interval、只測 0.8B，且 JevGuard managed API 與 SingGuard local GPU 的 deployment boundary 不同。
