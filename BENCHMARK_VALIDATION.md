# GPU benchmark validation

驗證日期：2026 年 9 月 21 日。這份報告記錄 JevGuard-NSFA 與 SingGuard-NSFA 0.8B 在同一批 NSFA benchmark sample 上的實測，以及三個官方 Parquet 的資料品質掃描。

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
| Dataset revision | 54b390c5b9c26ec40ce7f660e278d909f4dad8dc |
| Model revision | 455a72e4331b9ef37ae49154eff2a3715642c17a |
| Model weight SHA-256 | 07b9d21d13bb064d5187dfa73ab187e749ebac6cdb4d4b16058eca80ef002559 |
| Embedding size | 1024 |
| Pooler | raw LAST-token embedding; use_activation=false |

Container smoke test 回報 torch.cuda.is_available() = True、GPU = NVIDIA GB10、embedding length = 1024。正式 benchmark 期間 nvidia-smi 看見 vLLM EngineCore 約使用 17.5 GiB GPU memory，GPU utilization 曾達約 96%。

所有 SingGuard benchmark 使用 seed=42、threshold=0.5、warmup=8、max_tokens=1024、gpu_memory_utilization=0.8，以及完整官方 heads。SingGuard 成本以明示的 1.50 USD/GPU-hour 計算，這是成本情境假設。冷啟動另列，不混入穩態 latency 或 throughput。

## Dataset validation

在同一個 Docker 與 Hugging Face cache 環境完成固定 revision 的全量掃描，共 96,838 筆：

| Parquet | Rows | Label 0 / 1 | Languages | Empty text | Invalid labels | Duplicate nonblank IDs |
|---|---:|---:|---:|---:|---:|---:|
| query | 63,431 | 33,957 / 29,474 | 133 | 0 | 0 | 0 |
| response | 29,972 | 15,658 / 14,314 | 133 | 0 | 0 | 0 |
| cross-source-query | 3,435 | 1,120 / 2,315 | 133 | 0 | 0 | 2,873 |

query 與 response Parquet 的 ID 欄位大多為空；cross-source-query 有重複 ID。因此比較器使用 text、label、side、Level-1 domains、language 組成的 content fingerprint，ID digest 僅作診斷。

| Benchmark | Full-file content digest | 100-row fingerprint |
|---|---|---|
| query | 80d795d1af869753aaa8971be2ead4cb4f3a15501b059fa81ca28490245b275a | bb98428f6d42904613625d4f1b712e9c29d33e73622fe428df008a35abad02b0 |
| response | 23103d84810ea8fec144cfc27770b5c03fbd7c6c62b74a32f03b1ef01f1b6baa | 568243a67649a2a1b19200788a0a9ca634e7353a6c902225b694858080fb9620 |
| cross-source-query | f223d948ba8b718cfa0b3677158f21f83e5adc649665559690f1974461694 | 30b07270a21834a2458194e9bc9d573d94c867c8574fc436e443ff81798c0c13 |

固定版本的全量掃描輸出保存為 benchmark-results/data-quality-fixed.json；該目錄被 .gitignore 排除，避免把原始 benchmark outputs 提交到 repository。

## Fixed-revision SingGuard batch-1 comparison

下表使用三份既有 JevGuard 100-row 實測與本次固定 model/dataset revision 的 SingGuard GPU 實測。三組 JevGuard JSON 的 dataset.fingerprint 與 SingGuard 完全一致、成功數均為 100、失敗數為 0；JevGuard 舊報告的 dataset.revision 仍為 null，所以這是 content-aligned comparison，不是兩邊都在 JSON 中記錄 revision 的重跑。

JevGuard latency 是 managed API end-to-end latency；SingGuard latency 是本機 GPU inference latency，部署邊界不同。

| Benchmark | Jev F1 / accuracy | SingGuard F1 / accuracy | Jev p50 / p95 ms | SingGuard p50 / p95 ms | Jev req/s | SingGuard req/s | SingGuard cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| query | 0.8537 / 0.88 | 0.9647 / 0.97 | 313.97 / 728.13 | 45.89 / 49.92 | 7.83 | 21.67 | $0.01923 |
| response | 0.9512 / 0.96 | 0.9744 / 0.98 | 324.98 / 801.52 | 47.80 / 52.45 | 7.81 | 19.20 | $0.02170 |
| cross-source-query | 0.8955 / 0.86 | 0.8609 / 0.79 | 309.39 / 781.35 | 46.78 / 55.73 | 7.82 | 17.73 | $0.02350 |

在這組 100-row sample 上，SingGuard query 的 F1 高於 JevGuard 0.1110、response 高於 0.0231；cross-source-query 則低於 0.0346。這些是 sample estimates，不代表全母體差異。

三份 batch-1 comparator：

- benchmark-results/comparison-fixed-query-batch1.md
- benchmark-results/comparison-fixed-response-batch1.md
- benchmark-results/comparison-fixed-cross-source-query-batch1.md

## Fixed-revision SingGuard batch-16 throughput

batch-16 是 throughput test。每個 request 的 latency_ms 是整批完成等待時間，不能與 JevGuard 單請求 latency 直接比較；amortized_ms_per_sample 是批次效率指標。

| Benchmark | SingGuard F1 / accuracy | Request completion p50 / p95 ms | Amortized p50 / p95 ms | Throughput req/s | Cost / 1k |
|---|---:|---:|---:|---:|---:|
| query | 0.9647 / 0.97 | 111.18 / 150.72 | 7.34 / 9.42 | 128.72 | $0.00324 |
| response | 0.9744 / 0.98 | 127.29 / 160.35 | 8.34 / 10.02 | 117.80 | $0.00354 |
| cross-source-query | 0.8477 / 0.77 | 163.37 / 961.29 | 12.19 / 60.08 | 51.72 | $0.00806 |

cross-source-query 的長文本造成明顯 batch tail；同一固定 revision 與設定下，batch-16 的 p95 completion 為 961.29 ms、amortized p95 為 60.08 ms。不能把 amortized p95 當成每個請求的等待 latency。

三份 batch-16 comparator：

- benchmark-results/comparison-fixed-query-batch16.md
- benchmark-results/comparison-fixed-response-batch16.md
- benchmark-results/comparison-fixed-cross-source-query-batch16.md

Comparator 對 batch-16 的 quality delta 仍會輸出，並會 withheld latency delta，因為 batch size 大於 1。

## Accuracy and result validation

每份固定 SingGuard report 均通過下列獨立檢查：

- 由 TP/FP/TN/FN 重新計算 accuracy、precision、recall、F1，與 JSON 數值一致。
- confusion matrix total = successful samples = 100。
- attempted = successful + failed，且每份為 100 / 100 / 0。
- dataset.fingerprint = samples.successful_sha256。
- head_manifest.complete = true、baseline_complete = true；query/cross-source 使用 5 個 heads，response 使用 2 個 heads。
- model_revision.resolved 存在且等於 455a72e4331b9ef37ae49154eff2a3715642c17a。

可重現命令：

~~~bash
PYTHONPATH=src .venv/bin/python scripts/validate_benchmark_results.py \
  --report benchmark-results/singguard-fixed-query-100-batch1.json \
  --report benchmark-results/singguard-fixed-query-100-batch16.json \
  --report benchmark-results/singguard-fixed-response-100-batch1.json \
  --report benchmark-results/singguard-fixed-response-100-batch16.json \
  --report benchmark-results/singguard-fixed-cross-source-query-100-batch1.json \
  --report benchmark-results/singguard-fixed-cross-source-query-100-batch16.json
~~~

程式驗證已通過：git diff --check、ruff check .、完整 pytest 共 108 tests，以及 python -m compileall -q src tests scripts。

## Reproducibility and limitations

- SingGuard 的 model/head/tokenizer/vLLM 使用同一個 local snapshot；model revision 與 dataset revision 都寫入 JSON。
- checkpoint 先用 torch.load(weights_only=True, map_location=cpu) 驗證，再搬到 GPU；不 fallback 到 unrestricted pickle loading。
- JevGuard 既有實測使用相同 100-row content fingerprint，但沒有在 command line pin dataset revision，因此其 JSON revision 是 null。要取得兩邊 revision 欄位都固定的 JevGuard 結果，需重新將 sample 內容送至 .env 設定的 managed API；本次未在未明確確認外部資料傳送前執行。
- 每個 benchmark 只有 100 筆 sample，沒有 confidence interval；結果適合驗證執行路徑、資料對齊與量級，不足以宣稱全母體穩定差異。
- 0.8B、單張 NVIDIA GB10；尚未測試其他 SingGuard model sizes、GPU、量化或多實例部署。
- JevGuard 是 managed API，SingGuard 是 local GPU；latency、cost 與 throughput 不是相同部署邊界的純模型比較。
- SingGuard 成本使用 $1.50/GPU-hour 情境假設；JevGuard cost 使用報告中的 API input price，不包含所有可能的服務費用。
