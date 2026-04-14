# TriAttention design-code research results

Audit date: 2026-04-13

Primary upstream URLs:
- Repo: https://github.com/WeianMao/triattention
- Paper: https://arxiv.org/abs/2604.04921
- Project page: https://weianmao.github.io/tri-attention-project-page/

Scope audited here: `README.md`, `docs/results.md`, `docs/reproduction.md`, experiment scripts/configs under `scripts/` and `triattention/configs/`, shipped calibration assets, existing artifacts under `experiments/`, and `notes/triattention-design-code-env-check.md`.

## Claim inventory

1. **Headline performance claim**
   - Surface: `README.md`
   - Claim text: TriAttention "compress[es] KV cache by **10.7x** and boost[s] throughput by **2.5x** on long reasoning tasks -- with **no accuracy loss**."
   - Supporting figure surfaces: `docs/assets/tradeoff.png`, `docs/results.md`

2. **AIME24 / AIME25 accuracy table**
   - Surfaces: `README.md`, `docs/results.md`
   - Claims reported for `fullkv`, `snapkv`, `r1kv`, and `triattention` across:
     - `Qwen3-8B`
     - `DeepSeek-R1-Distill-Llama-8B`
     - `DeepSeek-R1-Distill-Qwen-7B`
     - `GPT-OSS-20B`

3. **MATH-500 accuracy table**
   - Surface: `docs/results.md`
   - Same method family comparison; includes `GPT-OSS-20B` in the table.

4. **Throughput table**
   - Surfaces: `README.md`, `docs/results.md`
   - Specific claimed rows:
     - MATH-500: `6.3x`
     - AIME24: `1.9x`
     - AIME25: `2.5x`
   - Claimed to compare `Full Throughput` vs `TriAttn Throughput` for `Qwen3-8B`.

5. **Accuracy-vs-budget and DFS memory-retention claim**
   - Surface: `docs/results.md`
   - Supporting figure surface: `docs/assets/results.png`
   - Claim text says TriAttention outperforms baselines across budget levels and is comparable to Full Attention on Recursive State Query / DFS memory-retention style evaluation.
   - Related benchmark assets exist under `triattention/benchmarks/dfs/`.

6. **Reproduction/readiness claim surface**
   - Surfaces: `docs/reproduction.md`, `README.md`, `docs/calibration.md`, `docs/environment.md`
   - These imply that the tables are rerunnable through `scripts/cli.py`, `scripts/experiments/`, shipped calibration files, and the documented environment.

7. **Operational execution claim in current fork**
   - Surface: `notes/triattention-design-code-env-check.md`
   - This is not a paper claim, but it is the strongest local evidence that the HuggingFace path runs in practice.

## Support matrix

| Claim | Repo surface(s) | Direct backing artifact found in repo | What is needed to rerun | Current status |
|---|---|---|---|---|
| 2.5x throughput on AIME25 while matching Full Attention accuracy | `README.md`, `docs/results.md`, `docs/assets/tradeoff.png` | No raw throughput logs, CSV, JSON, or benchmark output artifact found under `experiments/`; only figure/table surfaces | Need benchmark outputs for `fullkv` and `triattention` plus the throughput measurement harness, which is not clearly surfaced in repo scripts | **Stated only** |
| 10.7x KV memory reduction | `README.md`, `docs/assets/tradeoff.png` | No raw memory/KV-budget measurement artifact found | Need the exact measurement script or logging path; not surfaced in audited repo artifacts | **Stated only** |
| AIME24/AIME25 accuracy table for Qwen3-8B, DS-Llama-8B, DS-Qwen-7B | `README.md`, `docs/results.md` | No checked-in merged benchmark outputs or eval JSON/JSONL found; only smoke-test artifacts under `experiments/smoke_env/` | Need datasets, model weights, usable TriAttention stats path, generation, merge, and fixed evaluation stage | **Workflow partly exists, benchmark evidence absent** |
| MATH-500 accuracy table for supported HF models | `docs/results.md` | No checked-in MATH benchmark outputs/evals found | Same as above, plus a local `math.jsonl`/`math500` dataset path, which is currently missing in expected locations | **Stated only; currently blocked** |
| GPT-OSS-20B table entries | `README.md`, `docs/results.md` | No GPT-OSS model entry in `scripts/cli.py`, no GPT-OSS stats, configs, outputs, or logs found | A reproducible path is not exposed in the current repo | **Unsupported by current repo artifacts** |
| Accuracy-vs-budget curves | `docs/results.md`, `docs/assets/results.png`, `triattention/configs/shared/budgets.yaml`, `scripts/experiments/*/budget*/...` | Budget sweep wrappers/configs exist, but no saved sweep results were found under `experiments/outputs/` | Need working sweep runs, stats resolution, merge, and eval | **Partial workflow support only** |
| DFS / Recursive State Query memory-retention result | `docs/results.md`, `triattention/benchmarks/dfs/README.md`, `triattention/benchmarks/dfs/datasets/*`, `triattention/benchmarks/dfs/scripts/eval_dfs_state_query.py` | Benchmark assets exist, but no result JSON/JSONL/log artifact was found | Need the exact run harness and output provenance for the figure/table point | **Benchmark assets exist; result evidence missing** |
| Core HF execution path works locally | `notes/triattention-design-code-env-check.md`, `experiments/smoke_env/fullkv/*`, `experiments/smoke_env/triattention/*`, `scripts/merge_shards.py` | Yes: real smoke-test outputs, meta files, run logs, and a merged JSONL exist | Already rerunnable for smoke scope; this is the most credible current support | **Directly backed** |

## Reproducibility readiness

### What is directly runnable today

1. **Qwen3-8B HuggingFace smoke runs are proven locally**
   - Evidence:
     - `notes/triattention-design-code-env-check.md`
     - `experiments/smoke_env/fullkv/out/shard00/run000.jsonl`
     - `experiments/smoke_env/fullkv/out/shard00/run000.meta.json`
     - `experiments/smoke_env/fullkv/run.log`
     - `experiments/smoke_env/triattention/out/shard00/run000.jsonl`
     - `experiments/smoke_env/triattention/out/shard00/run000.meta.json`
     - `experiments/smoke_env/triattention/run.log`
     - `experiments/smoke_env/triattention/merged_smoke/merged.jsonl`
   - The TriAttention run log explicitly shows patch activation:
     - `[TriAttention] Applied compression (budget=48, divide_length=8, normalize_scores=True, per_head_pruning=True, per_layer_perhead_pruning=False)`

2. **AIME24/AIME25 dataset files exist in the current environment where `scripts/cli.py` looks first**
   - Present:
     - `/home/zijie/Code/aime24.jsonl`
     - `/home/zijie/Code/aime25.jsonl`
   - Not present in audited expected locations for MATH:
     - `/home/zijie/Code/math.jsonl` → missing
     - `data/math.jsonl` → missing
     - `math500.jsonl` in repo root → missing

3. **Qwen3-8B local model weights exist in the expected experiment model root**
   - Present:
     - `experiments/models/Qwen3-8B/config.json`
   - Missing in audited expected locations:
     - `experiments/models/DeepSeek-R1-Distill-Llama-8B/` usable config not found
     - `experiments/models/DeepSeek-R1-Distill-Qwen-7B/` usable config not found

4. **Shipped calibration files exist**
   - Present and documented in `docs/calibration.md`:
     - `triattention/calibration/for_aime24_experiment/qwen3_8b.pt`
     - `triattention/calibration/for_aime24_experiment/ds_llama8b.pt`
     - `triattention/calibration/for_aime24_experiment/ds_qwen7b.pt`
     - `triattention/calibration/for_aime25_experiment/qwen3_8b.pt`
     - `triattention/calibration/for_aime25_experiment/ds_llama8b.pt`
     - `triattention/calibration/for_aime25_experiment/ds_qwen7b.pt`

### What is needed to rerun the main paper tables

1. **For Qwen3-8B AIME runs**
   - Likely feasible after manual fixes/workarounds.
   - Minimum requirements:
     - `experiments/models/Qwen3-8B/`
     - `/home/zijie/Code/aime24.jsonl` and `/home/zijie/Code/aime25.jsonl`
     - Manual `--stats-path` override to a shipped calibration file because default stats lookup is broken
     - Working evaluation stage (currently blocked; see blockers)

2. **For DeepSeek model rows**
   - Not ready in current environment.
   - Missing local model directories are the first blocker.
   - Even after downloads, stats-path and evaluation-path issues remain.

3. **For MATH-500 rows**
   - Not ready in current environment.
   - Dataset path is missing from the locations `scripts/cli.py` checks.
   - Evaluation stage is also currently broken.

4. **For SnapKV baseline rows**
   - There is runnable method support in `scripts/cli.py` (`snapkv` is in `MODES`) and per-model wrappers such as `scripts/experiments/qwen3/run_snapkv.sh`.
   - However, the umbrella `run-default` / `run-sweep` automation in `scripts/cli.py` only covers `fullkv`, `r1kv`, and `triattention`; it does **not** include SnapKV, so the full comparison table is not reproduced by the main default automation path.

5. **For GPT-OSS-20B rows**
   - Not reproducible from the current repo surface.
   - No model support path was found in `scripts/cli.py` or in checked-in experiment wrappers/assets.

### Practical readiness summary

- **Smoke-test execution**: high readiness
- **Single-model AIME benchmark rerun (Qwen3)**: medium-low readiness with manual intervention
- **Full multi-model paper tables**: low readiness
- **Throughput / memory headline reproduction**: low readiness because measurement provenance is not exposed
- **vLLM / production path**: low readiness in current env because `vllm` is not installed per `notes/triattention-design-code-env-check.md`

## Blockers

1. **Default TriAttention stats path does not match shipped calibration assets**
   - `scripts/cli.py` computes TriAttention stats paths under:
     - `experiments/stats/<stats_dataset>/<model>/stats_budget_<budget>.pt`
   - Example generated config:
     - `experiments/configs/generated/aime24/qwen3-8b/triattention_budget_2048.yaml`
     - It points to: `/home/zijie/Code/triattention/experiments/stats/aime25/Qwen3-8B/stats_budget_2048.pt`
   - But the shipped assets are under:
     - `triattention/calibration/for_aime24_experiment/*.pt`
     - `triattention/calibration/for_aime25_experiment/*.pt`
   - Result: the default CLI path for TriAttention benchmark runs does not line up with the checked-in stats assets.

2. **Stats-generation scripts are not aligned with the CLI they call**
   - `scripts/experiments/build_all_stats.sh` runs `python scripts/cli.py build-stats` with no required `--input`.
   - Per-model scripts such as:
     - `scripts/experiments/qwen3/build_all_stats.sh`
     - `scripts/experiments/distill_qwen7b/build_all_stats.sh`
     - `scripts/experiments/distill_llama8b/build_all_stats.sh`
     pass unsupported `--dataset` flags and also omit the required `--input`.
   - `scripts/cli.py` explicitly requires:
     - `build-stats --input <plain_text_calibration_file>`

3. **Experiment wrappers reference missing extra-config files**
   - Referenced but not found in the audited tree:
     - `experiments/configs/extra_config/triattention_per_head_pruning.yaml`
     - `experiments/configs/extra_config/triattention_per_head_pruning_allow_prefill.yaml`
     - `experiments/configs/extra_config/triattention_disable_mlr.yaml`
     - `experiments/configs/extra_config/triattention_disable_trig.yaml`
     - `experiments/configs/extra_config/triattention_custom_stats.yaml`
   - Example callers:
     - `scripts/experiments/qwen3/run_triattention_per_head.sh`
     - `scripts/experiments/qwen3/budget512/run_triattention_per_head.sh`

4. **Evaluation stage is currently broken in the default pipeline**
   - `scripts/dispatch.py` points to:
     - `evaluation/eval_math_multi.py`
   - Actual file audited on disk:
     - `triattention/evaluation/eval_math_multi.py`
   - `notes/triattention-design-code-env-check.md` also reports evaluation failure because `timeout_decorator` is missing.
   - The missing dependency is not listed in:
     - `requirements.txt`
     - `setup.py` `install_requires`
     - `setup.py` `extras_require['eval']`
   - Outcome: generation may work, but the advertised merge+evaluate flow is not currently end-to-end ready.

5. **Dataset reproducibility surface is inconsistent with the README**
   - `README.md` says benchmark datasets are automatically downloaded on first run.
   - `scripts/cli.py` actually only resolves pre-existing local files from:
     - `/home/zijie/Code/<dataset>.jsonl`
     - `data/<dataset>.jsonl`
     - plus `math.jsonl` aliases for MATH
   - Current environment state from audit:
     - AIME datasets exist at `/home/zijie/Code/aime24.jsonl` and `/home/zijie/Code/aime25.jsonl`
     - MATH dataset is missing from expected locations

6. **Model support surfaces are inconsistent with result tables and wrappers**
   - `README.md` lists only three supported models, but results tables also include `GPT-OSS-20B`.
   - `scripts/cli.py` supports only:
     - `Qwen3-8B`
     - `DeepSeek-R1-Distill-Llama-8B`
     - `DeepSeek-R1-Distill-Qwen-7B`
   - `scripts/experiments/run_triattention_per_head.sh` also references unsupported:
     - `DeepSeek-R1-Distill-Qwen-14B`
   - Current environment only has audited evidence for local `Qwen3-8B` weights.

7. **Top-level reproduction docs point to stale or missing scripts/paths**
   - `docs/reproduction.md` is very thin and delegates to `scripts/experiments/` for the full story.
   - `scripts/experiments/qwen3/README.md` references paths such as `bash scripts/qwen3/...`, but the actual scripts live under `scripts/experiments/qwen3/...`.
   - The same README mentions `run_triattention.sh`, which was not found in the audited model-specific script trees.
   - `scripts/cli.py` error text also still suggests a missing helper:
     - `Run scripts/download_models_v2.sh first.`
     - `scripts/download_models_v2.sh` is not present.

8. **No checked-in benchmark outputs back the paper tables**
   - I found real run artifacts only under `experiments/smoke_env/`.
   - I did **not** find checked-in benchmark-grade merged outputs/eval outputs for the reported AIME24/AIME25/MATH500 tables.
   - I also did **not** find raw throughput or memory-measurement artifacts that would back the README headline numbers.

9. **vLLM path is not executable in the current environment**
   - `notes/triattention-design-code-env-check.md` reports `vllm` is not installed.
   - This does not block the HuggingFace research path, but it does block the repo’s advertised production/runtime path.

## Strongest evidence

1. **Best positive evidence: local HF execution is real**
   - `notes/triattention-design-code-env-check.md` records passing fullkv and triattention smoke tests.
   - `experiments/smoke_env/triattention/run.log` shows actual patch activation.
   - `experiments/smoke_env/triattention/out/shard00/run000.jsonl` and `run000.meta.json` prove concrete output creation.
   - `experiments/smoke_env/triattention/merged_smoke/merged.jsonl` proves the merge step works.

2. **Best evidence that official-looking experiment structure exists**
   - `scripts/cli.py`
   - `scripts/dispatch.py`
   - `scripts/merge_shards.py`
   - `triattention/configs/shared/defaults.yaml`
   - `triattention/configs/shared/budgets.yaml`
   - `triattention/configs/shared/runner_defaults.yaml`
   - `experiments/configs/generated/aime24/qwen3-8b/triattention_budget_2048.yaml`
   These clearly encode the intended artifact layout for runs, shard outputs, merge, and eval.

3. **Best evidence that calibration assets are genuinely shipped**
   - `docs/calibration.md`
   - `triattention/calibration/for_aime24_experiment/*.pt`
   - `triattention/calibration/for_aime25_experiment/*.pt`
   This is a real strength of the repo, even though the default CLI path does not currently consume them cleanly.

4. **Best evidence that result claims are mostly doc-level rather than artifact-level in the current repo**
   - Tables are consistent across `README.md` and `docs/results.md`.
   - Figures exist in `docs/assets/`.
   - But no corresponding checked-in raw benchmark outputs/evaluation results were found for those tables.

5. **Best evidence of unsupported table surface**
   - `GPT-OSS-20B` appears in `README.md` and `docs/results.md`.
   - It does not appear in the supported model list in `README.md` or in `scripts/cli.py` model definitions.

## Unresolved gaps

1. **Raw provenance of the paper tables is not recoverable from checked-in artifacts alone**
   - The repo surfaces the final tables and figures, but not the merged benchmark outputs or evaluation result files that would let a reviewer trace table cells back to run artifacts.

2. **Throughput and memory headline measurement pipeline is not exposed clearly enough to audit**
   - I did not find a benchmark harness or saved logs that explain how `tokens/sec` and `10.7x KV memory reduction` were computed.

3. **GPT-OSS-20B result provenance is unresolved**
   - The table entries exist, but the repo does not currently expose a matching model/config/stats/reproduction path.

4. **DFS claim provenance is incomplete**
   - DFS benchmark assets exist, but I did not find the exact result artifact used to support the `docs/results.md` narrative.
   - `triattention/benchmarks/dfs/README.md` also references commands "run from AQA-Bench root," suggesting some provenance may live outside this repo.

5. **It is unclear whether the missing benchmark outputs are intentionally excluded or simply not yet checked in**
   - If the raw outputs live externally, the current repo does not point to them.

6. **Environment docs are weaker than the execution note for actual readiness**
   - `docs/environment.md` is useful as setup guidance, but `notes/triattention-design-code-env-check.md` is currently the stronger evidence source because it documents real execution outcomes and concrete failures.
