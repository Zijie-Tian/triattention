# Environment Check: TriAttention execution status

Date: 2026-04-13
Repo: `/home/zijie/Code/triattention`
Conda env: `triattention`

## Goal
Check whether the current fork environment can actually execute the official TriAttention code, not just import it.

## Minimal oracle/tests
1. Verify Python / CUDA / key packages are available.
2. Verify the project imports successfully.
3. Run a real HuggingFace-backend smoke test with `scripts/worker.py` using local `Qwen3-8B` weights on 1 short sample in `fullkv` mode.
4. Run a real HuggingFace-backend smoke test with `scripts/worker.py` using local `Qwen3-8B` weights on 1 short sample in `triattention` mode with a provided stats file.
5. Check whether higher-level utilities (`merge_shards.py`, evaluation entrypoint, vLLM runtime import) are runnable.

## Environment observations
- Python: `3.10.20`
- Active interpreter: `/home/zijie/anaconda3/envs/triattention/bin/python`
- GPU visible: A100-SXM4-80GB on `CUDA_VISIBLE_DEVICES=0` was idle and available.
- `torch`: `2.9.1+cu128`
- `transformers`: `4.56.0`
- `datasets`: `4.8.4`
- `flash_attn`: `2.8.3`
- `triattention`: import OK
- `vllm`: not installed

## Real smoke tests

### Test A: HF worker, fullkv mode
Command:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/worker.py \
  --dataset_path experiments/smoke_env/aime25.jsonl \
  --output_dir experiments/smoke_env/fullkv/out \
  --model_path experiments/models/Qwen3-8B \
  --method fullkv \
  --shard_id 0 --num_shards 1 --num_samples 1 --max_examples 1 \
  --max_length 96 --attn_implementation sdpa --load_dtype bfloat16
```
Result: **PASS**
Evidence:
- Output file: `experiments/smoke_env/fullkv/out/shard00/run000.jsonl`
- Metadata file: `experiments/smoke_env/fullkv/out/shard00/run000.meta.json`
- Log: `experiments/smoke_env/fullkv/run.log`

### Test B: HF worker, triattention mode
Command:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/worker.py \
  --dataset_path experiments/smoke_env/aime25.jsonl \
  --output_dir experiments/smoke_env/triattention/out \
  --model_path experiments/models/Qwen3-8B \
  --method triattention \
  --triattention_stats_file triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
  --kv_budget 48 --window_size 8 --divide_length 8 \
  --count_prompt_tokens true --allow_prefill_compression true \
  --shard_id 0 --num_shards 1 --num_samples 1 --max_examples 1 \
  --max_length 96 --attn_implementation sdpa --load_dtype bfloat16
```
Result: **PASS**
Evidence:
- Output file: `experiments/smoke_env/triattention/out/shard00/run000.jsonl`
- Metadata file: `experiments/smoke_env/triattention/out/shard00/run000.meta.json`
- Log line confirms patch activation:
  - `[TriAttention] Applied compression (budget=48, divide_length=8, normalize_scores=True, per_head_pruning=True, per_layer_perhead_pruning=False)`
- Log: `experiments/smoke_env/triattention/run.log`

### Test C: Merge utility
Command:
```bash
python scripts/merge_shards.py --method-output-dir experiments/smoke_env/triattention/out --merged-dir-name merged_smoke
```
Result: **PASS**
Evidence:
- `experiments/smoke_env/triattention/merged_smoke/merged.jsonl`

## Blockers / partial failures

### 1. vLLM runtime is not runnable in current env
- `import vllm` fails with `ModuleNotFoundError`.
- `triattention.vllm.runtime.integration_monkeypatch` import therefore fails.
- Implication: the HuggingFace research path works, but the repo's production vLLM server path is **not currently executable** without installing `vllm`.

### 2. Evaluation pipeline currently fails on a missing dependency
Command:
```bash
python triattention/evaluation/eval_math_multi.py --help
```
Failure:
- `ModuleNotFoundError: No module named 'timeout_decorator'`

Implication:
- Generation works, but the evaluation entrypoint is **not currently runnable** until `timeout_decorator` is installed.
- This dependency does not appear in `requirements.txt`.

### 3. `scripts/dispatch.py` appears to reference the wrong evaluation path
Observed constants:
- Dispatch expects: `evaluation/eval_math_multi.py`
- Actual file exists at: `triattention/evaluation/eval_math_multi.py`

Implication:
- High-level sharded runs that reach the evaluation stage are likely to fail even after worker generation succeeds, unless this path is fixed or evaluation is skipped.

## Bottom line
- **Yes, the core HuggingFace TriAttention code executes successfully in the current environment.**
- **No, the full repo stack is not completely ready end-to-end yet** because:
  1. `vllm` is missing,
  2. evaluation is missing `timeout_decorator`, and
  3. `scripts/dispatch.py` appears to point to a non-existent eval script path.

## Next recommended actions
1. Install `timeout_decorator` and rerun the evaluation entrypoint.
2. Decide whether to install `vllm` now, depending on whether runtime/deployment paths need to be studied empirically.
3. Inspect / patch `scripts/dispatch.py` eval path before claiming the full CLI workflow is runnable end-to-end.
