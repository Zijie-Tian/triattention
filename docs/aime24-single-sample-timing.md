# AIME24 Single-Sample Timing (Qwen3-8B, TriAttention)

## Summary

On **April 14, 2026 (UTC)**, we measured end-to-end wall-clock time for **one AIME24 sample** (`sample_idx=0`) on **physical GPU 1** (`CUDA_VISIBLE_DEVICES=1`) using **Qwen3-8B + TriAttention**.

Test configuration:

- Dataset: `aime24`
- Sample count: `1`
- Draws: `1`
- Model: `Qwen3-8B`
- Method: `triattention`
- KV budget: `1024`
- Dtype: `bfloat16`
- Attention backend: `flash_attention_2`
- Divide length: `128`
- Window size: `128`
- Prompt mode: plain prompt (`--use_chat_template false`)
- Stats file: `experiments/stats/aime25/Qwen3-8B/stats_budget_2048.pt`

## Results

All three runs were launched in **fresh processes**, so the times below are **end-to-end wall-clock times including model load**.

| max_length | elapsed_seconds | prefill_tokens | output_tokens | total_tokens | derived throughput (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 67.62 | 185 | 1863 | 2048 | 30.29 |
| 4096 | 134.30 | 185 | 3911 | 4096 | 30.50 |
| 8192 | 258.52 | 185 | 8007 | 8192 | 31.69 |

Notes:

- `prefill_tokens` stayed constant at `185` for this sample.
- Each run filled the configured `max_length` exactly.
- Throughput above is **derived** from `total_tokens / elapsed_seconds`.

## Command Template

The timings were collected by running `scripts/worker.py` directly on GPU 1, once per `max_length`:

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/data/tzj/Code/triattention \
  conda run --no-capture-output -n triattention python scripts/worker.py \
  --dataset_path /home/tzj/data/datasets/aime24/test.jsonl \
  --output_dir experiments/outputs/ad_hoc/<RUN_TAG>/shards \
  --model_path /mnt/data/tzj/models/Qwen3-8B \
  --max_length <MAX_LEN> \
  --eval_batch_size 1 \
  --load_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --method triattention \
  --kv_budget 1024 \
  --window_size 128 \
  --divide_length 128 \
  --num_samples 1 \
  --temperature 0.6 \
  --top_p 0.95 \
  --triattention_stats_file /mnt/data/tzj/Code/triattention/experiments/stats/aime25/Qwen3-8B/stats_budget_2048.pt \
  --count_prompt_tokens true \
  --attention_layer_compression true \
  --slack_budget_trigger true \
  --triattention_normalize_scores true \
  --triattention_frequency_window 65536 \
  --triattention_score_aggregation mean \
  --per_head_pruning true \
  --per_layer_perhead_pruning false \
  --allow_prefill_compression false \
  --disable_mlr false \
  --disable_trig false \
  --pruning_seed 0 \
  --use_chat_template false \
  --max_examples 1 \
  --num_shards 1 \
  --shard_id 0
```

## Evidence and Artifacts

Artifacts from the measured runs:

- Logs: `experiments/logs/ad_hoc/timing_aime24_qwen3_8b_triattention_b1024_gpu1_20260414_155654/`
- Outputs: `experiments/outputs/ad_hoc/timing_aime24_qwen3_8b_triattention_b1024_gpu1_20260414_155654/`

The recorded sample metadata for all three runs was:

- `sample_idx=0`
- `prefill_tokens=185`

## Stats File Reuse Note

The local workspace only had `stats_budget_2048.pt` for `Qwen3-8B`. Before running the `budget=1024` timing tests, we inspected the serialized metadata and confirmed it contained model/rope/head-shape information but **no budget-specific field**, so the same stats file was reused for this timing-only check.
