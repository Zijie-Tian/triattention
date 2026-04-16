# GPU Configuration Rule

## Problem

The `scripts/dispatch.py` uses automatic GPU detection by default, which may not respect `CUDA_VISIBLE_DEVICES` environment variable.

## Solution

To restrict experiments to specific GPUs, use the `--gpus` argument with `dispatch.py`:

```bash
# Run on GPU 1 only
python scripts/dispatch.py --config <config_path> --dataset <dataset> --gpus 1

# Run on multiple specific GPUs
python scripts/dispatch.py --config <config_path> --dataset <dataset> --gpus 0,1
```

## Configuration Options

### Option 1: CLI argument (recommended for one-off runs)
```bash
python scripts/dispatch.py --config <config.yaml> --gpus 1
```

### Option 2: Modify config file
Edit the generated config YAML to set `gpus` explicitly:
```yaml
experiment:
  gpus: "1"  # or ["0", "1"] for multiple GPUs
  num_shards: 1  # adjust based on number of GPUs
```

### Option 3: Set memory threshold
```bash
python scripts/dispatch.py --config <config.yaml> --gpu-memory-threshold <MiB>
```

## How dispatch.py GPU Selection Works

1. If `--gpus` argument is provided, use it directly
2. Otherwise, check `experiment.gpus` in config
3. If `gpus: auto`, use `auto_detect_gpus(threshold)` to find GPUs with memory usage below threshold
4. Fallback to `auto_gpu_fallback` list

## Examples

### Run single experiment on GPU 1
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/dispatch.py \
    --config experiments/configs/generated/aime24/qwen3-8b/triattention_budget_2048.yaml \
    --dataset aime24 \
    --gpus 1
```

### Run with memory threshold
```bash
python scripts/dispatch.py --config <config.yaml> --gpu-memory-threshold 30000
```

## Note

- `CUDA_VISIBLE_DEVICES` alone is **not sufficient** because `dispatch.py` auto-detects all available GPUs
- Always use `--gpus` to explicitly restrict GPU selection
