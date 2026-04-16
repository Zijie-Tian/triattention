# AGENTS.md

This file provides guidance to AI agents (like Feynman) when working with code in this repository.

## Project Rules

See [.feynman/rules/](.feynman/rules/) for detailed project rules. Following our document management principles, detailed guidelines are kept in their respective files, while this document serves as the index and brief summary.

| Rule | Description |
|------|-------------|
| [.feynman/rules/documentation.md](.feynman/rules/documentation.md) | Documentation management guidelines (Detailed docs in `docs/` or rule files, summaries in index) |
| [.feynman/rules/storage.md](.feynman/rules/storage.md) | ModelScope download and storage management (Models in `~/models/`, Datasets in `~/data/datasets/`) |
| [.feynman/rules/gpu.md](.feynman/rules/gpu.md) | GPU configuration for running experiments (Use `--gpus` explicitly) |
| [.feynman/rules/runner-defaults.md](.feynman/rules/runner-defaults.md) | Canonical execution-configuration rule: set GPU, shards, conda env, lengths, and runner defaults via `triattention/configs/shared/runner_defaults.yaml` |
| [.feynman/rules/agents_sync.md](.feynman/rules/agents_sync.md) | Maintain synchronization between AGENTS.md and CLAUDE.md |

## Project Skills

| Skill | Description |
|-------|-------------|
| [.feynman/skills/modelscope-download.md](.feynman/skills/modelscope-download.md) | Download models and datasets from ModelScope |

## Project Documentation References

| Document | Description |
|----------|-------------|
| [docs/environment-setup.md](docs/environment-setup.md) | 新机器环境配置详细指南 (硬件要求、依赖安装、验证) |
| [docs/reproduction.md](docs/reproduction.md) | Experiment commands for reproducing paper results |
| [docs/calibration.md](docs/calibration.md) | Generating custom Q/K frequency statistics |
| [docs/results.md](docs/results.md) | Complete results tables and analysis |
| [docs/mlx.md](docs/mlx.md) | Apple Silicon MLX port setup and usage |
| [docs/openclaw.md](docs/openclaw.md) | OpenClaw integration guide |
| [docs/prefill_sparsity.md](docs/prefill_sparsity.md) | Extension of TriAttention to Prefill phase (Query-Pooling + Exact Key paradigm) |
| [docs/kvcache-rope-dataset.md](docs/kvcache-rope-dataset.md) | KV cache and RoPE dataset analysis |
| [docs/aime24-single-sample-timing.md](docs/aime24-single-sample-timing.md) | Single-sample AIME24 timing measurements for Qwen3-8B TriAttention on GPU1 |

## Project Overview

TriAttention is an efficient KV cache compression method for long-context reasoning with transformers. It uses trigonometric frequency-domain scoring to select important tokens, achieving 10.7x memory reduction with no accuracy loss.

Key papers: [TriAttention](https://arxiv.org/abs/2604.04921)

## Installation

```bash
pip install -e .
pip install flash-attn --no-build-isolation  # recommended
```

For MLX (Apple Silicon): `pip install mlx mlx-lm`

## Common Commands

### Running Experiments

```bash
# Run a single experiment
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048

# Available methods: fullkv, r1kv, snapkv, triattention
# Available datasets: aime24, aime25, math500
# Available models: Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B
```

### Calibrating Statistics

```bash
python scripts/calibrate.py \
    --model <model_path> \
    --input calibration_text.txt \
    --output triattention/calibration/<model>_stats.pt \
    --max-length 32768 \
    --device cuda
```

### Evaluating Results

```bash
python triattention/evaluation/evaluate.py \
    --file_path outputs/merged/results.jsonl \
    --data_name aime24
```

## Architecture

### Core Components

1. **`triattention/methods/triattention.py`** - HuggingFace attention-layer compression
   - `TriAttention` class: Main compressor using frequency-based scoring
   - `apply_triattention_patch()`: Patches model.forward for transparent compression
   - Supports per-head and per-layer-per-head independent pruning modes

2. **`triattention/vllm/`** - vLLM production integration
   - `core/compressor.py`: `TriAttentionCompressor` for vLLM runtime
   - `core/scoring.py`: Frequency-domain scoring implementation
   - `runtime/`: vLLM monkeypatching and integration hooks
   - Plugin registration: `triattention.vllm.plugin:register_triattention_backend`

3. **`triattention/mlx/`** - Apple Silicon MLX port (contrib by @DeadByDawn101)
   - `triattention_mlx.py`: Core MLX implementation
   - `calibrate_mlx.py`: Statistics calibration for MLX models

4. **`triattention/methods/baselines/`** - Baseline implementations
   - `snapkv.py`: SnapKV compression
   - `r1_kv.py`: R-KV compression

### Configuration

- **`triattention/configs/shared/`**: Default budgets, runner configurations
- **`triattention/vllm/stats/`**: Pre-calibrated frequency statistics for supported models
- Environment variables for vLLM runtime:
  - `TRIATTN_RUNTIME_KV_BUDGET`: Max tokens in KV cache (default: 2048)
  - `TRIATTN_RUNTIME_DIVIDE_LENGTH`: Compression interval (default: 128)
  - `TRIATTN_RUNTIME_WINDOW_SIZE`: Always-kept recent tokens (default: 128)
  - `TRIATTN_RUNTIME_SPARSE_STATS_PATH`: Path to .pt stats file

### Key Data Structures

- **Frequency statistics** (`.pt` files): Per-layer, per-head frequency scaling factors
- **KV cache format**: Tuple of (key, value) tensors per layer for HuggingFace; PagedAttention blocks for vLLM

## Environment Variables

For vLLM server mode:
```bash
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/vllm/stats/qwen3_32b_int4_stats.pt
export TRIATTN_RUNTIME_KV_BUDGET=2048
```

For OpenClaw chat mode:
```bash
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/vllm/stats/qwen3_32b_int4_stats.pt
export TRIATTN_RUNTIME_KV_BUDGET=12000  # larger for multi-turn chat
vllm serve <model> --enable-prefix-caching false --max-num-batched-tokens 1024
```

## vLLM Plugin

The package registers as a vLLM plugin via `entry_points` in `setup.py`. After installation, vLLM automatically discovers and activates it. Use `ENABLE_TRIATTENTION=false` to disable.

## Code Style

- Python 3.10+
- Type hints with `torch.Tensor` for PyTorch types
- Use `torch.float32` explicitly for compute dtype; `torch.bfloat16/float16` for model dtype
- Grouped imports with standard library first, then third-party, then local

## Ground Rules

### 1) Documentation management

- Detailed documentation belongs in `docs/`.
- If you add a new doc, add a reference to both `CLAUDE.md` and `AGENTS.md`.
- Use descriptive kebab-case doc names.

### 2) Model and dataset storage

For manual download/setup tasks, prefer the existing project storage rule:

- models → `~/models/`
- datasets → `~/data/datasets/`
- prefer ModelScope for manual agent-managed downloads
- do not re-download if the target directory already exists and is non-empty

Do **not** silently rewrite existing project code paths that currently fetch from other sources unless the task explicitly asks for that change.

### 3) Artifact discipline

- scratch notes → `notes/`
- local experiment scripts/logs/results → `experiments/` (gitignored)
- reviews, summaries, result memos → `outputs/`
- paper-style writeups → `papers/`

Use durable files for substantial work instead of leaving important context only in the chat.

If an experiment helper should be version-controlled, put it under `scripts/` or the package source tree instead of `experiments/`.

### 4) Evidence discipline

- Do not say a result is reproduced, verified, or confirmed unless you actually ran the check and can point to the command output, log, or artifact.
- Separate direct observations from inferences.
- Keep exact commands, config paths, and result file paths for quantitative claims.

### 5) Data safety

- Do not modify raw datasets or external dataset symlinks casually.
- Prefer writing derived artifacts to `outputs/`, `notes/`, or `experiments/`.

## Task Ledger Conventions

For substantial work, maintain a small task ledger in the active plan or note file.

- Track status as `todo`, `in_progress`, `done`, `blocked`, or `superseded`.
- Do not silently skip failed checks.
- Record where outputs were written.

## Verification Gates

Before delivering a substantial result, check the ones that apply:

- environment/import smoke test ran
- command lines recorded
- config paths recorded
- result/log paths recorded
- unsupported claims downgraded or removed
- new docs referenced from `CLAUDE.md` and `AGENTS.md`

## Agents and Skills Conventions

### Agents

When creating custom agents for this project:

1. **Location**: Place agent definitions in `.feynman/agents/`
2. **Naming**: Use kebab-case: `my-agent.md`
3. **Frontmatter**:
```markdown
---
name: agent-name
description: Brief one-line description
---
```

4. **Agent Structure**:
```markdown
---
name: my-agent
description: Analyzes TriAttention performance
---

# My Agent

## Purpose
...

## When to Use
...

## Guidelines
...
```

### Skills

When creating custom skills for this project:

1. **Location**: Place skill definitions in `.feynman/skills/`
2. **Naming**: Use kebab-case: `my-skill.md`
3. **Frontmatter**:
```markdown
---
name: my-skill
description: Brief description of what the skill does
trigger: /my-skill
---
```

4. **Skill Structure**:
```markdown
---
name: my-skill
description: Perform X task
trigger: /my-skill
---

# My Skill

## Purpose
...

## Usage
`/my-skill [args]`

## Steps
1. Step one
2. Step two
```

### General Guidelines

- **Be specific**: Each agent/skill should have a clear, focused purpose
- **Document triggers**: Always specify how to invoke the agent/skill
- **Include examples**: Show concrete usage examples
- **Keep focused**: Avoid combining unrelated functionality

## Session Logging

Use durable session notes for meaningful work. Keep logs under `notes/session-logs/`.
