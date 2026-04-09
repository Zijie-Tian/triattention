# Development Environment Setup

## Conda Environment: `triattention`

This document describes the development environment setup for TriAttention project.

## Environment Creation

```bash
# Create conda environment
mamba create -n triattention python=3.10 -y

# Or using conda
conda create -n triattention python=3.10 -y
```

## Python Version

- **Python**: 3.10

## Core Dependencies

| Package | Version | Source | Notes |
|---------|---------|--------|-------|
| torch | 2.4.0+cu121 | PyTorch (CUDA 12.1) | From `download.pytorch.org/whl/cu121` |
| transformers | 5.5.1 | pip | |
| datasets | 4.8.4 | pip | |
| accelerate | 1.13.0 | pip | |
| einops | 0.8.2 | pip | |
| sentencepiece | 0.2.1 | pip | |
| pyyaml | 6.0.3 | conda | |
| tqdm | 4.67.3 | pip | |
| matplotlib | 3.10.8 | pip | |
| pebble | 5.2.0 | pip | |
| sympy | 1.14.0 | conda | |
| scipy | 1.15.3 | pip | |
| latex2sympy2 | 1.9.1 | pip | |
| word2number | 1.1 | pip | |
| triton | 3.0.0 | pip | Comes with PyTorch 2.4 |
| antlr4-python3-runtime | 4.7.2 | pip | |
| cuda-toolkit | 12.1.0 | conda | |

## Installation Commands

```bash
# Activate environment
conda activate triattention
# or
mamba activate triattention

# Install PyTorch with CUDA 12.1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install core packages
pip install \
  transformers \
  datasets \
  huggingface-hub \
  accelerate \
  einops \
  sentencepiece \
  pyyaml \
  tqdm \
  matplotlib \
  regex \
  pebble \
  sympy \
  triton

# Install evaluation packages
pip install latex2sympy2 word2number antlr4-python3-runtime==4.7.2 scipy

# Install project in editable mode
pip install -e .
```

## Hardware

- **GPU**: NVIDIA A100-PCIE-40GB (x2)
- **CUDA**: 12.9 (driver: 580.95.05)
- **Compiler**: nvcc 12.9

## Optional Components

### Flash Attention

Flash Attention provides significant performance improvements but requires CUDA compilation:

```bash
pip install flash-attn --no-build-isolation
```

### vLLM (Production Server)

For vLLM-based inference server:

```bash
pip install vllm
```

### MLX (Apple Silicon)

For Apple Silicon Macs:

```bash
pip install mlx mlx-lm
```

## Verification

```bash
# Verify environment
conda run -n triattention python -c "
import torch
import transformers
import datasets
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'transformers: {transformers.__version__}')
"
```

## Quick Commands

```bash
# Run a single experiment
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048
```
