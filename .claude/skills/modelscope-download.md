---
name: modelscope-download
description: Download models and datasets from ModelScope
trigger: /modelscope
---

# ModelScope Download Skill

Download models and datasets from ModelScope using Python SDK.

## Prerequisites

1. Install ModelScope SDK:
   ```bash
   pip install modelscope
   ```

2. Login to ModelScope (one-time setup):
   ```python
   from modelscope import HubApi
   api = HubApi()
   api.login('<your_token>')
   ```
   Get your token from: https://modelscope.cn/my/settings/token

## Usage

### Download a Model

```bash
/modelscope download-model <repo_id> [local_name]
```

Example:
```bash
/modelscope download-model Qwen/Qwen2.5-7B-Instruct qwen2.5-7b
```

This downloads to `~/models/qwen2.5-7b/`

### Download a Dataset

```bash
/modelscope download-dataset <repo_id> [local_name]
```

Example:
```bash
/modelscope download-dataset AI-ModelScope/GSM8K gsm8k
```

This downloads to `~/data/datasets/gsm8k/`

## Python API

```python
from pathlib import Path
from modelscope import snapshot_download, dataset_snapshot_download

MODEL_DIR = Path.home() / "models"
DATASET_DIR = Path.home() / "data/datasets"

# Download model
model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=str(MODEL_DIR / "qwen2.5-7b"),
)

# Download dataset
dataset_path = dataset_snapshot_download(
    dataset_id="AI-ModelScope/GSM8K",
    local_dir=str(DATASET_DIR / "gsm8k"),
)
```

## Common Options

| Parameter | Description |
|-----------|-------------|
| `repo_id` | ModelScope repo ID (e.g., `Qwen/Qwen2.5-7B-Instruct`) |
| `revision` | Git revision (branch, tag, or commit) |
| `local_dir` | Custom download directory |
| `ignore_file_pattern` | Files to skip (e.g., `["*.bin", "*.onnx"]`) |
| `allow_patterns` | Only download matching files (e.g., `["*.safetensors"]`) |

## Storage Locations

Per project storage rules:
- **Models**: `~/models/`
- **Datasets**: `~/data/datasets/`

## Finding Repo IDs

1. Search on https://modelscope.cn/models
2. Search on https://modelscope.cn/datasets
3. Use the MCP server: `/mcp` then `search_models` or search tools
