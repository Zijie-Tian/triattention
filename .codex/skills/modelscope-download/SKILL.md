---
name: modelscope-download
description: Download models and datasets from ModelScope using the project's preferred storage layout.
---

# ModelScope Download Skill

Download models and datasets from ModelScope using the Python SDK.

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

## Repository storage rule

Per project storage rules:
- **Models**: `~/models/`
- **Datasets**: `~/data/datasets/`

## Usage Pattern

### Download a model

```python
from pathlib import Path
from modelscope import snapshot_download

MODEL_DIR = Path.home() / "models"
target = MODEL_DIR / "qwen3-8b"

if not (target.exists() and any(target.iterdir())):
    snapshot_download(
        repo_id="Qwen/Qwen3-8B",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
```

### Download a dataset

```python
from pathlib import Path
from modelscope import dataset_snapshot_download

DATASET_DIR = Path.home() / "data/datasets"
target = DATASET_DIR / "gsm8k"

if not (target.exists() and any(target.iterdir())):
    dataset_snapshot_download(
        dataset_id="AI-ModelScope/GSM8K",
        local_dir=str(target),
    )
```

## Full Python API Example

```python
from pathlib import Path
from modelscope import snapshot_download, dataset_snapshot_download

MODEL_DIR = Path.home() / "models"
DATASET_DIR = Path.home() / "data/datasets"

def download_model(model_name: str, repo_id: str):
    target_dir = MODEL_DIR / model_name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[skip] Model already exists: {target_dir}")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model_name} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return target_dir

def download_dataset(dataset_name: str, repo_id: str):
    target_dir = DATASET_DIR / dataset_name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[skip] Dataset already exists: {target_dir}")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {dataset_name} -> {target_dir}")
    dataset_snapshot_download(
        dataset_id=repo_id,
        local_dir=str(target_dir),
    )
    return target_dir
```

## Common Options

| Parameter | Description |
|-----------|-------------|
| `repo_id` | ModelScope repo ID |
| `dataset_id` | ModelScope dataset ID |
| `revision` | Git revision (branch, tag, or commit) |
| `local_dir` | Custom download directory |
| `ignore_file_pattern` | Files to skip |
| `allow_patterns` | Only download matching files |

## Checklist Before Downloading

1. Check if the target directory exists
2. Check if it is non-empty
3. Skip re-download when assets already exist
4. Keep downloaded assets outside the repository

## Notes for this repository

- The repository's own evaluation scripts may still fetch some resources from their built-in upstream sources. Do not silently rewrite that behavior unless the task explicitly asks for it.
- This skill is for **manual setup and controlled downloads** when an agent is managing local assets.
