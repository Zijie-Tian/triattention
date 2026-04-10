# Storage and Data Management Rule

## Principle

Due to limited disk space on `/mnt/data` and the root partition, all large downloads (models, datasets) should be stored in `~/data` directory. Use **ModelScope** as the primary download source.

## Storage Locations

| Type | Location |
|------|----------|
| Models | `~/models/` |
| Datasets | `~/data/datasets/` |

## Rules

### 1. Model Storage

- Download models to `~/models/` instead of project directory
- Model checkouts should be at `~/models/<model-name>/`
- Code should load models from `~/models/<model-name>/`

### 2. Dataset Storage

- Store datasets in `~/data/datasets/`
- Update dataset path configurations accordingly

### 3. Download Source: ModelScope

Use ModelScope SDK for downloading both models and datasets:

```python
from pathlib import Path
from modelscope import snapshot_download, dataset_snapshot_download

MODEL_DIR = Path.home() / "models"
DATASET_DIR = Path.home() / "data/datasets"

def download_model(model_name: str, repo_id: str):
    """Download model from ModelScope only if it doesn't exist."""
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
    """Download dataset from ModelScope only if it doesn't exist."""
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

### 4. Authentication

ModelScope uses tokens stored in `~/.modelscope/credentials/`. Ensure the user is logged in:
```python
from modelscope import HubApi
api = HubApi()
api.login('<your_token>')
```

Or use environment variable `MODELSCOPE_TOKEN`.

### 5. Checklist Before Downloading

1. Check if target directory exists: `model_path.exists()`
2. Check if directory is non-empty: `any(model_path.iterdir())`
3. Only download if both conditions are False
4. Print skip message if already exists

### 6. No Redundant Downloads

- Never re-download if model/dataset already exists
- Always verify before downloading
- Use the downloaded path directly in model loading

## How to Apply

When downloading models or datasets, always:
1. Use ModelScope SDK (`modelscope` package)
2. Download to `~/models/<model-name>/` for models
3. Download to `~/data/datasets/<dataset-name>/` for datasets
4. Check if already exists before downloading
5. Load models from the downloaded path
