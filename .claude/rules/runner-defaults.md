# Runner Defaults Configuration Rule

## Principle

For normal project execution, the **single source of truth** for runner / dispatch defaults is:

```text
triattention/configs/shared/runner_defaults.yaml
```

If someone informally says `default_runner.yaml`, interpret that as the canonical file above. The actual filename in this repository is **`runner_defaults.yaml`**, not `default_runner.yaml`.

This file is the required control surface for:

- GPU selection defaults
- shard count defaults
- conda environment selection
- runner entrypoint
- log / output directory derivation
- common worker arguments
- dataset-specific `max_length`
- dataset-specific `num_samples`

Generated configs under `experiments/configs/generated/` are **artifacts**, not the source of truth.

## Config-First Change Rule

If the requested behavior can be expressed through the existing configuration surfaces, Claude must implement it by editing config, **not** by editing Python launch code or shell wrappers.

In particular, do **not** patch files such as:

- `scripts/cli.py`
- `scripts/dispatch.py`
- `scripts/worker.py`
- `scripts/experiments/**/*.sh`

just to change durable execution behavior that already belongs in config.

Config changes should go through the appropriate YAML source of truth first (for example `triattention/configs/shared/runner_defaults.yaml` or the relevant method/config file), then flow through the normal generation path.

Only change code when the required behavior is genuinely impossible to express with the current config surface or when the config plumbing itself is broken. In that case, Claude must explicitly note why a config-only change was insufficient.

## Hard Rule

For durable project workflows, **do not** treat shell flags, ad-hoc environment variables, or hand-edited generated YAMLs as the primary configuration mechanism.

Instead:

1. Set the intended defaults in `triattention/configs/shared/runner_defaults.yaml`
2. Let `scripts/cli.py` load those defaults
3. Let `scripts/cli.py` generate the per-run config under `experiments/configs/generated/`
4. Let `scripts/dispatch.py` consume that generated config

## What Must Be Configured Here

The following settings must be expressed in `runner_defaults.yaml` for standard project operation:

### 1. Dispatch / Experiment Defaults

Under:

```yaml
experiment:
```

This section controls dispatch-level behavior such as:

- `conda_env`
- `runner_path`
- `merged_dir_name`
- `gpus`
- `auto_gpu_fallback`
- `num_shards`
- `gpu_memory_threshold`
- `env`

Examples:

```yaml
experiment:
  conda_env: triattention
  runner_path: scripts/worker.py
  gpus:
    - 0
  num_shards: 8
```

### 2. Shared Worker Defaults

Under:

```yaml
runner_args:
```

This section controls common `scripts/worker.py` arguments such as:

- `eval_batch_size`
- `seed`
- `attn_implementation`
- `load_dtype`
- `fp32_topk`
- `num_samples`
- `num_samples_by_dataset`
- `temperature`
- `top_p`
- `use_chat_template`
- `chat_system_prompt`

These become the base `runner_args` embedded into generated configs.

### 3. Dataset-Specific Length Defaults

Under:

```yaml
dataset_max_length:
```

This section controls the `max_length` that `scripts/cli.py` injects into generated configs by dataset name.

Example:

```yaml
dataset_max_length:
  aime24: 32768
  aime25: 32768
  math500: 8192
```

## How the Config Actually Works

The configuration flow is:

```text
shell helper script
  -> scripts/cli.py
    -> load triattention/configs/shared/runner_defaults.yaml
    -> build generated per-run YAML
    -> call scripts/dispatch.py --config <generated_yaml>
      -> load generated YAML
      -> launch scripts/worker.py with runner_args from that YAML
```

### Step 1: `scripts/cli.py` loads `runner_defaults.yaml`

Code path:

- `scripts/cli.py`
- `RUNNER_DEFAULTS_PATH = CONFIG_ROOT / "runner_defaults.yaml"`
- `load_runner_defaults()`

This means `runner_defaults.yaml` is not optional metadata. It is the file used to seed every generated run config.

### Step 2: `scripts/cli.py` builds a generated config

Code path:

- `config_output_path(...)`
- `build_config(...)`
- `write_config(...)`

Generated files land under:

```text
experiments/configs/generated/<dataset>/<model-slug>/<mode>_<tag>.yaml
```

For example:

```text
experiments/configs/generated/aime24/qwen3-8b/fullkv_full.yaml
```

### Step 3: `runner_defaults.yaml` values are copied into the generated config

Inside `build_config(...)`, the file contributes:

- `experiment` defaults
- `runner_args` defaults
- dataset-specific `max_length`
- dataset-specific `num_samples`

That generated config is then the concrete dispatch plan for a specific dataset/model/method combination.

### Step 4: `scripts/dispatch.py` consumes the generated config

`scripts/dispatch.py` reads:

```bash
python scripts/dispatch.py --config <generated_yaml> --dataset <dataset>
```

From that generated config, it uses:

- `experiment.gpus`
- `experiment.num_shards`
- `experiment.log_dir`
- `experiment.method_output_dir`
- `experiment.conda_env`
- `experiment.runner_path`
- `experiment.runner_args.*`

### Step 5: `dispatch.py` launches `worker.py`

`dispatch.py` converts `experiment.runner_args` into CLI flags for `scripts/worker.py`.

So a setting placed in `runner_defaults.yaml` flows through:

```text
runner_defaults.yaml
  -> generated YAML
  -> dispatch.py
  -> worker.py argument
```

## GPU Configuration Rule

GPU defaults must be controlled in:

```yaml
experiment:
  gpus: ...
```

### Allowed durable patterns

#### Fixed single GPU

```yaml
experiment:
  gpus:
    - 0
```

#### Fixed multi-GPU set

```yaml
experiment:
  gpus:
    - 0
    - 1
```

#### Automatic GPU selection

```yaml
experiment:
  gpus: auto
  auto_gpu_fallback:
    - 0
    - 1
```

### Rule

For standard project runs:

- Prefer setting GPU defaults in `runner_defaults.yaml`
- Do **not** rely on `CUDA_VISIBLE_DEVICES` alone as the durable configuration method
- Do **not** rely on `dispatch.py --gpus ...` as the permanent project default

Command-line `--gpus` is acceptable only for one-off debugging. It must not replace the canonical default in `runner_defaults.yaml`.

## Shard Configuration Rule

Shard defaults must be controlled in:

```yaml
experiment:
  num_shards: <N>
```

### Rule

- `num_shards` in `runner_defaults.yaml` is the normal source of truth
- Do **not** hardcode shard counts inside helper shell scripts
- Do **not** hand-edit generated configs to change shard count
- Do **not** rely on `dispatch.py --num-shards ...` as the durable project configuration

Use CLI override only for temporary debugging. If the intended default changes, update `runner_defaults.yaml`.

## Conda Environment Rule

The runtime environment must be configured in:

```yaml
experiment:
  conda_env: triattention
```

### Rule

- The default conda environment for experiments belongs in `runner_defaults.yaml`
- Do **not** duplicate the environment name across multiple shell scripts unless there is a documented, exceptional reason
- If the project changes its main environment name, update `runner_defaults.yaml` first

## Dataset Length Rule

Dataset-specific context lengths must be configured in:

```yaml
dataset_max_length:
```

### Rule

- Use `dataset_max_length` for durable per-dataset defaults
- Do **not** hardcode benchmark-specific `max_length` values in shell wrappers
- For standard runs, do **not** patch generated YAMLs by hand to change `max_length`

If a dataset needs a new stable default length, add or modify it here.

## Sampling / Draw Count Rule

Per-dataset draw counts must be configured in:

```yaml
runner_args:
  num_samples:
  num_samples_by_dataset:
```

### Rule

- Standard benchmark draw counts must live in `runner_defaults.yaml`
- Do not spread these defaults across helper scripts

## Generated Config Rule

Files under:

```text
experiments/configs/generated/
```

are generated execution artifacts.

### Rule

- Do not treat generated configs as the long-term place to store project defaults
- Do not commit hand-edited generated YAMLs as the main configuration mechanism
- Regenerate them from `runner_defaults.yaml` via `scripts/cli.py`

Hand-editing a generated file is acceptable only for narrow, disposable local debugging. If the change matters for future runs, move it back into `runner_defaults.yaml`.

## Helper Script Rule

Shell helpers such as:

- `scripts/experiments/qwen3/run_fullkv.sh`
- `scripts/experiments/qwen3/run_rkv.sh`
- `scripts/experiments/qwen3/run_snapkv.sh`
- `scripts/experiments/qwen3/run_triattention_per_head.sh`

must not become the canonical place for core runtime defaults.

### Rule

Helpers may choose:

- which dataset/model/method combinations to run
- batching behavior like `JOB_PARALLEL`

Helpers must not become the primary place to define:

- GPU defaults
- shard defaults
- conda environment defaults
- dataset `max_length` defaults
- common worker args

Those belong in `runner_defaults.yaml`.

## How to Add New Durable Config

When a new execution setting must become a standard project default, follow this sequence.

### Case A: Dispatch-level setting

If it affects dispatch behavior, add it under:

```yaml
experiment:
```

Examples:

- scheduler behavior
- default log routing
- GPU selection behavior
- shard count

Then ensure `scripts/dispatch.py` already reads it or add support there.

### Case B: Worker-level setting

If it is fundamentally a `scripts/worker.py` argument, add it under:

```yaml
runner_args:
```

Then ensure:

1. `scripts/worker.py` accepts the flag
2. `scripts/dispatch.py` passes it through from `runner_args`
3. `scripts/cli.py` includes it when building generated configs

### Case C: Dataset-specific default

If the setting varies by dataset and is a durable default, add a dedicated mapping near:

```yaml
dataset_max_length:
```

or introduce a new dataset-keyed mapping if the behavior truly depends on dataset identity.

### Required process for adding config

1. Add the config to `runner_defaults.yaml`
2. Verify the code path that consumes it
3. Generate a config with `scripts/cli.py --dry-run run-one ...`
4. Inspect the generated YAML
5. Confirm `dispatch.py` / `worker.py` receive the intended value
6. Update project guidance if the new field changes operator workflow

## What Not To Do

### Do not do these as the normal project workflow

- Do not hand-edit `experiments/configs/generated/*.yaml` and call that the project default
- Do not keep changing `dispatch.py --gpus ...` by hand for every run when a stable GPU default exists
- Do not change `dispatch.py --num-shards ...` for standard runs instead of setting `experiment.num_shards`
- Do not rely on `CUDA_VISIBLE_DEVICES` alone as the canonical project configuration
- Do not scatter defaults across multiple helper scripts

## Canonical Operator Workflow

For standard work:

1. Edit `triattention/configs/shared/runner_defaults.yaml`
2. Run a dry-run generation:

```bash
python scripts/cli.py --dry-run run-one --dataset aime24 --model Qwen3-8B --method fullkv
```

3. Inspect the generated config under:

```text
experiments/configs/generated/
```

4. Run the desired helper script or `scripts/cli.py run-one ...`

This ensures the generated config remains a faithful projection of the canonical defaults.

## Maintenance Rule

If a future change affects experiment launching behavior, first ask:

> Should this become a stable default in `runner_defaults.yaml`?

If the answer is yes, put it there first. Only then wire the rest of the stack.
