# Feynman Project Guide

This file is the repo-level guide for Feynman when working in this repository.

It is the Feynman-side counterpart to the existing Claude Code setup in `CLAUDE.md` and `.claude/`.

## Project Rules

See `.feynman/rules/` for detailed Feynman-side project rules.

| Rule | Description |
|------|-------------|
| [.feynman/rules/documentation.md](.feynman/rules/documentation.md) | Documentation management guidelines adapted from the Claude rule |
| [.feynman/rules/storage.md](.feynman/rules/storage.md) | ModelScope download and storage management adapted from the Claude rule |
| [.feynman/rules/notes.md](.feynman/rules/notes.md) | Notes and git-management policy for durable research notes |

## Project Skills

| Skill | Description |
|-------|-------------|
| [.feynman/skills/modelscope-download/SKILL.md](.feynman/skills/modelscope-download/SKILL.md) | Download models and datasets from ModelScope using the project's preferred storage layout |

## Project Documentation References

| Document | Description |
|----------|-------------|
| [docs/environment.md](docs/environment.md) | Conda environment setup and dependencies |
| [docs/reproduction.md](docs/reproduction.md) | Experiment commands for reproducing paper results |
| [docs/calibration.md](docs/calibration.md) | Generating custom Q/K frequency statistics |
| [docs/results.md](docs/results.md) | Complete results tables and analysis |
| [docs/mlx.md](docs/mlx.md) | Apple Silicon MLX port setup and usage |
| [docs/openclaw.md](docs/openclaw.md) | OpenClaw integration guide |
| [docs/feynman-environment.md](docs/feynman-environment.md) | Feynman-oriented project environment mapped from the existing Claude Code setup |

## Project Overview

- **Project:** TriAttention
- **Goal:** efficient KV-cache compression for long-context reasoning with transformers
- **Core claim:** trigonometric frequency-domain scoring can retain important KV entries with much lower memory use than full KV retention
- **Primary paper:** TriAttention (2026) — https://arxiv.org/abs/2604.04921
- **Primary repo docs:** see the Project Documentation References table above

## Repository Map

- `triattention/methods/triattention.py` — HuggingFace-side compression logic
- `triattention/vllm/` — vLLM runtime integration and plugin code
- `triattention/mlx/` — Apple Silicon MLX port
- `triattention/evaluation/` — evaluation and grading utilities
- `scripts/cli.py` — main experiment entrypoint
- `scripts/calibrate.py` — frequency-statistics calibration
- `docs/` — detailed project documentation
- `notes/`, `outputs/`, `experiments/`, `papers/` — work artifact directories (`experiments/` is gitignored for local runs)

## Environment-First Rule

Before making strong claims about behavior, results, or reproducibility:

1. Verify the local environment can execute the project.
2. Prefer a minimal import or smoke test before deeper analysis.
3. If working from an alternate checkout or worktree, confirm Python is importing `triattention` from the current workspace before trusting any output.

Minimal smoke checks:

```bash
python -c "import triattention; print(triattention.__file__)"
```

Typical install:

```bash
pip install -e .
pip install flash-attn --no-build-isolation  # recommended
```

For MLX on Apple Silicon:

```bash
pip install mlx mlx-lm
```

## Ground Rules

### 1) Documentation management

- Detailed documentation belongs in `docs/`.
- If you add a new doc, add a reference to it in `CLAUDE.md`.
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

## Common Commands

### Run one experiment

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048
```

Available methods: `fullkv`, `r1kv`, `snapkv`, `triattention`

### Calibrate statistics

```bash
python scripts/calibrate.py \
    --model <model_path> \
    --input calibration_text.txt \
    --output triattention/calibration/<model>_stats.pt \
    --max-length 32768 \
    --device cuda
```

### Evaluate results

```bash
python triattention/evaluation/evaluate.py \
    --file_path outputs/merged/results.jsonl \
    --data_name aime24
```

## Runtime Configuration Notes

Important runtime environment variables for vLLM / deployment work:

- `TRIATTN_RUNTIME_SPARSE_STATS_PATH`
- `TRIATTN_RUNTIME_KV_BUDGET`
- `TRIATTN_RUNTIME_DIVIDE_LENGTH`
- `TRIATTN_RUNTIME_WINDOW_SIZE`
- `ENABLE_TRIATTENTION`

For chat-style OpenClaw usage, larger KV budgets than the paper default may be appropriate. See `docs/openclaw.md`.

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
- new docs referenced from `CLAUDE.md`

## Feynman Project Assets

Project-local Feynman assets live under `.feynman/`:

- rules → `.feynman/rules/`
- agents → `.feynman/agents/`
- skills → `.feynman/skills/`

## Agents and Skills Conventions

### Agents

When creating custom Feynman agents for this project:

1. **Location**: place agent definitions in `.feynman/agents/`
2. **Naming**: use kebab-case filenames
3. **Frontmatter**: include at least `name` and `description`; add `thinking`, `output`, and other fields only when needed
4. **Scope**: keep agents focused on one role such as evidence gathering, writing, verification, or review

### Skills

When creating custom Feynman skills for this project:

1. **Location**: place skill definitions in `.feynman/skills/<name>/SKILL.md`
2. **Naming**: use kebab-case directory names
3. **Frontmatter**: include at least `name` and `description`
4. **Content**: keep the skill concise and operational; point to durable docs when detailed reference material is needed

### General Guidelines

- Be specific: each rule, skill, and agent should have a focused purpose.
- Keep durable project guidance in `.feynman/` and `AGENTS.md`.
- Keep stable technical documentation in `docs/`.
- When a new durable doc is added, update both `CLAUDE.md` and the Project Documentation References section in `AGENTS.md`.

## Session Logging

Use durable session notes for meaningful work. Keep logs under `notes/session-logs/`.
