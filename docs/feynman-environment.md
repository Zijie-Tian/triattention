# Feynman Environment for This Repository

This repository already ships a Claude Code-oriented project setup in `CLAUDE.md` and `.claude/`.

This document explains the parallel **Feynman-oriented** project environment that was added by mapping the existing Claude Code setup onto Feynman's current conventions.

## What the research showed

Two different layers matter:

1. **Feynman runtime configuration** is primarily documented as a user-level setup under `~/.feynman/` (installation, package stack, settings).
2. **Feynman project/repo conventions** are better illustrated by Feynman's own upstream repository and its `project init` scaffold, which create or use `AGENTS.md`, `notes/session-logs/`, artifact directories, and repo-local skills.

Because of that split, this repository's Feynman setup uses two layers:

### Layer A — strongly evidenced project pieces

- `AGENTS.md`
- `.feynman/agents/` (evidenced by the upstream Feynman repo)
- `notes/session-logs/`
- standard artifact directories such as `papers/`

### Layer B — repo management convention added here

- `.feynman/rules/`
- `.feynman/skills/`

These two directories were added to mirror the existing Claude Code structure and keep Feynman-facing project assets together in one place. Current public Feynman docs reviewed for this migration do **not** clearly establish them as guaranteed stock-runtime auto-load paths for arbitrary projects, so they should be treated as a **repo convention**, not a documented upstream requirement.

I still intentionally did **not** add a project-local copy of Feynman's bundled app files such as `.feynman/SYSTEM.md`, because the current public evidence shows that file as part of the **Feynman application bundle**, not the normal per-project scaffold.

## Mapping: Claude Code → Feynman

| Claude Code file / pattern | Purpose in this repo | Feynman-side mapping used here | Notes |
|---|---|---|---|
| `CLAUDE.md` | durable project instructions | `AGENTS.md` | Feynman's upstream `project init` scaffolds `AGENTS.md` as the durable project guide |
| `.claude/rules/documentation.md` | documentation policy | `.feynman/rules/documentation.md` + `AGENTS.md` | kept as project ground rules |
| `.claude/rules/storage.md` | storage and download policy | `.feynman/rules/storage.md` + skill guidance | preserved ModelScope + `~/models` / `~/data/datasets` policy |
| `.claude/skills/modelscope-download.md` | reusable download workflow | `.feynman/skills/modelscope-download/SKILL.md` | this repo now keeps the skill only under `.feynman/skills/` to avoid duplicate maintenance |
| `notes/` | durable notes | `notes/session-logs/README.md` | aligned with Feynman's session-log convention |
| artifact directories | outputs from longer work | `papers/.gitkeep` and existing `notes/`, `outputs/`, `experiments/` | aligned with Feynman's default artifact layout; `experiments/` is gitignored in this repo for local runs |
| `.mcp.json` | Claude Code MCP configuration | unchanged | current Feynman docs reviewed here emphasize runtime settings and packages, not a separate repo-level MCP migration path |

## Files added

### 1. `AGENTS.md`

This is the Feynman-side repo guide. It includes:

- TriAttention project overview
- repo map and key paths
- environment-first / smoke-test rule
- documentation management rule
- model/dataset storage rule
- artifact discipline
- verification gates
- common commands for experiments, calibration, and evaluation

### 2. `.feynman/rules/*`

This repo now has a Feynman-side rules area:

- `.feynman/rules/documentation.md`
- `.feynman/rules/storage.md`
- `.feynman/rules/notes.md`

These are a repo-management convention added to mirror the existing Claude rules layout.

### 3. `.feynman/agents/*`

This repo now has project-tuned Feynman subagent prompts:

- `.feynman/agents/researcher.md`
- `.feynman/agents/writer.md`
- `.feynman/agents/verifier.md`
- `.feynman/agents/reviewer.md`

This part is aligned with the upstream Feynman repository, which stores bundled subagent prompts under `.feynman/agents/`.

### 4. `.feynman/skills/modelscope-download/SKILL.md`

This is the management-copy version of the existing Claude Code ModelScope download skill.

It keeps the same essential behavior:

- models in `~/models/`
- datasets in `~/data/datasets/`
- avoid duplicate downloads
- prefer ModelScope for manual agent-managed downloads

### 5. `notes/session-logs/README.md`

This establishes the session-log location that Feynman's own scaffold expects for durable session summaries.

### 6. `papers/.gitkeep`

This adds the paper-draft artifact directory used by Feynman's default research output conventions.

## Usage

### Run Feynman from the repo root

```bash
feynman
```

Or run a one-shot prompt:

```bash
feynman "summarize the repo structure and identify the core TriAttention implementation"
```

### Use the repo guide

When Feynman is working in this repository, `AGENTS.md` should act as the durable repo guide for:

- what the project is
- where important code lives
- which commands matter
- which verification rules apply
- where durable artifacts should be written

### Use the project skill

For manual model or dataset setup tasks, use the project-managed skill definition:

- `.feynman/skills/modelscope-download/SKILL.md`

This repo does **not** keep a second mirrored root-level skill copy, because that created duplicate maintenance.

### Keep durable session notes

For substantial work, write session summaries under:

- `notes/session-logs/`

## Why this design was chosen

This setup is conservative on purpose.

The public Feynman docs we reviewed are explicit about:

- installation
- package stack
- user-level `~/.feynman/settings.json`

But they are less explicit about a full project-local config tree equivalent to Claude Code's `.claude/` directory.

The strongest current evidence for project-level Feynman structure comes from:

- the upstream `AGENTS.md`
- the upstream `.feynman/agents/` layout
- the upstream `project init` scaffold
- the upstream root `skills/` layout (used as background evidence, not copied literally here)

So this repository now uses a mixed approach:

- adopt the **project-level conventions that are directly evidenced**
- add a small `.feynman/rules/` and `.feynman/skills/` management layer as an explicit repo convention, with the caveat documented rather than hidden

## Sources

1. Claude Code settings — https://code.claude.com/docs/en/settings
2. Explore the `.claude` directory — https://code.claude.com/docs/en/claude-directory
3. Feynman installation docs — https://www.feynman.is/docs/getting-started/installation
4. Feynman package stack docs — https://www.feynman.is/docs/reference/package-stack
5. Feynman upstream `AGENTS.md` — https://github.com/getcompanion-ai/feynman/blob/main/AGENTS.md
6. Feynman upstream repo — https://github.com/getcompanion-ai/feynman
