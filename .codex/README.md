# Repo-local Codex Assets

This directory is the repository-managed home for Codex-facing configuration assets in this project.

## What is strongly evidenced vs repo convention

### Strongly evidenced by current public Codex sources

- `AGENTS.md` as the repo-level project guide
- `.codex/agents/*.md` as the source location for bundled Codex subagent prompts in the upstream Codex repository
- root-level `skills/<name>/SKILL.md` in the upstream Codex repository

### Repo convention added in this repository

- `.codex/rules/`
- `.codex/skills/`

These two folders are added here to mirror the existing Claude Code setup and keep Codex-oriented project assets together in one place.

Current public Codex docs reviewed for this migration do **not** clearly establish `.codex/rules/` or `.codex/skills/` as guaranteed stock-runtime auto-load paths for arbitrary projects. Treat them here as a **repo-management convention**.

## Practical layout in this repo

- `AGENTS.md` — top-level repo guide for Codex
- `.codex/rules/` — project policies derived from the existing Claude rules
- `.codex/agents/` — project-tuned researcher/reviewer/writer/verifier prompts
- `.codex/skills/` — project-managed skill sources

## Maintenance rule

This repository keeps Codex-managed skills only under `.codex/skills/` to avoid duplicate maintenance.

If a future runtime integration requires a root `skills/` tree, generate or mirror it from `.codex/skills/` deliberately instead of maintaining two independent copies.
