# Repo-local Feynman Assets

This directory is the repository-managed home for Feynman-facing configuration assets in this project.

## What is strongly evidenced vs repo convention

### Strongly evidenced by current public Feynman sources

- `AGENTS.md` as the repo-level project guide
- `.feynman/agents/*.md` as the source location for bundled Feynman subagent prompts in the upstream Feynman repository
- root-level `skills/<name>/SKILL.md` in the upstream Feynman repository

### Repo convention added in this repository

- `.feynman/rules/`
- `.feynman/skills/`

These two folders are added here to mirror the existing Claude Code setup and keep Feynman-oriented project assets together in one place.

Current public Feynman docs reviewed for this migration do **not** clearly establish `.feynman/rules/` or `.feynman/skills/` as guaranteed stock-runtime auto-load paths for arbitrary projects. Treat them here as a **repo-management convention**.

## Practical layout in this repo

- `AGENTS.md` — top-level repo guide for Feynman
- `.feynman/rules/` — project policies derived from the existing Claude rules
- `.feynman/agents/` — project-tuned researcher/reviewer/writer/verifier prompts
- `.feynman/skills/` — project-managed skill sources

## Maintenance rule

This repository keeps Feynman-managed skills only under `.feynman/skills/` to avoid duplicate maintenance.

If a future runtime integration requires a root `skills/` tree, generate or mirror it from `.feynman/skills/` deliberately instead of maintaining two independent copies.
