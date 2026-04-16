# Synchronize AGENTS.md and CLAUDE.md

Whenever `AGENTS.md` is updated (e.g., adding a new document reference, changing project rules, or updating skills), the equivalent path-aware update must be applied to `CLAUDE.md` when the shared project guidance also changes.

## Guidelines

1. **Keep content in sync:** The structure and content of `AGENTS.md` and `CLAUDE.md` should mirror each other exactly.
2. **Path Mapping:** 
   - `AGENTS.md` uses the `.codex/` directory for rules, agents, and skills.
   - `CLAUDE.md` uses the `.claude/` directory for rules, agents, and skills.
   - When copying content from `AGENTS.md` to `CLAUDE.md`, ensure all `.codex/` paths are updated to `.claude/`.
3. **Agent References:**
   - In `AGENTS.md`, AI assistants are referred to as "AI agents (like Codex)".
   - In `CLAUDE.md`, AI assistants are referred to as "Claude Code (claude.ai/code)".
4. **Consistency:** Do not add a project document reference to one without adding it to the other.