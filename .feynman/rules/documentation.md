# Documentation Management Rule

## Principle

This project uses **strict documentation management**. Detailed documentation belongs in `docs/`, with references and brief summaries kept in both `CLAUDE.md` and `AGENTS.md`.

## Rules

### 1. Where to Write Documentation

- **Detailed documentation** → `docs/` as markdown files
- **Summaries and references** → the Project Documentation References sections in `CLAUDE.md` and `AGENTS.md`

### 2. When to Create Documentation

Create a new `docs/*.md` file when:
- documenting features, workflows, or configurations
- explaining architecture or design decisions
- providing usage examples or tutorials

### 3. How to Reference

After creating a new doc, add a reference entry to both `CLAUDE.md` and `AGENTS.md`:

```markdown
| Document | Description |
|----------|-------------|
| [docs/new-feature.md](docs/new-feature.md) | Brief description |
```

### 4. Doc File Naming

- use kebab-case: `feature-name.md`
- be descriptive: `calibration.md`, `openclaw-integration.md`

### 5. Promotion Rule

Promote a note into `docs/` when it becomes any of the following:

- a repeatable setup guide
- an architecture explanation
- a design decision record worth preserving
- a stable reproduction or troubleshooting guide

## Examples

### Before (inline operational knowledge only in notes/comments)
```python
# This function does X.
# It requires Y and returns Z.
```

### After (durable project doc in `docs/feature.md`)

```markdown
## Feature X

This feature does X by:
1. Step one
2. Step two

### Usage
...
```

Then reference it from `CLAUDE.md`, `AGENTS.md`, or code with a link.
