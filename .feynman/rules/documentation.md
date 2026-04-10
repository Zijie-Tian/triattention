# Documentation Management Rule

## Principle

This project uses **strict documentation management**. Detailed documentation belongs in `docs/`, with references and brief summaries in `AGENTS.md`.

## Rules

### 1. Where to Write Documentation

- **Detailed documentation** → `docs/` as markdown files
- **Summaries and references** → `AGENTS.md` Project Documentation References section

### 2. When to Create Documentation

Create a new `docs/*.md` file when:
- Documenting features, workflows, or configurations
- Explaining architecture or design decisions
- Providing usage examples or tutorials

### 3. How to Reference

After creating a new doc, add a reference entry to `AGENTS.md`:
```markdown
| Document | Description |
|----------|-------------|
| [docs/new_feature.md](docs/new_feature.md) | Brief description |
```

### 4. Doc File Naming

- Use kebab-case: `feature-name.md`
- Be descriptive: `calibration.md`, `openclaw-integration.md`

## Examples

### Before (Inline in code comments)
```python
# This function does X.
# It requires Y and returns Z.
```

### After (In `docs/feature.md`)
```markdown
## Feature X

This feature does X by:
1. Step one
2. Step two

### Usage
...
```

Then reference from `AGENTS.md` or code with a link.
