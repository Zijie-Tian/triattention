# Codex Image Generation Output Rule

## Principle

Codex-generated image artifacts should have one predictable home in this repository.

## Rule

When using Codex `imagegen` to create or edit raster images for this project, save the resulting image files under:

```text
.codex/generated_images/
```

Create the directory if it does not already exist.

## Guidelines

- Do not scatter generated image outputs in the repository root, `docs/`, `outputs/`, or experiment folders unless the user explicitly asks for that location.
- Use descriptive filenames that include the purpose and, when helpful, a timestamp or short variant label.
- Treat `.codex/generated_images/` as the staging area for generated bitmap outputs. If a generated image becomes a durable project asset, copy or move it to the appropriate project asset path in a separate, intentional step.
