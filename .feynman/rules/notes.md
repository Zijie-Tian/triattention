# Notes and Git Rule

## Principle

Treat `notes/` as a research notebook area, but only commit notes that have durable value for the project.

## What to commit

- stable research notes
- design analyses worth preserving
- environment checks that future work depends on
- README files explaining note subdirectories

## What to avoid committing by default

- noisy per-session scratch logs
- disposable intermediate brainstorming
- repeated temporary exports that can be regenerated

## Directory guidance

- `notes/session-logs/` is for chronological session summaries
- `notes/*.md` can hold durable topic notes
- when a note becomes stable project documentation, promote it into `docs/`

## Git advice

A good default is:

- commit durable `notes/*.md`
- keep `notes/session-logs/README.md`
- ignore future high-frequency session-log entries unless you explicitly want them versioned
