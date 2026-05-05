# Agent Instructions

See [CLAUDE.md](CLAUDE.md) for full project context, stack, layout, and conventions.

## Key Rules

- Package manager: `uv` — never use `pip`
- All tuneable constants live in `config.py` — no magic numbers elsewhere
- Run tests with `uv run pytest tests/ -v`
- External dependencies (OpenAI, ChromaDB) must be mocked in tests

## Workflow

1. `ingest.py` — indexes PDFs from `data/` into ChromaDB (`storage/`)
2. `query.py` — interactive REPL that queries the index

Re-running `ingest.py` is safe; already-indexed files are skipped via sha256 fingerprint.
