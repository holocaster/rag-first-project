# RAG Agent — Project Guide

## What This Is

A CLI RAG agent that indexes PDFs into a local ChromaDB vector database and answers questions about them using OpenAI GPT-4o-mini.

## Stack

| Layer | Tool |
|---|---|
| Framework | LlamaIndex |
| Vector DB | ChromaDB (local, persisted to `storage/`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Package manager | `uv` — never use `pip` |

## Project Layout

```
config.py       — all tuneable constants (chunk size, model names, paths)
ingest.py       — scan data/, embed PDFs, store in ChromaDB
query.py        — load index, run interactive REPL
data/           — drop PDFs here
storage/        — ChromaDB index (auto-created, git-ignored)
tests/          — unit tests (pytest + pytest-mock)
```

## Setup

```bash
cp .env.example .env   # add OPENAI_API_KEY
# or: export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Index PDFs in data/
uv run python ingest.py

# Ask questions
uv run python query.py
```

Re-running `ingest.py` skips already-indexed files (fingerprint = sha256 of filename + mtime).

## Tests

```bash
uv run pytest tests/ -v
```

32 tests. All unit tests — external dependencies (OpenAI, ChromaDB) are mocked.

## Tuning

Edit `config.py` to change chunking, retrieval, or model settings. All constants live there — no magic numbers elsewhere.

## Adding Dependencies

```bash
uv add <package>          # runtime
uv add --dev <package>    # dev only
```
