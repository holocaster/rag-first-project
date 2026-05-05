# RAG PDF Agent

CLI tool that indexes PDFs into a local vector database and answers questions about them.

## Stack

- **LlamaIndex** — RAG framework
- **ChromaDB** — local vector store (persisted to `storage/`)
- **OpenAI** — `text-embedding-3-small` embeddings, `gpt-4o-mini` LLM

## Setup

```bash
cp .env.example .env   # add your OPENAI_API_KEY
```

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

## Usage

```bash
# 1. Drop PDFs into data/
# 2. Index them
uv run python ingest.py

# 3. Ask questions
uv run python query.py
```

Re-running `ingest.py` skips already-indexed files.

## Tests

```bash
uv run pytest tests/ -v
```

## Configuration

Edit `config.py` to change chunk size, top-k retrieval, or model names.
