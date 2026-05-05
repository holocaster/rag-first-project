import hashlib
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

load_dotenv()


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("Error: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    return key


def fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}{stat.st_mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()


def scan_pdfs(data_dir: str) -> list[Path]:
    return list(Path(data_dir).rglob("*.pdf"))


def get_indexed_fingerprints(collection) -> set[str]:
    metadatas = collection.get(include=["metadatas"]).get("metadatas")
    if not metadatas:
        return set()
    return {m["fingerprint"] for m in metadatas if m and "fingerprint" in m}


def ingest() -> None:
    pass  # implemented in Task 6


if __name__ == "__main__":
    ingest()
