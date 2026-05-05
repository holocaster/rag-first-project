import os
import sys

import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

load_dotenv()


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("Error: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    return key


def format_sources(response) -> str:
    seen: dict[str, str] = {}
    for node in response.source_nodes:
        fname = node.metadata.get("file_name", "unknown")
        page = node.metadata.get("page_label", "?")
        seen[fname] = page
    if not seen:
        return ""
    parts = [f"{fname} (p.{page})" for fname, page in seen.items()]
    return "Sources: " + ", ".join(parts)


def load_index():
    pass  # implemented in Task 8


def build_query_engine(index):
    pass  # implemented in Task 8


def run_repl(query_engine) -> None:
    pass  # implemented in Task 9


def main() -> None:
    pass  # implemented in Task 9


if __name__ == "__main__":
    main()
