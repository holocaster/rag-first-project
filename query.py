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
    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    try:
        collection = chroma_client.get_collection(config.COLLECTION_NAME)
    except Exception:
        print("Index not found. Run ingest.py first.")
        sys.exit(1)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = OpenAIEmbedding(model=config.EMBED_MODEL)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def build_query_engine(index):
    llm = OpenAI(model=config.LLM_MODEL)
    retriever = index.as_retriever(similarity_top_k=config.TOP_K)
    return RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)


def run_repl(query_engine) -> None:
    print("Ask questions about your PDFs. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break
        response = query_engine.query(question)
        print(response)
        sources = format_sources(response)
        if sources:
            print(sources)
        print()


def main() -> None:
    require_openai_key()
    index = load_index()
    engine = build_query_engine(index)
    run_repl(engine)


if __name__ == "__main__":
    main()
