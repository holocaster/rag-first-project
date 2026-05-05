import hashlib
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from embeddings import get_embedding_model

load_dotenv()


def require_api_key() -> str:
    if config.EMBED_PROVIDER == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            print("Error: OPENAI_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)
        return key
    elif config.EMBED_PROVIDER == "voyage":
        key = os.getenv("VOYAGE_API_KEY")
        if not key:
            print("Error: VOYAGE_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)
        return key
    raise ValueError(f"Unknown EMBED_PROVIDER: {config.EMBED_PROVIDER!r}")


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
    require_api_key()

    pdf_files = scan_pdfs(config.DATA_DIR)
    if not pdf_files:
        print(f"No PDF files found in {config.DATA_DIR}/")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF file(s)")

    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    indexed = get_indexed_fingerprints(collection)

    to_process = [p for p in pdf_files if fingerprint(p) not in indexed]
    skipped = len(pdf_files) - len(to_process)

    if not to_process:
        print(f"All {skipped} file(s) already indexed. Nothing to do.")
        return

    print(f"Indexing {len(to_process)} new file(s), skipping {skipped}...")

    embed_model = get_embedding_model()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    total_chunks = 0
    for pdf in to_process:
        try:
            docs = SimpleDirectoryReader(input_files=[str(pdf)]).load_data()
            nodes = splitter.get_nodes_from_documents(docs)
            fp = fingerprint(pdf)
            for node in nodes:
                node.metadata["fingerprint"] = fp
            VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
            total_chunks += len(nodes)
            print(f"  {pdf.name}: {len(nodes)} chunks")
        except Exception as e:
            print(f"  Warning: could not process {pdf.name}: {e}")

    print(f"\nDone. {len(to_process)} file(s) indexed, {total_chunks} total chunks stored.")


if __name__ == "__main__":
    ingest()
