import config


def test_chunk_size():
    assert config.CHUNK_SIZE == 512

def test_chunk_overlap():
    assert config.CHUNK_OVERLAP == 64

def test_top_k():
    assert config.TOP_K == 5

def test_embed_model():
    assert config.EMBED_MODEL == "text-embedding-3-small"

def test_llm_model():
    assert config.LLM_MODEL == "gpt-4o-mini"

def test_data_dir():
    assert config.DATA_DIR == "data"

def test_storage_dir():
    assert config.STORAGE_DIR == "storage"

def test_collection_name():
    assert config.COLLECTION_NAME == "pdf_index"

def test_embed_provider():
    assert config.EMBED_PROVIDER == "openai"

def test_voyage_embed_model():
    assert config.VOYAGE_EMBED_MODEL == "voyage-3"
