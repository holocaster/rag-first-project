import config


def _get_openai_embedding(model: str):
    from llama_index.embeddings.openai import OpenAIEmbedding
    return OpenAIEmbedding(model=model)


def _get_voyage_embedding(model: str):
    from llama_index.embeddings.voyageai import VoyageEmbedding
    return VoyageEmbedding(model_name=model)


def get_embedding_model():
    if config.EMBED_PROVIDER == "openai":
        return _get_openai_embedding(config.EMBED_MODEL)
    elif config.EMBED_PROVIDER == "voyage":
        return _get_voyage_embedding(config.VOYAGE_EMBED_MODEL)
    raise ValueError(f"Unknown EMBED_PROVIDER: {config.EMBED_PROVIDER!r}")
