import pytest


def test_get_embedding_model_returns_openai(monkeypatch, mocker):
    import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "openai")
    mock_fn = mocker.patch("embeddings._get_openai_embedding", return_value="openai_model")

    from embeddings import get_embedding_model
    result = get_embedding_model()

    mock_fn.assert_called_once_with(config.EMBED_MODEL)
    assert result == "openai_model"


def test_get_embedding_model_returns_voyage(monkeypatch, mocker):
    import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "voyage")
    mock_fn = mocker.patch("embeddings._get_voyage_embedding", return_value="voyage_model")

    from embeddings import get_embedding_model
    result = get_embedding_model()

    mock_fn.assert_called_once_with(config.VOYAGE_EMBED_MODEL)
    assert result == "voyage_model"


def test_get_embedding_model_raises_on_unknown_provider(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "foobar")

    from embeddings import get_embedding_model
    with pytest.raises(ValueError, match="foobar"):
        get_embedding_model()
