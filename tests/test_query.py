def test_format_sources_empty_nodes(mocker):
    from query import format_sources
    response = mocker.MagicMock()
    response.source_nodes = []
    assert format_sources(response) == ""


def test_format_sources_single_node(mocker):
    from query import format_sources
    node = mocker.MagicMock()
    node.metadata = {"file_name": "report.pdf", "page_label": "3"}
    response = mocker.MagicMock()
    response.source_nodes = [node]
    result = format_sources(response)
    assert "report.pdf" in result
    assert "p.3" in result


def test_format_sources_deduplicates_by_filename(mocker):
    from query import format_sources
    node1 = mocker.MagicMock()
    node1.metadata = {"file_name": "doc.pdf", "page_label": "3"}
    node2 = mocker.MagicMock()
    node2.metadata = {"file_name": "doc.pdf", "page_label": "7"}
    response = mocker.MagicMock()
    response.source_nodes = [node1, node2]
    result = format_sources(response)
    assert result.count("doc.pdf") == 1


def test_format_sources_missing_metadata(mocker):
    from query import format_sources
    node = mocker.MagicMock()
    node.metadata = {}
    response = mocker.MagicMock()
    response.source_nodes = [node]
    result = format_sources(response)
    assert "unknown" in result


import pytest


def test_load_index_exits_when_collection_missing(monkeypatch, mocker):
    import config
    from query import load_index
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "STORAGE_DIR", "/tmp/no-storage")
    monkeypatch.setattr(config, "COLLECTION_NAME", "pdf_index")

    mock_client = mocker.MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")
    mocker.patch("chromadb.PersistentClient", return_value=mock_client)

    with pytest.raises(SystemExit):
        load_index()


def test_load_index_returns_vector_store_index(monkeypatch, mocker):
    import config
    from query import load_index
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_collection = mocker.MagicMock()
    mock_client = mocker.MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mocker.patch("chromadb.PersistentClient", return_value=mock_client)

    mock_index = mocker.MagicMock()
    mocker.patch("query.VectorStoreIndex")
    mocker.patch("query.ChromaVectorStore")
    mocker.patch("query.OpenAIEmbedding")
    mocker.patch("query.VectorStoreIndex.from_vector_store", return_value=mock_index)

    result = load_index()
    assert result is mock_index
