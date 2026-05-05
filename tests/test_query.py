import pytest


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
    mocker.patch("query.get_embedding_model")
    mocker.patch("query.VectorStoreIndex.from_vector_store", return_value=mock_index)

    result = load_index()
    assert result is mock_index


def test_run_repl_exits_on_quit(mocker):
    from query import run_repl
    engine = mocker.MagicMock()
    mocker.patch("builtins.input", side_effect=["quit"])
    run_repl(engine)
    engine.query.assert_not_called()


def test_run_repl_exits_on_eof(mocker):
    from query import run_repl
    engine = mocker.MagicMock()
    mocker.patch("builtins.input", side_effect=EOFError)
    run_repl(engine)
    engine.query.assert_not_called()


def test_run_repl_queries_engine_and_prints_answer(mocker, capsys):
    from query import run_repl
    engine = mocker.MagicMock()
    response = mocker.MagicMock()
    response.__str__ = lambda self: "The answer is 42."
    response.source_nodes = []
    engine.query.return_value = response
    mocker.patch("builtins.input", side_effect=["what is the answer?", "quit"])
    run_repl(engine)
    engine.query.assert_called_once_with("what is the answer?")
    out = capsys.readouterr().out
    assert "The answer is 42." in out


def test_run_repl_skips_empty_input(mocker):
    from query import run_repl
    engine = mocker.MagicMock()
    mocker.patch("builtins.input", side_effect=["", "  ", "exit"])
    run_repl(engine)
    engine.query.assert_not_called()
