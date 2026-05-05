import pytest
from pathlib import Path


def test_fingerprint_returns_64_char_hex(tmp_path):
    from ingest import fingerprint
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = fingerprint(pdf)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_fingerprint_same_file_stable(tmp_path):
    from ingest import fingerprint
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert fingerprint(pdf) == fingerprint(pdf)


def test_fingerprint_different_names_differ(tmp_path):
    from ingest import fingerprint
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"%PDF-1.4")
    b.write_bytes(b"%PDF-1.4")
    assert fingerprint(a) != fingerprint(b)


def test_scan_pdfs_finds_pdfs(tmp_path):
    from ingest import scan_pdfs
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.pdf").write_bytes(b"")
    result = scan_pdfs(str(tmp_path))
    assert len(result) == 2


def test_scan_pdfs_ignores_non_pdfs(tmp_path):
    from ingest import scan_pdfs
    (tmp_path / "doc.txt").write_text("hello")
    (tmp_path / "report.pdf").write_bytes(b"")
    result = scan_pdfs(str(tmp_path))
    assert len(result) == 1
    assert result[0].name == "report.pdf"


def test_scan_pdfs_empty_dir(tmp_path):
    from ingest import scan_pdfs
    assert scan_pdfs(str(tmp_path)) == []


def test_scan_pdfs_recursive(tmp_path):
    from ingest import scan_pdfs
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.pdf").write_bytes(b"")
    result = scan_pdfs(str(tmp_path))
    assert len(result) == 1


def test_get_indexed_fingerprints_returns_set(mocker):
    from ingest import get_indexed_fingerprints
    collection = mocker.MagicMock()
    collection.get.return_value = {
        "metadatas": [
            {"fingerprint": "abc123"},
            {"fingerprint": "def456"},
        ]
    }
    result = get_indexed_fingerprints(collection)
    assert result == {"abc123", "def456"}


def test_get_indexed_fingerprints_empty_collection(mocker):
    from ingest import get_indexed_fingerprints
    collection = mocker.MagicMock()
    collection.get.return_value = {"metadatas": []}
    result = get_indexed_fingerprints(collection)
    assert result == set()


def test_get_indexed_fingerprints_skips_missing_key(mocker):
    from ingest import get_indexed_fingerprints
    collection = mocker.MagicMock()
    collection.get.return_value = {
        "metadatas": [
            {"fingerprint": "abc123"},
            {"other_key": "xyz"},
        ]
    }
    result = get_indexed_fingerprints(collection)
    assert result == {"abc123"}


def test_get_indexed_fingerprints_skips_none_entries(mocker):
    from ingest import get_indexed_fingerprints
    collection = mocker.MagicMock()
    collection.get.return_value = {
        "metadatas": [
            {"fingerprint": "abc123"},
            None,
            {"fingerprint": "def456"},
        ]
    }
    result = get_indexed_fingerprints(collection)
    assert result == {"abc123", "def456"}


def test_ingest_exits_when_no_pdfs(tmp_path, monkeypatch):
    import config
    from ingest import ingest
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        ingest()


def test_ingest_skips_all_already_indexed(tmp_path, monkeypatch, mocker, capsys):
    import config
    from ingest import ingest, fingerprint
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "STORAGE_DIR", str(tmp_path / "storage"))

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fp = fingerprint(pdf)

    mock_col = mocker.MagicMock()
    mock_col.get.return_value = {"metadatas": [{"fingerprint": fp}]}
    mock_client = mocker.MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col
    mocker.patch("chromadb.PersistentClient", return_value=mock_client)

    ingest()

    out = capsys.readouterr().out
    assert "nothing to do" in out.lower() or "already indexed" in out.lower()


def test_ingest_processes_new_pdfs(tmp_path, monkeypatch, mocker, capsys):
    import config
    from ingest import ingest
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "STORAGE_DIR", str(tmp_path / "storage"))

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    mock_col = mocker.MagicMock()
    mock_col.get.return_value = {"metadatas": []}
    mock_client = mocker.MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col
    mocker.patch("chromadb.PersistentClient", return_value=mock_client)

    mock_docs = [mocker.MagicMock()]
    mock_reader = mocker.MagicMock()
    mock_reader.load_data.return_value = mock_docs
    mocker.patch("ingest.SimpleDirectoryReader", return_value=mock_reader)

    mock_node = mocker.MagicMock()
    mock_node.metadata = {}
    mock_splitter = mocker.MagicMock()
    mock_splitter.get_nodes_from_documents.return_value = [mock_node]
    mocker.patch("ingest.SentenceSplitter", return_value=mock_splitter)

    mocker.patch("ingest.get_embedding_model")
    mocker.patch("ingest.ChromaVectorStore")
    mocker.patch("ingest.StorageContext")
    mocker.patch("ingest.VectorStoreIndex")

    ingest()

    out = capsys.readouterr().out
    assert "doc.pdf" in out


def test_ingest_exits_when_voyage_key_missing(tmp_path, monkeypatch):
    import config
    from ingest import ingest
    monkeypatch.setattr(config, "EMBED_PROVIDER", "voyage")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(SystemExit):
        ingest()
