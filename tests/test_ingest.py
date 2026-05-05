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
