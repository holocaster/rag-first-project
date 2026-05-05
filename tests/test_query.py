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
