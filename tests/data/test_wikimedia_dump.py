import bz2

import pipeline
from convert_wikimedia_dump import stream_pages


def test_bz2_xml_stream_parser_filters_namespace_and_preserves_revision(tmp_path):
    xml = b'''<?xml version="1.0" encoding="UTF-8"?>
    <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
      <page><title>Main Page</title><ns>0</ns><id>1</id><revision><id>11</id><parentid>10</parentid><timestamp>2026-10-01T00:00:00Z</timestamp><contributor><username>Example</username><id>7</id></contributor><comment>Update</comment><model>wikitext</model><format>text/x-wiki</format><sha1>abc</sha1><text bytes="20">A synthetic test article with enough text.</text></revision></page>
      <page><title>Talk:Main Page</title><ns>1</ns><id>2</id><revision><id>12</id><timestamp>2026-10-01T00:00:00Z</timestamp><text>Talk content.</text></revision></page>
    </mediawiki>'''
    path = tmp_path / "tiny.xml.bz2"
    path.write_bytes(bz2.compress(xml))
    rows = list(stream_pages(path, {"0"}))
    assert len(rows) == 1
    row = rows[0]
    assert row["page_id"] == "1"
    assert row["title"] == "Main Page"
    assert row["revision_id"] == "11"
    assert row["parent_revision_id"] == "10"
    assert row["content_model"] == "wikitext"
    assert row["source_url"].endswith("oldid=11")


def test_wikimedia_snapshot_is_a_strict_dated_revision():
    source = {"revision_kind": "dated_dump_snapshot", "revision": "20261001"}
    assert pipeline.parse_revision(source) == "20261001"
    try:
        pipeline.parse_revision({"revision_kind": "dated_dump_snapshot", "revision": "latest"})
    except pipeline.PipelineError as exc:
        assert "immutable 40-character" in str(exc)
    else:
        raise AssertionError("floating dump names must not be accepted")
