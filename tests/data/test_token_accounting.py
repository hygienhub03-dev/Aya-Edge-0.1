import pipeline


def test_development_byte_tokenizer_is_explicit_and_unicode_is_byte_counted():
    counter = pipeline.TokenCounter("aya-edge-dev-byte-v1")
    assert counter.kind == "utf8_byte_ids"
    assert counter.count("é") == 2
    assert counter.count("世界") == 6
    assert not pipeline.TokenCounter("aya-edge-dev-byte-v1").tokenizer_id.startswith("production")


def test_unknown_or_implicit_tokenizer_fails():
    try:
        pipeline.TokenCounter("")
    except pipeline.PipelineError as exc:
        assert "explicitly selected" in str(exc)
    else:
        raise AssertionError("empty tokenizer ID must be rejected")
