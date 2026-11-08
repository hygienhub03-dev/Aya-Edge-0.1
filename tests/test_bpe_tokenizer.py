import json

import pytest

from aya_edge.tokenizer.bpe import (
    PRODUCTION_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    ProductionTokenizerManifest,
    train_production_bpe,
)


def test_production_contract_is_fixed():
    assert PRODUCTION_VOCAB_SIZE == 32768
    assert tuple(SPECIAL_TOKEN_IDS) == SPECIAL_TOKENS
    assert SPECIAL_TOKEN_IDS["<|pad|>"] == 0
    assert SPECIAL_TOKEN_IDS["<|meta_sep|>"] == 15


def test_manifest_rejects_unpinned_values():
    with pytest.raises(ValueError):
        ProductionTokenizerManifest(
            format="aya-edge-production-bpe",
            version=1,
            tokenizer_sha256="PENDING",
            corpus_manifest_sha256="real-corpus-hash",
            implementation_commit="abc1234",
            tokenizers_version="0.20.0",
            vocab_size=PRODUCTION_VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS,
        )


def test_training_requires_optional_dependency_or_enough_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello tool result\n", encoding="utf-8")
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="optional 'tokenizers'"):
            train_production_bpe([corpus], tmp_path / "tokenizer.json")
    else:
        with pytest.raises(ValueError, match="expected vocabulary size"):
            train_production_bpe([corpus], tmp_path / "tokenizer.json")
