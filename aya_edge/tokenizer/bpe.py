"""Production tokenizer contract for AYA Edge.

The byte tokenizer remains the offline development adapter. This module wraps
Hugging Face ``tokenizers`` for the approved production direction: byte-level
BPE with a fixed 32,768 vocabulary and fixed IDs 0--15 for control tokens.
No production artifact is checked in until corpus approval is complete.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PRODUCTION_VOCAB_SIZE = 32_768
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
    "<|im_start|>", "<|im_sep|>", "<|im_end|>", "<|tool_call|>",
    "<|tool_result|>", "<|analysis|>", "<|final|>", "<|commentary|>",
    "<|json|>", "<|refusal|>", "<|end_of_task|>", "<|meta_sep|>",
)
SPECIAL_TOKEN_IDS = {token: index for index, token in enumerate(SPECIAL_TOKENS)}


def _require_tokenizers():
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "production BPE support requires the optional 'tokenizers' dependency; "
            "install the project's data extras"
        ) from exc
    return Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProductionTokenizerManifest:
    format: str
    version: int
    tokenizer_sha256: str
    corpus_manifest_sha256: str
    implementation_commit: str
    tokenizers_version: str
    vocab_size: int
    special_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.format != "aya-edge-production-bpe":
            raise ValueError("invalid production tokenizer manifest format")
        if self.version != 1:
            raise ValueError("unsupported tokenizer manifest version")
        if self.vocab_size != PRODUCTION_VOCAB_SIZE:
            raise ValueError("production tokenizer vocabulary must be exactly 32768")
        if tuple(self.special_tokens) != SPECIAL_TOKENS:
            raise ValueError("production special-token contract changed")
        for name, value in (("tokenizer_sha256", self.tokenizer_sha256), ("corpus_manifest_sha256", self.corpus_manifest_sha256), ("implementation_commit", self.implementation_commit), ("tokenizers_version", self.tokenizers_version)):
            if not value or value.startswith("TODO") or value.startswith("PENDING"):
                raise ValueError(f"{name} must be a real pinned value")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "version": self.version,
            "tokenizer_sha256": self.tokenizer_sha256,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "implementation_commit": self.implementation_commit,
            "tokenizers_version": self.tokenizers_version,
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            "special_token_ids": dict(SPECIAL_TOKEN_IDS),
        }


class ProductionBPETokenizer:
    """Validated wrapper around a trained byte-level BPE tokenizer JSON."""

    def __init__(self, backend, *, artifact_path: str | Path | None = None) -> None:
        self._backend = backend
        self.artifact_path = Path(artifact_path) if artifact_path is not None else None
        self._validate_contract()

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    def _validate_contract(self) -> None:
        if self.vocab_size != PRODUCTION_VOCAB_SIZE:
            raise ValueError(f"expected vocabulary size {PRODUCTION_VOCAB_SIZE}, got {self.vocab_size}")
        vocab = self._backend.get_vocab()
        for token, expected_id in SPECIAL_TOKEN_IDS.items():
            if vocab.get(token) != expected_id:
                raise ValueError(f"special token {token} must have ID {expected_id}, got {vocab.get(token)}")

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = self._backend.encode(text).ids
        if add_bos:
            ids.insert(0, SPECIAL_TOKEN_IDS["<|bos|>"])
        if add_eos:
            ids.append(SPECIAL_TOKEN_IDS["<|eos|>"])
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        return self._backend.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(destination))

    @classmethod
    def load(cls, path: str | Path) -> "ProductionBPETokenizer":
        Tokenizer, *_ = _require_tokenizers()
        return cls(Tokenizer.from_file(str(path)), artifact_path=path)

    def assert_model_alignment(self, model_or_config: object) -> None:
        model_vocab = getattr(model_or_config, "vocab_size", None)
        if model_vocab is None and hasattr(model_or_config, "config"):
            model_vocab = getattr(model_or_config.config, "vocab_size", None)
        if model_vocab != self.vocab_size:
            raise ValueError(f"tokenizer/model vocabulary mismatch: {self.vocab_size} != {model_vocab}")


def train_production_bpe(
    files: Sequence[str | Path],
    output_path: str | Path,
    *,
    vocab_size: int = PRODUCTION_VOCAB_SIZE,
    min_frequency: int = 2,
) -> ProductionBPETokenizer:
    """Train a byte-level BPE artifact; corpus approval is enforced by the CLI."""
    if vocab_size != PRODUCTION_VOCAB_SIZE:
        raise ValueError("production BPE vocabulary is fixed at 32768")
    if not files:
        raise ValueError("at least one approved corpus file is required")
    Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer = _require_tokenizers()
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=PRODUCTION_VOCAB_SIZE,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
    )
    tokenizer.train([str(Path(path)) for path in files], trainer)
    result = ProductionBPETokenizer(tokenizer)
    result.save(output_path)
    return ProductionBPETokenizer.load(output_path)


def write_manifest(
    tokenizer_path: str | Path,
    manifest_path: str | Path,
    *,
    corpus_manifest_sha256: str,
    implementation_commit: str,
) -> ProductionTokenizerManifest:
    try:
        import tokenizers
    except ImportError as exc:
        raise RuntimeError("tokenizers is required to write a production manifest") from exc
    artifact = ProductionBPETokenizer.load(tokenizer_path)
    manifest = ProductionTokenizerManifest(
        format="aya-edge-production-bpe",
        version=1,
        tokenizer_sha256=sha256_file(tokenizer_path),
        corpus_manifest_sha256=corpus_manifest_sha256,
        implementation_commit=implementation_commit,
        tokenizers_version=tokenizers.__version__,
        vocab_size=artifact.vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return manifest
