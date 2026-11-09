"""Production tokenizer contract for AYA Edge.

The byte tokenizer remains the offline development adapter. This module wraps
Hugging Face ``tokenizers`` for the approved production direction: byte-level
BPE with a fixed 32,768 vocabulary and the following locked ID layout:

  0–15:     16 reserved special tokens
  16–271:   256 byte-alphabet fallback tokens (ByteLevel initial alphabet)
  272–32767: 32,496 learned BPE entries

No production artifact is checked in until corpus approval is complete.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PRODUCTION_VOCAB_SIZE = 32_768
SPECIAL_TOKEN_COUNT = 16
BYTE_FALLBACK_START = 16
BYTE_FALLBACK_END = 271  # inclusive; 256 tokens
BPE_START = 272

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
    "<|im_start|>", "<|im_sep|>", "<|im_end|>", "<|tool_call|>",
    "<|tool_result|>", "<|analysis|>", "<|final|>", "<|commentary|>",
    "<|json|>", "<|refusal|>", "<|end_of_task|>", "<|meta_sep|>",
)
SPECIAL_TOKEN_IDS = {token: index for index, token in enumerate(SPECIAL_TOKENS)}

assert len(SPECIAL_TOKENS) == SPECIAL_TOKEN_COUNT
assert BYTE_FALLBACK_END - BYTE_FALLBACK_START + 1 == 256
assert BPE_START == BYTE_FALLBACK_END + 1
assert PRODUCTION_VOCAB_SIZE - BPE_START == 32_496


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


def byte_fallback_alphabet() -> list[str]:
    """Return the 256-character ByteLevel alphabet in trainer insertion order."""
    _, _, _, ByteLevel, _ = _require_tokenizers()
    alphabet = list(ByteLevel.alphabet())
    if len(alphabet) != 256 or len(set(alphabet)) != 256:
        raise RuntimeError("ByteLevel.alphabet() must contain 256 distinct characters")
    return alphabet


def iter_corpus_texts(files: Sequence[str | Path]) -> Iterable[str]:
    """Yield document text. JSONL contributes the ``text`` field, not the raw row."""
    found = False
    for path in files:
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            if source.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
                    if not isinstance(row, dict):
                        raise TypeError(f"expected object at {source}:{line_number}")
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"{source}:{line_number} is missing non-empty text")
                    found = True
                    yield text
            else:
                for line in handle:
                    if line.strip():
                        found = True
                        yield line
    if not found:
        raise ValueError("corpus files contain no training text")


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
    min_frequency: int = 2

    def __post_init__(self) -> None:
        if self.format != "aya-edge-production-bpe":
            raise ValueError("invalid production tokenizer manifest format")
        if self.version != 1:
            raise ValueError("unsupported tokenizer manifest version")
        if self.vocab_size != PRODUCTION_VOCAB_SIZE:
            raise ValueError("production tokenizer vocabulary must be exactly 32768")
        if tuple(self.special_tokens) != SPECIAL_TOKENS:
            raise ValueError("production special-token contract changed")
        if not isinstance(self.min_frequency, int) or isinstance(self.min_frequency, bool) or self.min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer")
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
            "min_frequency": self.min_frequency,
            "normalization": "none",
            "pretokenizer": "ByteLevel(add_prefix_space=False)",
            "byte_fallback_verified": True,
            "id_layout": {
                "specials": "0-15",
                "byte_fallback": "16-271",
                "learned_bpe": "272-32767",
            },
        }


def assert_locked_id_layout(backend, *, expected_size: int) -> None:
    """Require specials at 0–15 and the ByteLevel alphabet at 16–271, in order."""
    size = backend.get_vocab_size(with_added_tokens=True)
    if size != expected_size:
        raise ValueError(
            f"expected vocabulary size {expected_size}, got {size}; "
            "the corpus did not contain enough repeated text to fill the vocabulary"
        )
    if expected_size < BPE_START or expected_size > PRODUCTION_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be between {BPE_START} and {PRODUCTION_VOCAB_SIZE}")
    vocab = backend.get_vocab()
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        actual = vocab.get(token)
        if actual != expected_id:
            raise ValueError(f"special token {token} must have ID {expected_id}, got {actual}")
    alphabet = byte_fallback_alphabet()
    for offset, token in enumerate(alphabet):
        expected_id = BYTE_FALLBACK_START + offset
        actual = vocab.get(token)
        if actual != expected_id:
            raise ValueError(
                f"byte-fallback token {offset} must occupy ID {expected_id}, got {actual}"
            )
    for token, token_id in vocab.items():
        if token in SPECIAL_TOKEN_IDS and token_id >= BYTE_FALLBACK_START:
            raise ValueError(f"special token {token} must not occupy ID {token_id}")


class ProductionBPETokenizer:
    """Validated wrapper around a trained byte-level BPE tokenizer JSON."""

    def __init__(
        self,
        backend,
        *,
        artifact_path: str | Path | None = None,
        expected_vocab_size: int = PRODUCTION_VOCAB_SIZE,
    ) -> None:
        self._backend = backend
        self.artifact_path = Path(artifact_path) if artifact_path is not None else None
        self._expected_vocab_size = expected_vocab_size
        self._validate_contract()

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    def _validate_contract(self) -> None:
        assert_locked_id_layout(self._backend, expected_size=self._expected_vocab_size)
        if self._expected_vocab_size == PRODUCTION_VOCAB_SIZE and self.vocab_size != PRODUCTION_VOCAB_SIZE:
            raise ValueError(f"expected vocabulary size {PRODUCTION_VOCAB_SIZE}, got {self.vocab_size}")

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


def train_byte_level_bpe(
    files: Sequence[str | Path],
    output_path: str | Path,
    *,
    vocab_size: int,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> ProductionBPETokenizer:
    """Train byte-level BPE and require the locked special and byte-fallback IDs.

    ``vocab_size`` includes the 16 special tokens. Production training must pass
    32,768. Smaller sizes exist so tests can prove the ID layout without a
    corpus large enough to fill every learned merge.
    """
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        raise TypeError("vocab_size must be an integer")
    if vocab_size < BPE_START or vocab_size > PRODUCTION_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be between {BPE_START} and {PRODUCTION_VOCAB_SIZE}")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")
    if not files:
        raise ValueError("at least one corpus file is required")
    Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer = _require_tokenizers()
    alphabet = byte_fallback_alphabet()
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=alphabet,
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(iter_corpus_texts(files), trainer=trainer)
    trained = ProductionBPETokenizer(tokenizer, expected_vocab_size=vocab_size)
    trained.save(output_path)
    if vocab_size == PRODUCTION_VOCAB_SIZE:
        return ProductionBPETokenizer.load(output_path)
    return trained


def train_production_bpe(
    files: Sequence[str | Path],
    output_path: str | Path,
    *,
    vocab_size: int = PRODUCTION_VOCAB_SIZE,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> ProductionBPETokenizer:
    """Train the production artifact. Corpus approval is enforced by the CLI."""
    if vocab_size != PRODUCTION_VOCAB_SIZE:
        raise ValueError("production BPE vocabulary is fixed at 32768")
    return train_byte_level_bpe(
        files,
        output_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
    )


def write_manifest(
    tokenizer_path: str | Path,
    manifest_path: str | Path,
    *,
    corpus_manifest_sha256: str,
    implementation_commit: str,
    min_frequency: int = 2,
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
        min_frequency=min_frequency,
    )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return manifest
