"""A fixed-vocabulary UTF-8 byte tokenizer.

The first 256 token IDs are always the corresponding byte values. Special tokens
occupy a deterministic block after the byte vocabulary. No training data is
needed, and loading a config never grows or otherwise changes the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

BYTE_VOCAB_SIZE = 256
DEFAULT_SPECIAL_TOKENS = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "unk": "<|unk|>",
    "turn_start": "<|turn_start|>",
    "turn_end": "<|turn_end|>",
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "tool_call": "<|tool_call|>",
    "tool_result": "<|tool_result|>",
}


@dataclass(frozen=True)
class TokenizerConfig:
    """Serializable tokenizer policy; IDs are assigned by special-token order."""

    special_tokens: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SPECIAL_TOKENS)
    )
    byte_vocab_size: int = BYTE_VOCAB_SIZE
    normalization: str = "none"
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.byte_vocab_size != BYTE_VOCAB_SIZE:
            raise ValueError("byte_vocab_size must remain 256 for byte-level compatibility")
        if self.normalization != "none":
            raise ValueError("only normalization='none' is supported")
        if self.encoding.lower().replace("_", "-") != "utf-8":
            raise ValueError("only UTF-8 encoding is supported")
        tokens = dict(self.special_tokens)
        if tuple(tokens) != tuple(DEFAULT_SPECIAL_TOKENS):
            raise ValueError("special token names and order are fixed; vocabulary changes are not silent")
        if len(set(tokens.values())) != len(tokens):
            raise ValueError("special token strings must be unique")
        if any(not isinstance(value, str) or not value for value in tokens.values()):
            raise ValueError("special token strings must be non-empty strings")

    @property
    def vocab_size(self) -> int:
        return self.byte_vocab_size + len(self.special_tokens)

    def to_dict(self) -> dict:
        return {
            "format": "aya-edge-byte-tokenizer",
            "version": 1,
            "byte_vocab_size": self.byte_vocab_size,
            "encoding": self.encoding,
            "normalization": self.normalization,
            "special_tokens": dict(self.special_tokens),
            "special_token_ids": {
                name: self.byte_vocab_size + index
                for index, name in enumerate(self.special_tokens)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TokenizerConfig":
        if data.get("format") != "aya-edge-byte-tokenizer":
            raise ValueError("not an Aya Edge byte tokenizer config")
        if data.get("version") != 1:
            raise ValueError("unsupported tokenizer config version")
        config = cls(
            byte_vocab_size=int(data.get("byte_vocab_size", BYTE_VOCAB_SIZE)),
            encoding=str(data.get("encoding", "utf-8")),
            normalization=str(data.get("normalization", "none")),
            special_tokens=dict(data.get("special_tokens", DEFAULT_SPECIAL_TOKENS)),  # type: ignore[arg-type]
        )
        expected_ids = config.to_dict()["special_token_ids"]
        if data.get("special_token_ids", expected_ids) != expected_ids:
            raise ValueError("special token IDs do not match the deterministic layout")
        return config


class ByteLevelTokenizer:
    """Encode UTF-8 bytes directly, with deterministic special-token support."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.special_tokens = dict(self.config.special_tokens)
        self.special_token_ids = {
            name: BYTE_VOCAB_SIZE + index
            for index, name in enumerate(self.special_tokens)
        }
        self.id_to_special = {token_id: token for token_id, token in (
            (token_id, self.special_tokens[name])
            for name, token_id in self.special_token_ids.items()
        )}
        self._special_by_text = sorted(
            ((token, token_id) for token_id, token in self.id_to_special.items()),
            key=lambda item: (-len(item[0]), item[0]),
        )

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def token_id(self, name: str) -> int:
        try:
            return self.special_token_ids[name]
        except KeyError as exc:
            raise KeyError(f"unknown special token name: {name}") from exc

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text; recognized special-token strings become one token each."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids: list[int] = [self.token_id("bos")] if add_bos else []
        position = 0
        while position < len(text):
            for special_text, token_id in self._special_by_text:
                if text.startswith(special_text, position):
                    ids.append(token_id)
                    position += len(special_text)
                    break
            else:
                next_special = min(
                    (text.find(special_text, position) for special_text, _ in self._special_by_text
                     if text.find(special_text, position) >= 0),
                    default=len(text),
                )
                ids.extend(text[position:next_special].encode(self.config.encoding))
                position = next_special
        if add_eos:
            ids.append(self.token_id("eos"))
        return ids

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
        errors: str = "strict",
    ) -> str:
        """Decode IDs to text, preserving special markers unless skipped."""
        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                output.append(bytes(byte_buffer).decode(self.config.encoding, errors=errors))
                byte_buffer.clear()

        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token IDs must be integers")
            if 0 <= token_id < BYTE_VOCAB_SIZE:
                byte_buffer.append(token_id)
            elif token_id in self.id_to_special:
                flush_bytes()
                if not skip_special_tokens:
                    output.append(self.id_to_special[token_id])
            else:
                raise ValueError(f"token ID out of range: {token_id}")
        flush_bytes()
        return "".join(output)

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode role/content messages with explicit turn boundaries."""
        ids: list[int] = [self.token_id("bos")] if add_bos else []
        for message in messages:
            if set(message) != {"role", "content"}:
                raise ValueError("each message must contain exactly role and content")
            role = message["role"]
            content = message["content"]
            if role not in self.special_token_ids:
                raise ValueError(f"unsupported message role: {role}")
            ids.extend((self.token_id("turn_start"), self.token_id(role)))
            ids.extend(self.encode(content))
            ids.append(self.token_id("turn_end"))
        if add_eos:
            ids.append(self.token_id("eos"))
        return ids

    def decode_messages(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Decode a conversation sequence; this is a text view, not a parser."""
        return self.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        """Write a stable, human-readable JSON configuration."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("tokenizer config must contain a JSON object")
        return cls(TokenizerConfig.from_dict(data))
