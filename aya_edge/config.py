from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class AyaEdgeConfig:
    """Configuration for the experimental approximately 85.1M scaffold."""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    latent_dim: int = 128
    ffn_dim: int = 1024
    n_experts: int = 4
    top_k: int = 2
    use_shared_expert: bool = True
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "latent_dim": self.latent_dim,
            "ffn_dim": self.ffn_dim,
            "n_experts": self.n_experts,
            "max_seq_len": self.max_seq_len,
        }
        non_integer = [
            name for name, value in positive.items()
            if not isinstance(value, int) or isinstance(value, bool)
        ]
        if non_integer:
            raise TypeError(
                f"Configuration values must be integers: {', '.join(non_integer)}"
            )
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {', '.join(invalid)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model // self.n_heads % 2 != 0:
            raise ValueError("d_model // n_heads must be even for RoPE")
        if self.latent_dim > self.d_model:
            raise ValueError("latent_dim should not exceed d_model")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        if not isinstance(self.use_shared_expert, bool):
            raise TypeError("use_shared_expert must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AyaEdgeConfig":
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {', '.join(unknown)}")
        return cls(**values)
