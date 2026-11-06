from dataclasses import dataclass

@dataclass
class AyaEdgeConfig:
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

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
