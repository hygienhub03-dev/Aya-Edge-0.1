import torch
from aya_edge import AyaEdgeConfig, AyaEdgeForCausalLM


def tiny_config():
    return AyaEdgeConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=4,
                         latent_dim=8, ffn_dim=64, n_experts=4, top_k=2,
                         max_seq_len=32)


def test_forward_shapes_and_losses():
    model = AyaEdgeForCausalLM(tiny_config())
    tokens = torch.randint(0, 128, (2, 12))
    result = model(tokens, labels=tokens)
    assert result["logits"].shape == (2, 12, 128)
    assert result["loss"].ndim == 0
    assert result["aux_loss"].ndim == 0


def test_invalid_top_k_rejected():
    try:
        AyaEdgeConfig(n_experts=4, top_k=5)
    except ValueError:
        return
    raise AssertionError("Expected invalid top_k to raise ValueError")
