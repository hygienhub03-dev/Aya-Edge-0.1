import torch

from aya_edge import (
    AyaEdgeConfig,
    AyaEdgeForCausalLM,
    count_active_parameters,
    load_checkpoint,
    save_checkpoint,
)


def tiny_config():
    return AyaEdgeConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        latent_dim=8,
        ffn_dim=64,
        n_experts=4,
        top_k=2,
        max_seq_len=32,
    )


def test_forward_shapes_losses_and_routing_diagnostics():
    model = AyaEdgeForCausalLM(tiny_config())
    tokens = torch.randint(0, 128, (2, 12))
    result = model(tokens, labels=tokens)
    assert result["logits"].shape == (2, 12, 128)
    assert result["loss"].ndim == 0
    assert result["aux_loss"].ndim == 0
    assert set(result["routing_stats"]) == {"0", "1"}
    assert result["routing_stats"]["0"]["assignment_fraction"].shape == (4,)


def test_invalid_top_k_rejected():
    try:
        AyaEdgeConfig(n_experts=4, top_k=5)
    except ValueError:
        return
    raise AssertionError("Expected invalid top_k to raise ValueError")


def test_invalid_dimensions_rejected():
    for kwargs in (
        {"vocab_size": 0},
        {"d_model": 30, "n_heads": 4},
        {"d_model": 32, "n_heads": 4, "latent_dim": 64},
    ):
        try:
            AyaEdgeConfig(**kwargs)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Expected invalid config to raise: {kwargs}")


def test_embedding_initialization_and_weight_tying():
    config = tiny_config()
    model = AyaEdgeForCausalLM(config)
    assert model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert abs(model.embed.weight.std().item() - 0.02) < 0.002


def test_generation_uses_sliding_context_window():
    config = tiny_config()
    model = AyaEdgeForCausalLM(config)
    prompt = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    generated = model.generate(prompt, max_new_tokens=3, temperature=1.0)
    assert generated.shape == (1, config.max_seq_len + 3)


def test_generation_rejects_invalid_sampling_values():
    model = AyaEdgeForCausalLM(tiny_config())
    tokens = torch.randint(0, 128, (1, 4))
    for kwargs in ({"temperature": 0}, {"top_k": 0}, {"max_new_tokens": -1}):
        try:
            model.generate(tokens, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid generation args to raise: {kwargs}")


def test_active_parameter_estimate_is_below_total_for_sparse_moe():
    model = AyaEdgeForCausalLM(tiny_config())
    assert 0 < count_active_parameters(model) < sum(
        p.numel() for p in model.parameters()
    )


def test_checkpoint_round_trip(tmp_path):
    config = tiny_config()
    model = AyaEdgeForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, config.vocab_size, (2, 12))
    result = model(tokens, labels=tokens)
    result["loss"].backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        step=7,
        tokens_processed=168,
        config=config,
        dataset_version="test-v1",
    )
    restored = AyaEdgeForCausalLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    metadata = load_checkpoint(path, restored, restored_optimizer)
    assert metadata["step"] == 7
    assert metadata["tokens_processed"] == 168
    assert metadata["dataset_version"] == "test-v1"
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
