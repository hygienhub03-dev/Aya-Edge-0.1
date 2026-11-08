"""Small, readable AYA Edge architecture prototype.

MLA-inspired latent attention with RoPE.
Simple top-k MoE with shared expert and load-balancing loss.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import AyaEdgeConfig


class LatentAttention(nn.Module):
    """MLA-inspired latent attention with rotary positional embeddings."""

    def __init__(self, cfg: AyaEdgeConfig):
        super().__init__()
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_heads = cfg.n_heads

        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_down = nn.Linear(cfg.d_model, cfg.latent_dim, bias=False)
        self.k_up = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.v_up = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        inv_freq = 1.0 / (
            10000 ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)

        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()[None, None, :, :].to(dtype=q.dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=q.dtype)

        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        latent = self.kv_down(x)
        k = self.k_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.apply_rope(q, k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TopKMoE(nn.Module):
    def __init__(self, cfg: AyaEdgeConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router_z_loss_weight = float(cfg.router_z_loss_weight)
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.ffn_dim) for _ in range(cfg.n_experts)]
        )
        self.shared = (
            SwiGLU(cfg.d_model, cfg.ffn_dim) if cfg.use_shared_expert else None
        )
        self.last_routing_stats: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        logits = self.router(flat)
        probs = logits.softmax(dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        routed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = (indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel():
                contribution = expert(flat[token_idx]) * weights[token_idx, slot_idx, None]
                routed.index_add_(0, token_idx, contribution)

        if self.shared is not None:
            routed = routed + self.shared(flat)

        # Switch-style load-balancing auxiliary loss
        assignment = F.one_hot(indices, self.n_experts).float().mean(dim=(0, 1))
        importance = probs.mean(dim=0)
        balance_loss = self.n_experts * torch.sum(assignment * importance)
        router_z_loss = logits.logsumexp(dim=-1).square().mean()
        aux_loss = balance_loss + self.router_z_loss_weight * router_z_loss
        safe_probs = probs.clamp_min(1e-9)
        self.last_routing_stats = {
            "assignment_fraction": assignment.detach(),
            "importance_fraction": importance.detach(),
            "router_entropy": (-(safe_probs * safe_probs.log()).sum(dim=-1).mean()).detach(),
            "balance_loss": balance_loss.detach(),
            "router_z_loss": router_z_loss.detach(),
            "selected_expert_ids": indices.detach(),
        }

        return routed.view(shape), aux_loss


class Block(nn.Module):
    def __init__(self, cfg: AyaEdgeConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = LatentAttention(cfg)
        self.moe_norm = nn.LayerNorm(cfg.d_model)
        self.moe = TopKMoE(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        moe_out, aux = self.moe(self.moe_norm(x))
        return x + moe_out, aux


class AyaEdgeForCausalLM(nn.Module):
    def __init__(self, config: AyaEdgeConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        x = self.embed(input_ids)

        aux_losses = []
        for block in self.blocks:
            x, aux = block(x)
            aux_losses.append(aux)

        logits = self.lm_head(self.norm(x))

        aux_loss = (
            torch.stack(aux_losses).mean()
            if aux_losses
            else logits.new_zeros(())
        )
        routing_stats = {
            str(i): block.moe.last_routing_stats
            for i, block in enumerate(self.blocks)
            if block.moe.last_routing_stats
        }

        loss = None
        if labels is not None:
            # ignore_index=-100 is the default; keep it explicit for clarity
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "routing_stats": routing_stats,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Use only the last context window if needed
            context = generated[:, -self.config.max_seq_len :]
            logits = self(context)["logits"][:, -1, :]  # (B, V)

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_active_parameters(model: AyaEdgeForCausalLM) -> int:
    """Estimate parameters used for one token under the configured router.

    This is a structural estimate, not a runtime FLOP count. Tied embedding
    and output weights are counted once; selected routed experts and the shared
    expert are counted per layer.
    """
    cfg = model.config
    total = model.embed.weight.numel()
    for name, parameter in model.named_parameters():
        if name == "embed.weight" or ".moe.experts." in name or ".moe.shared." in name:
            continue
        total += parameter.numel()
    expert_params = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    total += cfg.n_layers * cfg.top_k * expert_params
    if model.blocks[0].moe.shared is not None:
        shared_params = sum(p.numel() for p in model.blocks[0].moe.shared.parameters())
        total += cfg.n_layers * shared_params
    return total
