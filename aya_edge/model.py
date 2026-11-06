"""Small, readable AYA Edge architecture prototype.

The attention module is MLA-inspired, not a faithful production MLA implementation.
The MoE dispatch uses a simple token loop for clarity, not throughput.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .config import AyaEdgeConfig


class LatentAttention(nn.Module):
    def __init__(self, cfg: AyaEdgeConfig):
        super().__init__()
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_heads = cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_down = nn.Linear(cfg.d_model, cfg.latent_dim, bias=False)
        self.k_up = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.v_up = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        latent = self.kv_down(x)
        k = self.k_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TopKMoE(nn.Module):
    def __init__(self, cfg: AyaEdgeConfig):
        super().__init__()
        self.n_experts, self.top_k = cfg.n_experts, cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, cfg.ffn_dim) for _ in range(cfg.n_experts)])
        self.shared = SwiGLU(cfg.d_model, cfg.ffn_dim) if cfg.use_shared_expert else None

    def forward(self, x):
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
        # Switch-style auxiliary load-balancing term.
        assignment = F.one_hot(indices, self.n_experts).float().mean(dim=(0, 1))
        importance = probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(assignment * importance)
        return routed.view(shape), aux_loss


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = LatentAttention(cfg)
        self.moe_norm = nn.LayerNorm(cfg.d_model)
        self.moe = TopKMoE(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        moe_out, aux = self.moe(self.moe_norm(x))
        return x + moe_out, aux


class AyaEdgeForCausalLM(nn.Module):
    def __init__(self, config: AyaEdgeConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Tie input embeddings and output projection
        self.lm_head.weight = self.embed.weight

        # Controlled token embedding initialisation
        nn.init.normal_(
            self.embed.weight,
            mean=0.0,
            std=0.02
        )

    def forward(self, input_ids, labels=None):
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

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
