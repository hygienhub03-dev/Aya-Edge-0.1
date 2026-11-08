"""One-step, synthetic-data DDP infrastructure smoke test for AYA Edge 0.2.

When a non-production tokenizer artifact is present at
``artifacts/tokenizer_nonprod.json`` the smoke test loads it, asserts the
locked ID layout (0–15 specials / 16–271 byte fallback / 272–32767 BPE),
and verifies model/tokenizer alignment. Synthetic input IDs are always
drawn from the full production range 0…32767.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from aya_edge import AyaEdgeConfig, AyaEdgeForCausalLM, count_active_parameters, count_parameters

MODEL_VOCAB_SIZE = 32768
SEQUENCE_LENGTH = 64
NONPROD_TOKENIZER_PATH = Path("artifacts/tokenizer_nonprod.json")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("dual_t4_ddp_smoke requires exactly two visible CUDA devices")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"expected WORLD_SIZE=2, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        raw_config = yaml.safe_load(Path("configs/aya_edge_0_2.yaml").read_text(encoding="utf-8"))
        cfg = AyaEdgeConfig(**raw_config["model"])
        if cfg.vocab_size != MODEL_VOCAB_SIZE:
            raise RuntimeError("model vocabulary changed; update this smoke-test contract")

        tokenizer_loaded = False
        tokenizer_path = None
        if NONPROD_TOKENIZER_PATH.exists():
            from aya_edge.tokenizer import ProductionBPETokenizer
            tok = ProductionBPETokenizer.load(NONPROD_TOKENIZER_PATH)
            tok.assert_model_alignment(cfg)
            tokenizer_loaded = True
            tokenizer_path = str(NONPROD_TOKENIZER_PATH)
            if rank == 0:
                print(f"Loaded NON-PRODUCTION tokenizer from {tokenizer_path} (vocab={tok.vocab_size})")

        torch.manual_seed(42 + rank)
        model = AyaEdgeForCausalLM(cfg).to(device)
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        # Full production vocabulary range (0 … 32767)
        input_ids = torch.randint(0, MODEL_VOCAB_SIZE, (1, SEQUENCE_LENGTH), device=device)
        before = ddp_model.module.embed.weight.detach()[0, :8].clone()
        start = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = ddp_model(input_ids, labels=input_ids)
            loss = output["loss"] + 0.01 * output["aux_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"rank {rank}: non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        after = ddp_model.module.embed.weight.detach()[0, :8].clone()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        changed = not torch.equal(before, after)
        dist.barrier()
        if rank == 0:
            output_dir = Path("outputs/dual_t4_smoke")
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = output_dir / "aya_edge_02_ddp_smoke.pt"
            torch.save({
                "model": ddp_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": 1,
                "config": cfg.to_dict(),
                "purpose": "synthetic DDP smoke test only",
                "tokenizer_vocab_size": MODEL_VOCAB_SIZE,
                "model_vocab_size": cfg.vocab_size,
                "tokenizer_loaded": tokenizer_loaded,
                "tokenizer_path": tokenizer_path,
            }, checkpoint)
            metrics = {
                "status": "PASSED",
                "world_size": world_size,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
                "parameters": count_parameters(ddp_model.module),
                "active_parameter_estimate": count_active_parameters(ddp_model.module),
                "loss": float(loss.detach().cpu()),
                "aux_loss": float(output["aux_loss"].detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "embedding_weights_changed": bool(changed),
                "elapsed_seconds_rank0": elapsed,
                "checkpoint": str(checkpoint),
                "training_data": "synthetic token IDs drawn from full 0..32767 production range",
                "production_training_allowed": False,
                "tokenizer_loaded": tokenizer_loaded,
                "tokenizer_path": tokenizer_path,
                "id_layout": "0-15 specials / 16-271 byte fallback / 272-32767 BPE",
            }
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            print("DDP_SMOKE_METRICS=" + json.dumps(metrics))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
