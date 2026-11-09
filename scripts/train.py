"""Train AYA Edge. A 32,768-vocabulary configuration requires the production tokenizer."""
from aya_edge.training.language import main

if __name__ == "__main__":
    main()
"""Run a resumable AYA Edge experiment.

Dry-run is safe for architecture inspection and uses the development byte
tokenizer. Any real (non-dry-run) training run requires an approved 32 768-
vocabulary production BPE tokenizer artifact and will refuse the development
adapter. This is a Milestone A enforcement point.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

from aya_edge import AyaEdgeConfig, AyaEdgeForCausalLM, count_active_parameters, count_parameters
from aya_edge.tokenizer import (
    ByteLevelTokenizer,
    PRODUCTION_VOCAB_SIZE,
    ProductionBPETokenizer,
)
from aya_edge.training import load_checkpoint, save_checkpoint, set_seed


def packed_tokens(path: Path, tokenizer, seq_len: int) -> torch.Tensor:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        values.extend(tokenizer.encode(str(row["text"]), add_eos=True))
    usable = len(values) // seq_len * seq_len
    if usable == 0:
        raise ValueError(f"{path} contains fewer than one complete packed sequence")
    return torch.tensor(values[:usable], dtype=torch.long).view(-1, seq_len)


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def evaluate(model, tokens: torch.Tensor, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index in range(min(len(tokens), max_batches)):
            batch = tokens[index:index + 1].to(device)
            losses.append(float(model(batch, labels=batch)["loss"]))
    model.train()
    return sum(losses) / max(1, len(losses))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path("configs/aya_edge_0_2.yaml"))
parser.add_argument("--input-jsonl", type=Path)
parser.add_argument("--validation-jsonl", type=Path)
parser.add_argument("--output", type=Path, default=Path("outputs/aya_edge_0_2_latest.pt"))
parser.add_argument("--resume", type=Path)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--dry-run", action="store_true",
                    help="Architecture inspection only; uses development byte tokenizer")
parser.add_argument(
    "--tokenizer",
    type=Path,
    help="Path to production tokenizer.json (required for non-dry-run training)",
)
args = parser.parse_args()

raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
cfg = AyaEdgeConfig(**raw["model"])
train_cfg = raw["training"]
set_seed(int(train_cfg["seed"]), deterministic=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AyaEdgeForCausalLM(cfg).to(device)
print(json.dumps({
    "device": str(device),
    "total_parameters": count_parameters(model),
    "active_parameter_estimate": count_active_parameters(model),
    "config": cfg.to_dict(),
    "status": "architecture smoke test; not production training" if args.dry_run else "production training path",
}, indent=2, default=str))

if args.dry_run:
    raise SystemExit(0)

if args.input_jsonl is None:
    raise SystemExit("--input-jsonl is required unless --dry-run is used")

# ------------------------------------------------------------------
# Milestone A gate: real training requires the production 32k BPE.
# The development byte tokenizer must never be used for genuine runs.
# ------------------------------------------------------------------
if args.tokenizer is None:
    raise SystemExit(
        "Real training requires --tokenizer pointing to an approved production "
        "32 768-vocabulary BPE artifact (tokenizer.json). "
        "The development ByteLevelTokenizer is refused for non-dry-run runs. "
        "Train it with scripts/train_bpe_tokenizer.py after corpus approval."
    )

if not args.tokenizer.exists():
    raise SystemExit(f"tokenizer artifact not found: {args.tokenizer}")

try:
    tokenizer = ProductionBPETokenizer.load(args.tokenizer)
except Exception as exc:
    raise SystemExit(f"failed to load production tokenizer: {exc}") from exc

if tokenizer.vocab_size != PRODUCTION_VOCAB_SIZE:
    raise SystemExit(
        f"tokenizer vocab_size {tokenizer.vocab_size} != required {PRODUCTION_VOCAB_SIZE}"
    )

if cfg.vocab_size != tokenizer.vocab_size:
    raise SystemExit(
        f"model/tokenizer vocabulary mismatch: model={cfg.vocab_size} "
        f"tokenizer={tokenizer.vocab_size}. Align configs before training."
    )

tokenizer.assert_model_alignment(cfg)
print(json.dumps({
    "tokenizer": str(args.tokenizer),
    "tokenizer_vocab_size": tokenizer.vocab_size,
    "model_vocab_size": cfg.vocab_size,
    "alignment": "ok",
}))

seq_len = min(cfg.max_seq_len, int(train_cfg.get("sequence_length", cfg.max_seq_len)))
tokens = packed_tokens(args.input_jsonl, tokenizer, seq_len)
validation_path = args.validation_jsonl or args.input_jsonl.with_name("validation.jsonl")
validation_tokens = packed_tokens(validation_path, tokenizer, seq_len) if validation_path.exists() else None
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(train_cfg["learning_rate"]),
    weight_decay=float(train_cfg["weight_decay"]),
)
max_steps = args.max_steps if args.max_steps is not None else int(train_cfg["max_steps"])
if max_steps <= 0:
    raise ValueError("max_steps must be positive")
scheduler = make_scheduler(optimizer, int(train_cfg.get("warmup_steps", 0)), max_steps)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
step = 0
if args.resume:
    metadata = load_checkpoint(
        args.resume, model, optimizer, scheduler=scheduler, scaler=scaler, map_location=device
    )
    step = int(metadata.get("step", 0))
    print("resumed", metadata)
accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
checkpoint_interval = int(train_cfg["checkpoint_interval_steps"])
validation_interval = int(train_cfg.get("validation_interval_steps", checkpoint_interval))
model.train()
start = time.time()
while step < max_steps:
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for micro_step in range(accumulation):
        batch = tokens[(step * accumulation + micro_step) % len(tokens)].unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch, labels=batch)
            loss = (output["loss"] + float(train_cfg["aux_loss_weight"]) * output["aux_loss"]) / accumulation
        scaler.scale(loss).backward()
        total_loss += float(loss.detach())
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite gradient norm")
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    step += 1
    metrics = {
        "step": step,
        "loss": total_loss,
        "grad_norm": float(grad_norm),
        "learning_rate": scheduler.get_last_lr()[0],
    }
    if validation_tokens is not None and (step == 1 or step % validation_interval == 0 or step == max_steps):
        metrics["validation_loss"] = evaluate(
            model, validation_tokens, device, int(train_cfg.get("validation_batches", 8))
        )
    if step == 1 or step % 10 == 0 or "validation_loss" in metrics:
        metrics["tokens_per_second"] = step * accumulation * seq_len / max(time.time() - start, 1e-9)
        print(json.dumps(metrics))
    if step % checkpoint_interval == 0 or step == max_steps:
        save_checkpoint(
            args.output,
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            tokens_processed=step * accumulation * seq_len,
            config=cfg,
            dataset_version=f"train={args.input_jsonl};validation={validation_path};tokenizer={args.tokenizer}",
        )
        print("checkpoint", args.output)
