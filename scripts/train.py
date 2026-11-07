"""Run a resumable AYA Edge experiment; dry-run is safe for architecture inspection."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

from aya_edge import AyaEdgeConfig, AyaEdgeForCausalLM, count_active_parameters, count_parameters
from aya_edge.tokenizer import ByteLevelTokenizer
from aya_edge.training import load_checkpoint, save_checkpoint, set_seed


def packed_tokens(path: Path, tokenizer: ByteLevelTokenizer, seq_len: int) -> torch.Tensor:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        values.extend(tokenizer.encode(str(row["text"]), add_eos=True))
    usable = len(values) // seq_len * seq_len
    if usable == 0:
        raise ValueError("input contains fewer than one complete packed sequence")
    return torch.tensor(values[:usable], dtype=torch.long).view(-1, seq_len)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/aya_edge_0_2.yaml"))
parser.add_argument("--input-jsonl", type=Path)
parser.add_argument("--output", type=Path, default=Path("outputs/aya_edge_0_2_latest.pt"))
parser.add_argument("--resume", type=Path)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--dry-run", action="store_true")
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
    "status": "architecture smoke test; not production training",
}, indent=2, default=str))
if args.dry_run:
    raise SystemExit(0)
if args.input_jsonl is None:
    raise SystemExit("--input-jsonl is required unless --dry-run is used")

# The byte tokenizer is a deterministic development adapter. A production run
# must replace it with an approved 32,768-vocabulary artifact and record its hash.
tokenizer = ByteLevelTokenizer()
if tokenizer.vocab_size > cfg.vocab_size:
    raise ValueError("tokenizer vocabulary exceeds model vocabulary")
tokens = packed_tokens(args.input_jsonl, tokenizer, cfg.max_seq_len)
optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
step = 0
if args.resume:
    metadata = load_checkpoint(args.resume, model, optimizer, scaler=scaler, map_location=device)
    step = int(metadata.get("step", 0))
    print("resumed", metadata)
max_steps = args.max_steps if args.max_steps is not None else int(train_cfg["max_steps"])
accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
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
    step += 1
    if step == 1 or step % 10 == 0:
        tokens_per_second = step * accumulation * cfg.max_seq_len / max(time.time() - start, 1e-9)
        print(json.dumps({"step": step, "loss": total_loss, "grad_norm": float(grad_norm), "tokens_per_second": tokens_per_second}))
    if step % int(train_cfg["checkpoint_interval_steps"]) == 0 or step == max_steps:
        save_checkpoint(args.output, model, optimizer, scaler=scaler, step=step, tokens_processed=step * accumulation * cfg.max_seq_len, config=cfg, dataset_version=str(args.input_jsonl))
        print("checkpoint", args.output)
