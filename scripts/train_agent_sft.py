"""Run a bounded AYA Edge agent-SFT experiment on validated JSONL trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import yaml

from aya_edge import AyaEdgeConfig, AyaEdgeForCausalLM
from aya_edge.tokenizer.bpe import ProductionBPETokenizer, SPECIAL_TOKEN_IDS
from aya_edge.training.agent_sft import collate, iter_batches, read_encoded_jsonl
from aya_edge.training.checkpoint import load_checkpoint, save_checkpoint, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/aya_edge_0_2_language.yaml"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/aya_edge_agent_sft.pt"))
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def _git_state(root: Path) -> dict[str, object]:
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False)
    return {"commit": commit.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}


def evaluate(model, examples, *, batch_size: int, device: torch.device) -> tuple[float, int]:
    """Return token-weighted masked validation loss and supervised-token count."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_examples in iter_batches(examples, batch_size):
            input_ids, labels = collate(batch_examples, pad_token_id=SPECIAL_TOKEN_IDS["<|pad|>"])
            input_ids, labels = input_ids.to(device), labels.to(device)
            result = model(input_ids, labels=labels)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError("non-finite agent SFT validation loss")
            supervised = int((labels != -100).sum())
            total_loss += float(result["loss"]) * supervised
            total_tokens += supervised
    model.train()
    if total_tokens == 0:
        raise ValueError("validation contains no supervised assistant tokens")
    return total_loss / total_tokens, total_tokens


def train(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    git = _git_state(root)
    if git["dirty"] and not args.allow_dirty:
        raise SystemExit("refusing agent SFT from a dirty worktree; commit code or pass --allow-dirty")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = AyaEdgeConfig(**raw["model"])
    train_cfg = raw["training"]
    tokenizer = ProductionBPETokenizer.load(args.tokenizer)
    tokenizer.assert_model_alignment(cfg)
    max_length = min(cfg.max_seq_len, args.max_length or int(train_cfg.get("sequence_length", cfg.max_seq_len)))
    train_examples = read_encoded_jsonl(args.input_jsonl, tokenizer, max_length=max_length)
    validation_examples = read_encoded_jsonl(args.validation_jsonl, tokenizer, max_length=max_length)
    set_seed(int(train_cfg["seed"]), deterministic=bool(train_cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AyaEdgeForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    max_steps = args.max_steps if args.max_steps is not None else int(train_cfg["max_steps"])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    step = 0
    if args.resume:
        metadata = load_checkpoint(args.resume, model, optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        step = int(metadata.get("step", 0))
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    validation_interval = args.validation_interval or int(train_cfg.get("validation_interval_steps", 100))
    if validation_interval < 1:
        raise ValueError("validation interval must be positive")
    metrics_path = args.metrics or args.output.with_suffix(".metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model.train()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        supervised = 0
        for micro in range(accumulation):
            batch_examples = train_examples[((step * accumulation + micro) * args.batch_size) % len(train_examples):]
            batch_examples = list(batch_examples[:args.batch_size]) or train_examples[:args.batch_size]
            input_ids, labels = collate(batch_examples, pad_token_id=SPECIAL_TOKEN_IDS["<|pad|>"])
            input_ids, labels = input_ids.to(device), labels.to(device)
            supervised += int((labels != -100).sum())
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(input_ids, labels=labels)
                loss = (result["loss"] + float(train_cfg.get("aux_loss_weight", 0.0)) * result["aux_loss"]) / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite agent SFT loss")
            scaler.scale(loss).backward()
            total_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite agent SFT gradient norm")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        record = {"step": step, "loss": total_loss, "supervised_tokens": supervised, "grad_norm": float(grad_norm), "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": time.time() - started}
        if step == 1 or step % validation_interval == 0 or step == max_steps:
            validation_loss, validation_tokens = evaluate(
                model, validation_examples, batch_size=args.batch_size, device=device
            )
            record["validation_loss"] = validation_loss
            record["validation_supervised_tokens"] = validation_tokens
        if step == 1 or step % 10 == 0 or step == max_steps:
            print(json.dumps(record))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        interval = int(train_cfg.get("checkpoint_interval_steps", 500))
        if step % interval == 0 or step == max_steps:
            save_checkpoint(args.output, model, optimizer, scheduler=scheduler, scaler=scaler, step=step, tokens_processed=step * args.batch_size * accumulation * max_length, config=cfg, dataset_version=f"agent-sft={args.input_jsonl};validation={args.validation_jsonl};tokenizer={args.tokenizer}", run={"git": git, "train_examples": len(train_examples), "validation_examples": len(validation_examples), "max_length": max_length, "objective": "assistant_only_loss"})
            print(json.dumps({"checkpoint": str(args.output), "step": step}))


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
