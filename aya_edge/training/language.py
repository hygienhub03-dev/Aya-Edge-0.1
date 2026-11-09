"""First language-model training run for the AYA Edge 0.2 configuration.

The 32,768-vocabulary production tokenizer is mandatory whenever the model
vocabulary is 32,768. The 268-token byte tokenizer remains available only for
the smaller development configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import yaml

from aya_edge.config import AyaEdgeConfig
from aya_edge.model import AyaEdgeForCausalLM, count_active_parameters, count_parameters
from aya_edge.tokenizer import PRODUCTION_VOCAB_SIZE, ByteLevelTokenizer
from aya_edge.tokenizer.bpe import ProductionBPETokenizer, sha256_file
from aya_edge.training.checkpoint import load_checkpoint, save_checkpoint, set_seed


def git_state(root: Path) -> dict[str, object]:
    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=root, check=False, capture_output=True, text=True)

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    diff = run(["git", "diff", "HEAD"])
    commit_text = commit.stdout.strip() if commit.returncode == 0 else ""
    dirty_text = dirty.stdout if dirty.returncode == 0 else ""
    return {
        "commit": commit_text or None,
        "dirty": bool(dirty_text.strip()) if dirty.returncode == 0 else None,
        "diff_sha256": hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest() if diff.returncode == 0 else None,
    }


def packed_tokens(path: Path, tokenizer, seq_len: int) -> torch.Tensor:
    values: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number} is missing non-empty text")
        values.extend(tokenizer.encode(text, add_eos=True))
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
        limit = min(len(tokens), max_batches)
        for index in range(limit):
            batch = tokens[index:index + 1].to(device)
            loss = model(batch, labels=batch)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            losses.append(float(loss))
    model.train()
    if not losses:
        raise ValueError("validation produced no batches")
    return sum(losses) / len(losses)


def load_training_tokenizer(cfg: AyaEdgeConfig, tokenizer_path: Path | None, allow_byte_tokenizer: bool):
    if cfg.vocab_size == PRODUCTION_VOCAB_SIZE:
        if tokenizer_path is None:
            raise SystemExit(
                "the 32,768-vocabulary model requires --tokenizer pointing at the production BPE artifact"
            )
        tokenizer = ProductionBPETokenizer.load(tokenizer_path)
        tokenizer.assert_model_alignment(cfg)
        return tokenizer, {
            "kind": "production-bpe",
            "path": str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
        }
    if not allow_byte_tokenizer:
        raise SystemExit(
            "this configuration does not use the production vocabulary; pass --allow-byte-tokenizer "
            "only for a development run"
        )
    tokenizer = ByteLevelTokenizer()
    if tokenizer.vocab_size > cfg.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds model vocabulary")
    return tokenizer, {"kind": "byte-level-development", "vocab_size": tokenizer.vocab_size}


def run_block(block, hidden):
    return block(hidden)


def train(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = AyaEdgeConfig(**raw["model"])
    train_cfg = raw["training"]
    revision = git_state(root)
    set_seed(int(train_cfg["seed"]), deterministic=bool(train_cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AyaEdgeForCausalLM(cfg).to(device)
    summary = {
        "device": str(device),
        "total_parameters": count_parameters(model),
        "active_parameter_estimate": count_active_parameters(model),
        "config": cfg.to_dict(),
        "git": revision,
        "status": "dry-run" if args.dry_run else "language-training",
    }
    print(json.dumps(summary, indent=2, default=str))
    if args.dry_run:
        return
    if args.require_clean_git and revision["dirty"]:
        raise SystemExit(
            "refusing to train from a dirty worktree. Commit the training code, "
            "or pass --allow-dirty for a local run that cannot be cited as the recorded revision."
        )
    if not revision["commit"]:
        raise SystemExit("refusing to train: the Git commit could not be read")

    tokenizer_path = args.tokenizer
    if tokenizer_path is None and raw.get("tokenizer", {}).get("artifact"):
        tokenizer_path = Path(raw["tokenizer"]["artifact"])
    tokenizer, tokenizer_record = load_training_tokenizer(cfg, tokenizer_path, args.allow_byte_tokenizer)
    if args.input_jsonl is None:
        raise SystemExit("--input-jsonl is required unless --dry-run is used")
    validation_path = args.validation_jsonl or args.input_jsonl.with_name("validation.jsonl")
    if not validation_path.exists():
        raise SystemExit(f"validation JSONL is required and was not found: {validation_path}")

    seq_len = min(cfg.max_seq_len, int(train_cfg.get("sequence_length", cfg.max_seq_len)))
    tokens = packed_tokens(args.input_jsonl, tokenizer, seq_len)
    validation_tokens = packed_tokens(validation_path, tokenizer, seq_len)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    max_steps = args.max_steps if args.max_steps is not None else int(train_cfg["max_steps"])
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    scheduler = make_scheduler(optimizer, int(train_cfg.get("warmup_steps", 0)), max_steps)
    use_cuda_amp = device.type == "cuda" and str(train_cfg.get("precision", "auto")) != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    step = 0
    if args.resume:
        metadata = load_checkpoint(
            args.resume, model, optimizer, scheduler=scheduler, scaler=scaler, map_location=device
        )
        step = int(metadata.get("step", 0))
        print(json.dumps({"resumed_step": step}, default=str))

    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    checkpoint_interval = int(train_cfg["checkpoint_interval_steps"])
    validation_interval = int(train_cfg.get("validation_interval_steps", checkpoint_interval))
    checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    metrics_path = args.metrics or args.output.with_suffix(".metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_record = {
        "git": revision,
        "tokenizer": tokenizer_record,
        "model_config": cfg.to_dict(),
        "training_config": {key: value for key, value in train_cfg.items()},
        "sequence_length": seq_len,
        "train_sequences": int(tokens.shape[0]),
        "validation_sequences": int(validation_tokens.shape[0]),
    }
    model.train()
    start = time.time()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        token_count = 0
        for micro_step in range(accumulation):
            batch = tokens[(step * accumulation + micro_step) % len(tokens)].unsqueeze(0).to(device)
            token_count += int(batch.numel())
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda_amp):
                if checkpointing:
                    output = _forward_checkpointed(model, batch)
                else:
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
        elapsed = max(time.time() - start, 1e-9)
        metrics = {
            "step": step,
            "loss": total_loss,
            "grad_norm": float(grad_norm),
            "learning_rate": scheduler.get_last_lr()[0],
            "tokens_per_second": (step * accumulation * seq_len) / elapsed,
            "elapsed_seconds": elapsed,
            "token_count_this_step": token_count,
        }
        if device.type == "cuda":
            metrics["gpu_max_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        if step == 1 or step % validation_interval == 0 or step == max_steps:
            metrics["validation_loss"] = evaluate(
                model, validation_tokens, device, int(train_cfg.get("validation_batches", 8))
            )
        if step == 1 or step % 10 == 0 or "validation_loss" in metrics:
            print(json.dumps(metrics))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
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
                dataset_version=f"train={args.input_jsonl};validation={validation_path}",
                run=run_record,
            )
            print(json.dumps({"checkpoint": str(args.output), "step": step}))


def _forward_checkpointed(model: AyaEdgeForCausalLM, batch: torch.Tensor) -> dict:
    hidden = model.embed(batch)
    aux_losses = []
    for block in model.blocks:
        hidden, aux = torch.utils.checkpoint.checkpoint(run_block, block, hidden, use_reentrant=False)
        aux_losses.append(aux)
    logits = model.lm_head(model.norm(hidden))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        batch[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
    return {"loss": loss, "aux_loss": aux_loss}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AYA Edge on real language data")
    parser.add_argument("--config", type=Path, default=Path("configs/aya_edge_0_2_language.yaml"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--validation-jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/aya_edge_0_2_language.pt"))
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-byte-tokenizer", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.require_clean_git = not args.allow_dirty
    train(args)


if __name__ == "__main__":
    main()
