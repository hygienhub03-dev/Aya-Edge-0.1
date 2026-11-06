import argparse
import yaml
from aya_edge import AyaEdgeConfig, AyaEdgeForCausalLM
from aya_edge.model import count_parameters

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/aya_edge_0_1.yaml")
args = p.parse_args()
with open(args.config, encoding="utf-8") as f:
    raw = yaml.safe_load(f)
cfg = AyaEdgeConfig(**raw["model"])
model = AyaEdgeForCausalLM(cfg)
print(f"Trainable parameters: {count_parameters(model):,}")
print(f"Layers: {cfg.n_layers} | Experts/layer: {cfg.n_experts} | Top-k: {cfg.top_k}")
print("Prototype only: this is not yet the 310M target architecture.")
