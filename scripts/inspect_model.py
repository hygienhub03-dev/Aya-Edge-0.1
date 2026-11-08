import argparse
import yaml
from aya_edge import AyaEdgeConfig, AyaEdgeForCausalLM
from aya_edge.model import count_active_parameters, count_parameters

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/aya_edge_0_1.yaml")
args = p.parse_args()
with open(args.config, encoding="utf-8") as f:
    raw = yaml.safe_load(f)
cfg = AyaEdgeConfig(**raw["model"])
model = AyaEdgeForCausalLM(cfg)
print(f"Trainable parameters: {count_parameters(model):,}")
print(f"Structural active-parameter estimate: {count_active_parameters(model):,}")
print(f"Layers: {cfg.n_layers} | Experts/layer: {cfg.n_experts} | Top-k: {cfg.top_k}")
if count_parameters(model) < 300_000_000:
    print("Prototype only: this is the approximately 85.1M baseline, not the 0.2 candidate.")
else:
    print("Research candidate only: parameter target reached, but this is not production training or a quality claim.")
