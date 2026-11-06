
# AYA Edge 0.1

An experimental compact, agent-specialised sparse Mixture-of-Experts language model.

## Project status

**Phase:** Architecture scaffold

AYA Edge 0.1 explores a compact architecture combining:

- MLA-inspired latent attention
- Fine-grained sparse Mixture-of-Experts (MoE)
- Top-k expert routing
- Shared expert layers
- Instruction following and agentic tool-use capabilities

## Development approach

The project will progress through:

1. Architecture design and validation
2. Pretraining
3. Mid-training
4. Supervised fine-tuning (SFT)
5. Reasoning and agentic alignment

## Teacher model

MiniCPM5-1B is the initial teacher model for experimentation and knowledge distillation.

## Important

This is an experimental research project. The initial architecture is a prototype and has not yet reached its intended 310M parameter target.

See upcoming architecture documentation for implementation details.
