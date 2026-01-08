# toygpt

A PyTorch implementation of the **Manifold-Constrained Hyper-Connection Expert Transformer (mHC-Expert)**, featuring architectural stability enhancements and Mixture-of-Experts (MoE) efficiency. This is a research codebase for training and interacting with transformer models that use hyper-connections constrained to the doubly stochastic manifold.

## Overview

The model combines:

- MoE layers with shared SwiGLU experts
- RoPE positional embeddings for efficient context extension
- Depth gating for dynamic layer contribution scaling
- Sinkhorn–Knopp projections for optimal routing (FP32-stable)

The codebase is designed for training language models on textual data and includes a full-featured chat interface with KV-caching and adaptive temperature sampling.

## Quick Start

### 1. Create a Base Model

Run the model definition script to generate a fresh transformer:

```bash
python model_mhc_expert.py
```

This creates `mhc_stable_transformer.pt` with:

- 8 layers, 8 heads, hidden dim 512
- 8 SwiGLU experts
- 4× stream expansion
- Vocabulary: GPT-2 tokenizer (50,257 tokens)

### 2. Train on Text Data

Launch the training GUI to select model and training text:

```bash
python train_expert_act.py
```

**Training Configuration (defaults):**

- Total steps: 50,000
- Warmup: 2,000 steps
- Peak LR: 3e-4 → Min LR: 1e-5
- Batch size: 8 (effective: 32 with gradient accumulation)
- Sequence length: 256 tokens
- Mixed precision: BF16 (auto-detected)
- Loss penalties:
  - λ_lb = 0.005
  - λ_ent_moe = 0.005
  - λ_ent_attn = 0.002
- Validation split: 10%
- Early stopping: Patience = 10 (active after 5,000 steps)

**Outputs:**

- `*_checkpoint.pt`: Periodic saves every 500 steps
- `*_best.pt`: Best validation-loss model
- `*_trained.pt`: Final model after completion

### 3. Chat with Your Model

Launch the interactive chat interface:

```bash
python chat_expert_act.py
```

**Chat Commands:**

- `/dynamic` — Enable dynamic temperature (default)
- `/fixed` — Use fixed temperature
- `/temp <value>` — Set base temperature (default: 0.8)
- `/stats` — Toggle generation statistics
- `/verbose` — Show per-token confidence/temperature
- `quit` / `exit` — Exit chat

**Generation Features:**

- KV cache for O(1) per-token complexity
- Adaptive temperature based on token entropy
- Top-k = 50, top-p = 0.95 nucleus sampling
- Per-token confidence and entropy tracking

**Memory Usage:** ~1.5 GB BF16 for base configuration (8 layers, 512 dim)

