# Indus

> *A general-purpose foundation model specialized for reasoning, coding,
> scientific thinking, and autonomous problem solving. It is not just a
> chatbot.*

Indus is a from-scratch LLM in PyTorch with the full modern stack:
**RoPE + GQA + SwiGLU + RMSNorm + (optional) MoE**, plus an
**agent loop (ReAct + Reflexion)**, a **continuous-learning Kernel** with
**LoRA adapters**, and training pipelines for **pre-training, SFT, and DPO**.

The architecture is grounded in ~30 research papers — see
[ARCHITECTURE.md](ARCHITECTURE.md) for the full mapping.

## Quick start

```bash
# 1. Verify everything wires up
python scripts/smoke_test.py

# 2. Pre-train a tiny Indus from scratch (~15 min on CPU)
python scripts/train_pretrain.py

# 3. Run the agent on a few tasks
python scripts/run_agent.py

# 4. SFT a tiny instruction-tuned variant
python scripts/train_sft.py

# 5. DPO alignment
python scripts/train_dpo.py
```

## What's in here

| Path | What it does |
|---|---|
| `indus/config.py` | Model configs: `indus-tiny`, `indus-small`, `indus-base`, `indus-moe-1b`, `indus-moe-7b` |
| `indus/model.py` | The transformer: `Indus`, `IndusBlock` |
| `indus/rope.py` | Rotary position embeddings (+ LongRoPE scaling) |
| `indus/norms.py` | RMSNorm |
| `indus/attention.py` | Grouped Query Attention with PyTorch SDPA |
| `indus/moe.py` | Sparse top-k Mixture of Experts + load-balancing aux loss |
| `indus/tokenizer.py` | BPE wrapper (tiktoken) |
| `indus/data.py` | Pre-training, SFT, DPO datasets |
| `indus/train.py` | Unified trainer (pretrain / SFT / DPO) |
| `indus/generate.py` | Sampling with temperature / top-k / top-p / EOS |
| `indus/kernel.py` | **Indus Kernel** — LoRA + DataStore + Evaluator + Promoter |
| `indus/agent.py` | **Indus Agent** — ReAct + Reflexion |
| `indus/tools.py` | Built-in tool registry (calculator, python, …) |
| `scripts/` | End-to-end training & demo scripts |
| `ARCHITECTURE.md` | Every design choice → paper |
| `KERNEL.md` | Continuous-learning API |
| `AGENT.md` | Agent loop & tool design |

## Architecture at a glance

```
Input ids [B, T]
     │
     ▼
┌──────────────────────┐
│ Token embedding      │
│ Dropout              │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ IndusBlock × N       │ × N layers
│   RMSNorm            │
│   GQA + RoPE         │  ← QK have rotary positions, GQA reduces KV heads
│   RMSNorm            │
│   SwiGLU  or  MoE    │  ← SwiGLU by default, MoE for large variants
└──────────────────────┘
     │
     ▼
RMSNorm → LM head (tied) → logits
```

For the full mapping to the underlying papers, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Why this design

**Coding, reasoning, research, planning, creation** all share a
common backbone: a strong language model. The unique requirements are:

| Capability | Architectural choice |
|---|---|
| Coding | Long context + strong reasoning; GQA + RoPE |
| Reasoning | MoE for parameter capacity, then CoT/ReAct at inference |
| Research | Long context (LongRoPE) + tool use for retrieval |
| Learning | LoRA + DataStore in the Kernel for continuous updates |
| Agents | ReAct + Reflexion loop with extensible tool registry |

We do **not** use a single monolithic prompt. Each capability is
implemented as a separate layer that can be scaled independently.

## Scaling path

| Stage | Compute | Token budget | Expected time |
|---|---|---|---|
| Demo (this repo) | 1 CPU | 1 MB | 15 min |
| `indus-small` (50M) | 1× A100 | 5 GB | 6 hours |
| `indus-base` (125M) | 1× A100 | 13 GB | 1 day |
| `indus-moe-1b` | 8× H100 | 50 GB | 3 days |
| `indus-moe-7b` | 64× H100 | 500 GB | 1 month |

For larger runs, integrate with **DeepSpeed** (ZeRO) and **Petals**
for community-distributed training.

## License

MIT. Use, fork, extend, ship.
