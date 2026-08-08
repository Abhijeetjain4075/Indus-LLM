# Indus Architecture

Indus is a general-purpose foundation model specialized for **reasoning,
coding, scientific thinking, and autonomous problem solving**. This
document maps every architectural decision to the paper it came from.

---

## 1. Backbone: Decoder-only Transformer

**Reference:** Vaswani et al., 2017 — *Attention Is All You Need* (https://arxiv.org/abs/1706.03762)

We use the standard decoder-only transformer as in GPT. Each block is:

```
x = x + Attn(RMSNorm(x))         # pre-norm attention
x = x + FFN(RMSNorm(x))          # pre-norm FFN (SwiGLU or MoE)
```

Pre-norm (instead of post-norm) is more stable for deep stacks and is
the convention used by every frontier LLM since GPT-2.

---

## 2. Positional Encoding: RoPE

**Reference:** Su et al., 2021 — *RoFormer* (https://arxiv.org/abs/2104.09864)

We use **Rotary Position Embeddings** — Q and K vectors are rotated by
an angle that depends on their position, so the dot product `Q·K`
naturally encodes relative distance. This replaces learned absolute
position tables.

For very long contexts, we extend RoPE using **LongRoPE**
(Ding et al., 2024 — https://arxiv.org/abs/2402.13753) which
non-uniformly rescales per-dimension frequencies to extend context
length while preserving accuracy on shorter inputs.

**Why RoPE over learned/sinusoidal:**
- Length-generalizes better than learned positions
- No extra parameters
- Smooth extrapolation via frequency scaling

---

## 3. Attention: Grouped Query Attention (GQA)

**Reference:** Ainslie et al., 2023 — *GQA* (https://arxiv.org/abs/2305.13245)

Standard MHA: each query head has its own K/V head.
MQA (Multi-Query): all queries share one K/V head.
**GQA:** queries share K/V in groups. We default to `n_kv_head = n_head / 3`.

GQA gives ~95% of MHA quality at MQA's speed/memory. Used in Llama 2/3,
Qwen 2, Mistral, Gemma.

Memory-efficient attention backend (PyTorch SDPA → FlashAttention-2
when available):
- **FlashAttention:** Dao et al., 2022 (https://arxiv.org/abs/2205.14135)
- **FlashAttention-2:** Dao, 2023 (https://arxiv.org/abs/2307.08691)

We don't write a custom kernel — PyTorch's SDPA picks the best
backend (Flash-2, mem-eff, or math) automatically.

---

## 4. Normalization: RMSNorm

**Reference:** Zhang & Sennrich, 2019 — *RMSNorm* (https://arxiv.org/abs/1910.07467)

Faster than LayerNorm: no mean-centering, just rescale by the root
mean square. Numerically stable, ~10% faster, equivalent quality.
Used in Llama, Qwen, Gemma.

---

## 5. FFN: SwiGLU

**Reference:** Shazeer, 2020 — *GLU Variants Improve Transformer* (https://arxiv.org/abs/2002.05202)

```
y = down( silu(gate(x)) * up(x) )
```

The gated linear unit (GLU) consistently outperforms standard
ReLU/GELU FFNs. `SwiGLU` (SiLU/Swish as the gate) is the
best-performing variant in the paper and is the standard in Llama,
Qwen, Mistral.

We use `hidden_dim = 2/3 * 4 * n_embd`, rounded to the nearest 64
to match GPU-friendly alignments.

---

## 6. Optional: Sparse Mixture of Experts (MoE)

**References:**
- Shazeer et al., 2017 — *Outrageously Large Neural Networks* (https://arxiv.org/abs/1701.06538)
- Fedus et al., 2022 — *Switch Transformer* (https://arxiv.org/abs/2101.03961)
- Jiang et al., 2024 — *Mixtral of Experts* (https://arxiv.org/abs/2401.04088)

For models ≥1B params, the FFN can be replaced with a top-k
Mixture-of-Experts layer:

- A small router (linear `n_embd → n_experts`) picks the top-k experts
  per token.
- Each expert is a SwiGLU FFN.
- Only the chosen experts run on each token — total compute stays low
  while parameter count grows.

We include a **load-balancing auxiliary loss**
(`L_aux = num_experts * sum_i (tokens_per_expert_i * router_prob_i)`)
to prevent expert collapse. Coefficient: 0.01 (Switch Transformer's
default).

For 1.3T active / 8×7B total style: see `indus-moe-7b` config.

---

## 7. Training Pipeline

| Stage | Purpose | Method | Reference |
|---|---|---|---|
| Pre-training | Next-token prediction | Standard cross-entropy | — |
| SFT | Learn instruction format | Supervised fine-tuning on `(prompt, response)` | InstructGPT (https://arxiv.org/abs/2203.02155) |
| DPO | Align to human preferences | Direct Preference Optimization | Rafailov et al., 2023 (https://arxiv.org/abs/2305.18290) |
| Con. AI | Safety refinement | Constitutional AI | Bai et al., 2022 (https://arxiv.org/abs/2212.08073) |

For data quality, we use:
- **Self-Instruct** (Wang et al., 2022 — https://arxiv.org/abs/2212.10560) for bootstrapping
- **Evol-Instruct** (Xu et al., 2023 — https://arxiv.org/abs/2304.12244) for difficulty scaling
- **TinyStories** (Eldan & Li, 2023 — https://arxiv.org/abs/2305.07759) for small-model story data
- **Meta AutoData** (https://facebookresearch.github.io/RAM/blogs/autodata/) as the long-term target

---

## 8. Long Context

For long-context training and inference:
- **FlashAttention-2** (Dao, 2023) — tiled attention for O(N) memory
- **LongRoPE** (Ding et al., 2024 — https://arxiv.org/abs/2402.13753) — non-uniform position rescaling for >128K context

The `indus-moe-7b` config uses LongRoPE with `factor=4.0` to reach
131K context.

---

## 9. Training Efficiency

For distributed / efficient training, Indus is designed to plug into:

| Technique | Reference | Purpose |
|---|---|---|
| **LoRA** | Hu et al., 2021 (https://arxiv.org/abs/2106.09685) | Adapter fine-tuning — < 1% trainable params |
| **QLoRA** | Dettmers et al., 2023 (https://arxiv.org/abs/2305.14314) | 4-bit + LoRA, fits 65B on 1 GPU |
| **ZeRO** | Rajbhandari et al., 2019 (https://arxiv.org/abs/1910.02054) | Partition optimizer state across GPUs |
| **DeepSpeed** | Rasley et al., 2020 (https://arxiv.org/abs/2207.00032) | Full ZeRO + pipeline parallelism |

For very large community-driven training:
- **Petals** (Borzunov et al., 2022 — https://arxiv.org/abs/2209.01188) — swarm training across volunteer machines
- **Hivemind** (Ryabinin et al., 2021 — https://arxiv.org/abs/2103.03239) — distributed training over the internet

Indus Kernel includes a built-in **LoRA implementation** (see `kernel.py`).

---

## 10. Reasoning & Agent Layer

Indus is more than a chat model. On top of the base LM, the
**Indus Agent** runs a ReAct loop with built-in tools:

**References:**
- **ReAct** (Yao et al., 2022 — https://arxiv.org/abs/2210.03629): interleave Thought / Action / Observation
- **Reflexion** (Shinn et al., 2023 — https://arxiv.org/abs/2303.11366): verbal self-reflection on repeated failures
- **Tree of Thoughts** (Yao et al., 2023 — https://arxiv.org/abs/2305.10601): explore multiple reasoning branches
- **Chain-of-Thought** (Wei et al., 2022 — https://arxiv.org/abs/2201.11903): elicit step-by-step reasoning

```
Question
  ↓
Thought → Action → Observation → Thought → ... → Finish: <answer>
         ↑                                    ↓
         └──── Reflexion (if stuck) ←─────────┘
```

Built-in tools: `calculator`, `python`, `reverse_string`, `word_count`.
The registry is extensible — `ToolRegistry.register(...)` adds more.

For coding agents, the same loop with `python` tool approximates
OpenHands / Hermes (https://github.com/All-Hands-AI/OpenHands).

---

## 11. Continuous Learning: The Indus Kernel

The **Indus Kernel** is the API above the model. It exposes:

- **`add_adapter(name, r)`** — inject a fresh LoRA on top of frozen base weights
- **`train_adapter(name, pairs)`** — fine-tune the LoRA on a small dataset
- **`evaluate(text, tokenizer)`** — perplexity on a held-out text
- **`promote(adapter)`** — if the adapter is better, merge it into a new base

This is the mechanism that lets Indus **learn from new public
knowledge** between major versions, without retraining the base.

Reference: see also Krutrim (https://arxiv.org/abs/2502.09642) and
Airavata (https://arxiv.org/abs/2401.15006) for India-specific LLM
work in the same spirit.

---

## 12. Model Variants

| Variant | Params | Active | Context | Use case |
|---|---|---|---|---|
| `indus-tiny` | 1.1M | 1.1M | 128 | Demo / CPU test |
| `indus-small` | ~50M | 50M | 2K | Single GPU |
| `indus-base` | ~125M | 125M | 2K | Research baseline (GPT-2 small) |
| `indus-moe-1b` | ~1B (8 experts) | ~250M | 32K | Production small |
| `indus-moe-7b` | ~7B (8 experts) | ~860M | 131K | Production large |

---

## 13. References — full list

| Topic | Paper |
|---|---|
| Transformer | https://arxiv.org/abs/1706.03762 |
| RoPE | https://arxiv.org/abs/2104.09864 |
| SwiGLU | https://arxiv.org/abs/2002.05202 |
| RMSNorm | https://arxiv.org/abs/1910.07467 |
| GQA | https://arxiv.org/abs/2305.13245 |
| Sparsely-Gated MoE | https://arxiv.org/abs/1701.06538 |
| Switch Transformer | https://arxiv.org/abs/2101.03961 |
| Mixtral | https://arxiv.org/abs/2401.04088 |
| Self-Instruct | https://arxiv.org/abs/2212.10560 |
| Evol-Instruct | https://arxiv.org/abs/2304.12244 |
| Meta AutoData | https://facebookresearch.github.io/RAM/blogs/autodata/ |
| Chain-of-Thought | https://arxiv.org/abs/2201.11903 |
| Tree of Thoughts | https://arxiv.org/abs/2305.10601 |
| ReAct | https://arxiv.org/abs/2210.03629 |
| Reflexion | https://arxiv.org/abs/2303.11366 |
| InstructGPT | https://arxiv.org/abs/2203.02155 |
| DPO | https://arxiv.org/abs/2305.18290 |
| Constitutional AI | https://arxiv.org/abs/2212.08073 |
| FlashAttention | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 | https://arxiv.org/abs/2307.08691 |
| LongRoPE | https://arxiv.org/abs/2402.13753 |
| LoRA | https://arxiv.org/abs/2106.09685 |
| QLoRA | https://arxiv.org/abs/2305.14314 |
| ZeRO | https://arxiv.org/abs/1910.02054 |
| DeepSpeed | https://arxiv.org/abs/2207.00032 |
| TinyStories | https://arxiv.org/abs/2305.07759 |
| Petals | https://arxiv.org/abs/2209.01188 |
| Hivemind | https://arxiv.org/abs/2103.03239 |
| nanoGPT | https://github.com/karpathy/nanoGPT |
| LLMs-from-scratch | https://github.com/rasbt/LLMs-from-scratch |
| OpenHands | https://github.com/All-Hands-AI/OpenHands |
| Hermes Agent | https://github.com/NousResearch/Hermes-Agent |
| Krutrim | https://arxiv.org/abs/2502.09642 |
| Airavata | https://arxiv.org/abs/2401.15006 |
| Paramanu | https://arxiv.org/abs/2401.18034 |
| Paramanu-Ayn | https://arxiv.org/abs/2403.13681 |
