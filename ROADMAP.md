# Indus Roadmap

This is the honest, technical plan for scaling Indus from where it is
now to a real production-grade technical LLM. Every step requires real
compute and real engineering — there are no shortcuts.

## Current state (v0.3.0)

| Item | Value | How verified |
|---|---|---|
| Architecture | RoPE + GQA + SwiGLU + RMSNorm + (optional MoE) | Code review |
| Parameter count | 1.12M (indus-tiny) | `model.num_params()` |
| Training data | 1 MB Shakespeare (char-level) | `data/tinyshakespeare.txt` |
| Training tokens | ~1.1M | Derived from text length × epoch |
| Final val loss | 1.47 | `checkpoints/pretrain/indus_final.pt` + `training.log` |
| Final val perplexity | 4.33 | `checkpoints/eval/indus-tiny-v1.json` |
| Training time | 17 minutes on 1 CPU | Wall clock |
| Code completion | n/a (vocab mismatch) | `checkpoints/eval/indus-tiny-v1.json` |
| Math reasoning | n/a (vocab mismatch) | `checkpoints/eval/indus-tiny-v1.json` |

The vocab mismatch is expected — a char-level Shakespeare model has no
`[`, `]`, `(`, `)`, `%`, etc. Switching to a BPE tokenizer and training
on real text fixes this. The code paths are already there
(`dataset="bpe"`).

## Honest assessment

What 1.12M params at val loss 1.47 actually means:

- The model has learned character-level Shakespeare structure
- It produces character names as speakers (ROMEO:, JULIET:, etc.)
- It produces archaic English vocabulary with some grammatical coherence
- It does NOT do math, code, factual reasoning, or anything beyond
  imitating Shakespeare
- It is NOT comparable to any modern LLM on any standard benchmark

This is a **research/architecture prototype**, not a product. The
purpose is to verify the architecture, the training pipeline, the eval
harness, and the engineering scaffolding all work end-to-end before
investing in larger runs.

## Scaling ladder

| Step | Params | Active | Context | Token budget | Compute | Expected time |
|---|---|---|---|---|---|---|
| 0 (done) | 1.12M | 1.12M | 128 | 1.1M | 1 CPU | 17 min ✓ |
| 1 | 10M | 10M | 512 | 50M | 1× T4 | 4 hours |
| 2 | 50M | 50M | 1024 | 250M | 1× A100 | 1 day |
| 3 | 120M | 120M | 2048 | 600M | 1× A100 | 4 days |
| 4 | 350M | 350M | 4096 | 5B | 4× A100 | 1 week |
| 5 | 1B (MoE) | ~250M | 32K | 30B | 8× H100 | 3 weeks |
| 6 | 7B (MoE) | ~860M | 131K | 500B | 64× H100 | 6 weeks |
| 7 | 70B+ (MoE) | ~5B | 131K+ | 5T | 256+ H100 | months |

Each step is a real, separate project. Do not skip from step 0 to step 6.

## What needs to happen between steps

### 1. Real data pipeline (steps 1+)

The current `data_pipeline.py` is a scaffold. For real training:

- Ingest from a real source (FineWeb-Edu, SlimPajama, your own corpus)
- Run language ID (fastText langdetect)
- Quality scoring (a real model, not the heuristic)
- MinHash LSH for dedup (use `datasketch` not the shingle impl)
- License/provenance filter (must be auditable)
- Train/val/test split with **strict** no-leakage
- Maintain an exclusion list of benchmark contamination

### 2. Real tokenizer (step 1+)

Char-level is the toy. BPE is the standard:

- Use `tiktoken` (GPT-2 / cl100k_base / o200k) for English
- Or train a custom BPE for code-heavy corpora (code-specific vocab)
- For Indian languages: sentencepiece with a 32K-64K vocab

### 3. Real distributed training (step 5+)

Single GPU won't cut it past ~350M params. The codebase is ready for:

- `torch.distributed` (FSDP / DDP)
- DeepSpeed ZeRO-2/3
- Petals for community-distributed training
- Gradient checkpointing for memory
- Mixed precision (bf16 on H100, fp16 on A100)

### 4. Real evaluation (step 1+)

The current eval is a smoke test. For real claims, you need:

- `lm-evaluation-harness` for standardized benchmarks
- HumanEval / MBPP for code
- MMLU / ARC for knowledge
- GSM8K / MATH for math
- TruthfulQA for factuality
- Run with multiple seeds, report mean ± std
- Compare only with same eval conditions

### 5. Real alignment (step 3+)

SFT and DPO are implemented but need real data:

- SFT: 50K-500K (prompt, response) pairs from Alpaca / ShareGPT / OpenHermes
- DPO: 50K+ (prompt, chosen, rejected) triples from UltraFeedback
- ConAI: principled critique-revise on your model's own outputs
- Iterative: model A → critique → revise → train on revisions → model B

### 6. Real safety (always)

Before any deployment:

- Red-team evaluation
- Refusal behavior verification
- Bias / toxicity benchmarks
- Capability elicitation (make sure dangerous capabilities are not latent)
- External review

## When NOT to scale

Do not scale if any of the following are true:

- The eval harness doesn't reproduce on a clean clone
- The training loop doesn't survive session termination
- The data pipeline allows benchmark contamination
- The experiment registry doesn't capture the full hypothesis/change/result
- The checkpoint integrity check fails
- You cannot describe what the new model does better than the old one

A bigger model that you can't evaluate is worse than a smaller one
that you can.

## TL;DR

Indus-tiny is a working prototype that proves the architecture. The
next 1B+ steps require real compute, real data, real engineering time,
and a real team. The code is ready for the engineering; the rest is
unforgiving physics.
