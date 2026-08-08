# Indus — Final Status (v0.3.0)

This is the honest status of the Indus LLM project as it stands.
It is **not** a frontier model. It is a working, from-scratch
implementation of a modern LLM architecture, trained to the limits
of a 2GB-RAM CPU sandbox, with full engineering scaffolding around
it for real production use.

---

## What is real, verified, and runs end-to-end

### The model
- **1,150,188 parameters** (`scripts/smoke_test.py` → `[1]`)
- Trained from scratch on Shakespeare (1 MB, char-level)
- **Final validation loss: 1.47** (perplexity **4.33**)
- Generates coherent Shakespeare-style text with character names
  (see `SAMPLES.md` for raw output)
- Architecture: RoPE + GQA + SwiGLU + RMSNorm + (optional) MoE
- Code: `indus/model.py`, `indus/attention.py`, `indus/rope.py`,
  `indus/norms.py`, `indus/moe.py`, `indus/bitnet.py`, `indus/mod.py`

### The training pipeline
- AdamW + linear warmup + cosine decay + grad clipping
- bf16/fp16 mixed precision (auto-detects GPU)
- Sliding-window datasets for pretraining
- SFT with response-only loss masking
- DPO with reference model + sigmoid preference loss
- Code: `indus/train.py`

### The agent framework
- ReAct loop (Thought / Action / Observation / Finish)
- Reflexion: verbal self-reflection on repeated failures
- Tool registry with `calculator`, `python`, `reverse_string`,
  `word_count` — extensible via `ToolRegistry.register(...)`
- Safe-AST evaluator for the calculator; sandbox for the Python tool
- Code: `indus/agent.py`, `indus/tools.py`

### The continuous-learning kernel
- LoRA adapter injection (parameter-efficient fine-tuning, < 1%
  of base params trainable)
- Versioned data store with immutable snapshots
- Evaluator with perplexity + comparison vs reference
- Code: `indus/kernel.py`

### The research components
- **BitLinear** (BitNet b1.58): 1.58-bit weight quantization
  with STE for training
- **Mixture of Depths**: dynamic per-token layer skipping for
  compute efficiency
- **Test-time compute scaling**: Best-of-N, Self-Consistency,
  Verifier-guided search
- **Constitutional AI**: principle-based self-critique and revision
- Code: `indus/bitnet.py`, `indus/mod.py`, `indus/ttcs.py`,
  `indus/constitution.py`

### The engineering infrastructure
- **Experiment registry**: hypothesis/change/baseline/result/decision
  tracking with environment snapshot for reproducibility
- **Data pipeline scaffold**: source validation → license check →
  format normalization → corruption filter → quality scoring →
  exact dedup → near-dup dedup (MinHash-ready) → train/val/test
  split → tokenization → versioned dataset
- **Evaluation harness**: perplexity, code completion, math
  exact-match, security pattern scan — with JSON report output
- **Smoke test**: 12 checks covering every component
  (`scripts/smoke_test.py`)

### Documentation
- `README.md` — overview, quick start
- `ARCHITECTURE.md` — every design decision mapped to its paper
- `KERNEL.md` — continuous-learning API
- `AGENT.md` — agent loop and tool design
- `ROADMAP.md` — honest scaling ladder
- `SAFETY.md` — safety process + pre-deployment checklist
- `SAMPLES.md` — raw output from the trained model
- `STATUS.md` — this file

---

## Verified numbers

| Metric | Value | Source |
|---|---|---|
| Parameter count | 1,150,188 | `scripts/smoke_test.py` |
| Vocab size | 65 chars | `checkpoints/pretrain/indus_final.pt` |
| Context length | 128 tokens | `TrainConfig.block_size` |
| Final val loss | 1.47 | training log |
| Final val perplexity | 4.33 | `checkpoints/eval/indus-tiny-v1.json` |
| Training time | ~17 min on 1 CPU | wall clock |
| Training tokens | ~1.1M (1 epoch) | derived |
| Smoketest | 12/12 pass | `scripts/smoke_test.py` |

---

## What is NOT real and would be a lie to claim

- **Indus does not rank on any standard benchmark.** It is too small.
- **Indus does not beat Claude, GPT, Qwen, or any frontier model.** It
  cannot. The training compute difference is 6+ orders of magnitude.
- **Indus is not safe for production use.** The `SAFETY.md` checklist
  is unfinished.
- **No code was pushed to any GitHub repo.** I never had credentials,
  and refused to use the ones that were pasted in chat.
- **No training was run on Kaggle.** I never had credentials, and
  refused to use the ones that were pasted in chat.

---

## How to actually scale this

The codebase is designed for the following pipeline:

1. **Get real compute.** A single A100 (~$1/hr) for 1B-class, 8× H100
   for 7B, 64+ H100 for 70B+.

2. **Get real data.** FineWeb-Edu (1.3T tokens, free) or a curated
   technical corpus. The data pipeline scaffold is ready to receive
   it. Run license/provenance/dedup/quality filters.

3. **Switch to BPE tokenization.** Set `dataset="bpe"` in
   `TrainConfig`. The code path is already there.

4. **Use the experiment registry for every change.** Hypothesis,
   change, baseline, result, decision. Always.

5. **Eval with `lm-evaluation-harness`.** Standardized. Cite
   conditions.

6. **SFT + DPO** on instruction + preference data before any
   user-facing deployment.

7. **External safety review** before any real release.

8. **Iterate.** A 1B model that beats 1.5B on technical benchmarks
   is more useful than a 7B model you can't evaluate.

---

## File tree

```
/workspace/indus/
├── indus/                        # the package
│   ├── __init__.py
│   ├── config.py                 # model configs (tiny → 7B)
│   ├── model.py                  # the transformer
│   ├── attention.py              # GQA
│   ├── rope.py                   # RoPE + LongRoPE
│   ├── norms.py                  # RMSNorm
│   ├── moe.py                    # Mixture of Experts
│   ├── bitnet.py                 # BitLinear (1.58-bit)
│   ├── mod.py                    # Mixture of Depths
│   ├── ttcs.py                   # Test-time compute
│   ├── constitution.py           # Con AI
│   ├── tokenizer.py              # BPE + char
│   ├── data.py                   # datasets
│   ├── data_pipeline.py          # versioned, dedup, quality
│   ├── train.py                  # pretrain / SFT / DPO
│   ├── generate.py               # sampling
│   ├── kernel.py                 # continuous learning
│   ├── agent.py                  # ReAct + Reflexion
│   ├── tools.py                  # tool registry
│   ├── experiments.py            # experiment registry
│   └── evaluator.py              # eval harness
├── scripts/
│   ├── smoke_test.py             # 12 checks
│   ├── train_pretrain.py         # 5M-class pretraining
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── run_eval.py               # run eval, save JSON
│   ├── run_agent.py              # ReAct + Reflexion demo
│   └── gen_samples.py            # sample + save SAMPLES.md
├── checkpoints/
│   ├── pretrain/indus_final.pt   # 1.12M trained model
│   └── eval/indus-tiny-v1.json   # eval report
├── data/                         # training data (gitignored)
├── README.md
├── ARCHITECTURE.md
├── KERNEL.md
├── AGENT.md
├── ROADMAP.md
├── SAFETY.md
├── STATUS.md
├── SAMPLES.md
├── requirements.txt
└── .gitignore
```

Total: 3,637 lines of Python across the `indus/` package, ~700
lines of scripts, ~1,500 lines of documentation.
