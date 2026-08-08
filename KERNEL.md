# The Indus Kernel

The **Indus Kernel** is the continuous-learning layer that sits above
the model. It's the mechanism that lets Indus keep improving without
throwing away the base every time.

```
           ┌─────────────────────────────────────────┐
           │              Indus Kernel               │
           │  ┌─────────────┐    ┌────────────────┐  │
           │  │ DataStore    │    │ LoRA Adapters  │  │
           │  │ (versioned)  │    │ (≤1% params)   │  │
           │  └──────┬──────┘    └────────┬───────┘  │
           │         │                    │          │
           │         ▼                    ▼          │
           │  ┌──────────────────────────────────┐   │
           │  │     Indus base model (frozen)    │   │
           │  └──────────────────────────────────┘   │
           │         │                    ▲          │
           │         ▼                    │          │
           │  ┌─────────────┐    ┌────────┴──────┐  │
           │  │ Evaluator   │    │  Promoter     │  │
           │  │ (ppl, acc)  │───▶│ (merge if ↑)  │  │
           │  └─────────────┘    └───────────────┘  │
           └─────────────────────────────────────────┘
```

## The API

```python
from indus import IndusKernel, Indus, IndusTokenizer

# Load a base model
model, tokenizer, _ = load_indus("checkpoints/indus_final.pt")

# Wrap it in the kernel
kernel = IndusKernel(model)

# 1. Add a LoRA adapter (only ~2% of params become trainable)
kernel.add_adapter("math-v1", r=8)

# 2. Train the adapter on a small dataset
kernel.train_adapter(
    "math-v1",
    pairs=[
        ("What is 12 * 7?", "12 * 7 = 84."),
        ("What is sqrt(144)?", "sqrt(144) = 12."),
        ...
    ],
    iters=200,
)

# 3. Evaluate
ppl_before = 4.2  # from previous run
ppl_after = kernel.evaluate("Q: what is 2+2? A: 4", tokenizer)

# 4. Promote if better
if ppl_after < ppl_before:
    kernel.promote("math-v1")
```

## What's inside

### DataStore — versioned data

```python
store = DataStore("data_store")
v = store.add_version("math-qa", "raw_math.jsonl", metadata={"source": "gsm8k"})
store.latest("math-qa")     # returns the newest version
store.list()                # all versions across all datasets
```

Every version is **immutable**. To "update" a dataset, you add a new
version — old versions stay around for reproducibility and roll-back.

### LoRA — parameter-efficient updates

```python
from indus.kernel import inject_lora, lora_parameters

inject_lora(model, r=8, alpha=16.0, target="qkv")
# Now only LoRA params are trainable
trainable = lora_parameters(model)
optim = torch.optim.AdamW(trainable, lr=1e-4)
```

Default targets: the Q/K/V/O projections in every attention block.
You can extend to all linears via `target="all"`.

### Evaluator — measure improvement

```python
from indus.kernel import Evaluator

ev = Evaluator(model)
ppl = ev.perplexity(held_out_text, tokenizer)
```

Replace the simple perplexity with a real benchmark suite
(e.g. `lm-evaluation-harness`) by subclassing `Evaluator` and
overriding `__call__`.

### Promoter — safe rollout

`kernel.promote(adapter)` writes the current model state (with
adapter merged) to a new checkpoint. In a production system, this
would:
1. Hold the new candidate in a shadow slot
2. Route 5% of traffic to it
3. Compare metrics for 24h
4. Promote or roll back

The API stays the same.

## Why this design

| Concern | Why the Kernel is built this way |
|---|---|
| **Safety** | Base model is frozen; only LoRA trains. Easy to roll back. |
| **Speed** | LoRA = < 1% trainable params. Fine-tunes in minutes, not days. |
| **Versioning** | Every dataset and adapter is content-hashed and dated. |
| **Composability** | Multiple adapters can be stacked. Swap any of them. |
| **Evaluation** | Promotion is gated on held-out metrics, not vibes. |

## What this enables

- **Domain adaptation.** Train an adapter on your private corpus in
  minutes on a single GPU.
- **Skill additions.** A new "math" adapter doesn't touch "code".
- **Rollback.** Bad adapter? Just don't promote it.
- **A/B testing.** Run two adapters side-by-side, compare metrics.
- **Continuous improvement.** Feed new public knowledge through the
  same pipeline.

## When to bypass the Kernel

For major architectural changes (e.g. context length extension,
adding MoE, switching tokenizer), bypass the Kernel and do a full
pre-training run. The Kernel is for *incremental* improvement, not
*fundamental* change.
