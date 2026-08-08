"""Run the evaluation suite on a trained Indus checkpoint.

Outputs:
  - Console summary
  - JSON report at checkpoints/eval/<model_version>.json
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, "/workspace/indus")

from indus.generate import load_indus
from indus.evaluator import run_full_eval


def main(ckpt: str, model_version: str = "indus-tiny-v1"):
    print("=" * 70)
    print(f"INDUS EVAL — {model_version}")
    print("=" * 70)
    model, tokenizer, ck = load_indus(ckpt)
    print(f"Loaded {model.num_params()/1e6:.2f}M params, vocab={tokenizer.vocab_size}, iters={ck.get('iter', '?')}")

    report = run_full_eval(
        model, tokenizer,
        model_version=model_version,
        eval_name="full",
        heldout_text=(
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune,\n"
            "Or to take arms against a sea of troubles\n"
            "And by opposing end them."
        ),
    )
    print()
    print(report.summary())

    out_dir = Path("checkpoints/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_version}.json"
    report.save(str(out_file))
    print(f"\nReport saved: {out_file}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pretrain/indus_final.pt"
    main(ckpt)
