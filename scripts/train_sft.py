"""SFT a pre-trained Indus on a small instruction set.

This is a tiny demo. For real instruction-tuning, use a corpus of
~100K+ (prompt, response) pairs.
"""
import sys
sys.path.insert(0, "/workspace/indus")

from indus.train import sft, TrainConfig


def main():
    cfg = TrainConfig(
        out_dir="checkpoints/sft",
        dataset="sft",
        block_size=256,
        n_layer=6, n_head=6, n_kv_head=2, n_embd=156, dropout=0.0,
        ffn_kind="swiglu",
        batch_size=4, grad_accum_steps=1,
        max_iters=200, lr=1e-4, warmup_iters=10,
        log_every=10, save_every=200,
    )
    sft(cfg)


if __name__ == "__main__":
    main()
