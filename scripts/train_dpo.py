"""DPO a SFT'd Indus on preference triples.

Demonstrates the alignment step. For real RLHF, scale up the dataset
and use a stronger preference model.
"""
import sys
sys.path.insert(0, "/workspace/indus")

from indus.train import dpo, TrainConfig


def main():
    cfg = TrainConfig(
        out_dir="checkpoints/dpo",
        dataset="dpo",
        block_size=256,
        n_layer=6, n_head=6, n_kv_head=2, n_embd=156, dropout=0.0,
        ffn_kind="swiglu",
        batch_size=2, grad_accum_steps=1,
        max_iters=100, lr=5e-5, warmup_iters=10,
        dpo_beta=0.1, log_every=10, save_every=100,
    )
    dpo(cfg)


if __name__ == "__main__":
    main()
