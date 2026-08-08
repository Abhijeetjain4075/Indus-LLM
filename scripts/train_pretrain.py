"""Pre-train an Indus-tiny model from scratch on Shakespeare.

This is a small demo (sandbox-safe CPU run). For real pre-training,
bump the config in INDUS_CONFIGS to 'indus-small' / 'indus-base' and
use a real corpus + GPU.
"""
import sys
sys.path.insert(0, "/workspace/indus")

from indus.train import pretrain, TrainConfig


def main():
    cfg = TrainConfig(
        # data
        data_path="data/tinyshakespeare.txt",
        dataset="char",
        block_size=128,

        # model — sandbox-safe (~1.1M params)
        n_layer=6,
        n_head=6,
        n_kv_head=2,
        n_embd=156,
        dropout=0.1,
        ffn_kind="swiglu",

        # optimization
        batch_size=24,
        grad_accum_steps=1,
        max_iters=1500,
        lr=5e-4,
        warmup_iters=50,
        weight_decay=0.1,

        # logging
        log_every=20,
        eval_every=100,
        save_every=500,
        out_dir="checkpoints/pretrain",
    )
    pretrain(cfg)


if __name__ == "__main__":
    main()
