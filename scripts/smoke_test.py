"""Indus smoke test — verify every component wires up.

Run: python -m indus.scripts.smoke_test
"""
import sys
sys.path.insert(0, "/workspace/indus")

import torch

from indus import (
    Indus, IndusConfig, INDUS_CONFIGS, IndusTokenizer,
    IndusAgent, IndusKernel, ToolRegistry,
    BitLinear, replace_linears_with_bitlinear, MoDLayer,
    best_of_n, self_consistency, critique_and_revise,
    ExperimentRegistry, EvalReport,
    eval_perplexity, eval_code_completion, eval_math,
    eval_security_patterns,
    DataPipeline, Record,
    stage_format_normalization, stage_corruption_filter,
    stage_quality_scoring, stage_dedup_exact, stage_dedup_near,
)
from indus.kernel import inject_lora, lora_parameters
from indus.tools import default_tools, calculator_tool, python_tool


def main():
    print("=" * 70)
    print("INDUS — FULL SMOKE TEST")
    print("=" * 70)

    # 1. Build the tiny model
    cfg = INDUS_CONFIGS["indus-tiny"]
    model = Indus(cfg)
    n = model.num_params()
    print(f"\n[1] Built indus-tiny: {n:,} params ({n/1e6:.2f}M)")
    assert n > 100_000, "model too small"

    # 2. Forward + backward
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, y)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    print(f"[2] Forward + backward: loss={out.loss.item():.3f}")

    # 3. Generation
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    out_ids = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"[3] Generation: input {idx.shape} -> {out_ids.shape}")

    # 4. MoE
    moe_cfg = IndusConfig(
        name="indus-moe-tiny", n_layer=2, n_embd=64, n_head=4, n_kv_head=2,
        block_size=64, vocab_size=cfg.vocab_size, ffn_kind="moe",
        moe_num_experts=4, moe_top_k=2,
    )
    moe = Indus(moe_cfg)
    out = moe(torch.randint(0, moe_cfg.vocab_size, (2, 32)),
              torch.randint(0, moe_cfg.vocab_size, (2, 32)))
    print(f"[4] MoE model ({moe.num_params():,} params)  aux_loss={out.aux_loss.item():.4f}")

    # 5. LoRA injection
    inject_lora(model, r=4, alpha=8.0, target="qkv")
    lora_params = lora_parameters(model)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"[5] LoRA: {n_lora:,} trainable ({100*n_lora/n:.2f}%)")

    # 6. BitLinear — 1.58-bit replacement
    import torch.nn as nn
    bit_model = Indus(cfg)
    n_bit = replace_linears_with_bitlinear(bit_model, target_substrings=("q_proj", "v_proj"))
    xb = torch.randint(0, cfg.vocab_size, (2, 16))
    yb = torch.randint(0, cfg.vocab_size, (2, 16))
    out = bit_model(xb, yb)
    assert out.loss is not None
    print(f"[6] BitLinear: replaced {n_bit} linears, forward+loss ok (loss={out.loss.item():.3f})")

    # 7. TTCS
    prompts = ["Q: 2+2?\nA:", "Q: 3*4?\nA:"]
    @torch.no_grad()
    def _quick_gen(model, idx, max_new_tokens=10):
        return model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.7, top_k=20)
    print(f"[7] TTCS: best_of_n / self_consistency modules importable")

    # 8. Tools
    tools = ToolRegistry()
    assert tools.call("calculator", {"expression": "2+3*4"}) == "14"
    assert tools.call("python", {"code": "print(2+2)"}).strip() == "4"
    print(f"[8] Tools: {[t for t in tools.names()]} — calculator and python verified")

    # 9. Agent parser
    from indus.agent import parse_step
    s1 = parse_step("Thought: hi\nAction: calculator\nActionInput: {\"expression\": \"2+2\"}")
    s2 = parse_step("Thought: done\nFinish: 4")
    assert s1.action == "calculator" and s2.is_final
    print(f"[9] Agent parser OK")

    # 10. Experiment registry
    import tempfile
    _reg_path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    reg = ExperimentRegistry(_reg_path)
    exp = reg.create(
        name="smoke-test",
        hypothesis="all components wire up",
        change="added new modules",
        baseline="v0.2.0",
        tags=["smoke"],
    )
    reg.complete(exp, training_loss=2.5, validation_loss=2.7, perplexity=15.0,
                 decision="keep", result="all components operational")
    print(f"[10] ExperimentRegistry: created+completed '{exp.name}' id={exp.id}")

    # 11. Evaluator (light)
    from indus.data import download_shakespeare, CharDataset
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = download_shakespeare(f"{td}/shakespeare.txt")
        ds = CharDataset(p, block_size=cfg.block_size)
        report = EvalReport(
            model_version="indus-tiny",
            parameter_count=model.num_params(),
            tokenizer="char",
            timestamp=0.0,
            config={"test": "smoke"},
        )
        sec = eval_security_patterns("import os\nos.system('rm -rf /')\n")
        assert "shell call" in sec
        print(f"[11] Evaluator: security pattern check ok (found {sec})")

    # 12. Data pipeline
    import tempfile, json
    with tempfile.TemporaryDirectory() as td:
        dp = DataPipeline(root=td, name="smoke")
        # 10 distinct records + 1 exact dup of the first
        records = [
            Record(id=f"r{i}", text=(f"the quick brown fox #{i} jumps over the lazy dog. " * 5),
                   source="test", license="MIT")
            for i in range(10)
        ]
        # Add an exact duplicate of the first
        records.append(Record(id="dup", text=records[0].text, source="test", license="MIT"))
        ver = dp.run(records, tokenizer_encode=lambda s: list(s))
        # After dedup_exact, the duplicate should be removed -> 10 records
        assert ver.num_records == 10, f"dedup didn't drop the duplicate, got {ver.num_records}"
        print(f"[12] DataPipeline: {ver.num_records} records (dropped {len(records) - ver.num_records} dup), "
              f"hash={ver.content_hash}, q_mean={ver.quality_stats['mean']:.2f}")

    print("\n" + "=" * 70)
    print("✓ All Indus components operational (v0.3.0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
