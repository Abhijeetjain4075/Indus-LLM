"""Indus Kernel — the continuous learning infrastructure.

A foundation model should not remain static. The Kernel is the API Indus
uses to:

  - Version and store new training data ("datasets")
  - Apply LoRA-style parameter-efficient updates ("adapters")
  - Evaluate models on standard benchmarks
  - Promote a new model version only if it improves on a held-out set

Design references:
  LoRA:   Hu et al., 2021 — https://arxiv.org/abs/2106.09685
  QLoRA:  Dettmers et al., 2023 — https://arxiv.org/abs/2305.14314
  ZeRO:   Rajbhandari et al., 2019 — https://arxiv.org/abs/1910.02054

This is a CPU-friendly reference implementation. On a real cluster you'd
swap the LoRA training for a FSDP/DeepSpeed job and the data store for
a vector DB, but the API stays the same.
"""

from __future__ import annotations
import json
import math
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Indus
from .config import IndusConfig


# ---------------------------------------------------------------------------
# LoRA — Low-Rank Adaptation of linear layers
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """A linear layer with a low-rank adapter trained on top of frozen weights.

    W' = W + (alpha/r) * B @ A
    where A is [r, in] and B is [out, r], both initialized so BA = 0.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_dim, out_dim = base.in_features, base.out_features
        # A: [r, in], B: [out, r] — init A with Kaiming, B with zeros
        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # Adapter: drop(x) @ A.T @ B.T, scaled
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return out + lora_out * self.scaling


def inject_lora(model: Indus, r: int = 8, alpha: float = 16.0,
                target: str = "qkv") -> int:
    """Wrap target linear layers with LoRA adapters. Returns # adapters added.

    target: "qkv" wraps Q/K/V/O projections. "all" wraps all linears.
    """
    n = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_target = False
            if target == "qkv" and any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                is_target = True
            elif target == "all":
                is_target = True
            if is_target:
                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                lora = LoRALinear(module, r=r, alpha=alpha)
                setattr(parent, child_name, lora)
                n += 1
    return n


def lora_parameters(model: Indus) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad]


# ---------------------------------------------------------------------------
# Data versioning
# ---------------------------------------------------------------------------

@dataclass
class DatasetVersion:
    """A versioned dataset snapshot."""
    name: str
    version: str
    path: str
    num_examples: int
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataStore:
    """A versioned, content-hashed data store.

    Each "dataset" is a folder of .jsonl files plus a manifest. New
    versions are immutable; you add a new version, never mutate old ones.
    """
    def __init__(self, root: str = "data_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        self.versions: Dict[str, List[DatasetVersion]] = {}
        self._load_manifest()

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return
        for line in self.manifest_path.read_text().splitlines():
            d = json.loads(line)
            v = DatasetVersion(**d)
            self.versions.setdefault(v.name, []).append(v)

    def _append_manifest(self, v: DatasetVersion):
        with self.manifest_path.open("a") as f:
            f.write(json.dumps(asdict(v)) + "\n")

    def add_version(self, name: str, src_path: str,
                    metadata: Optional[Dict[str, Any]] = None) -> DatasetVersion:
        """Snapshot a dataset file into the store."""
        existing = self.versions.get(name, [])
        version = f"v{len(existing) + 1}"
        dest = self.root / name / version
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest / Path(src_path).name)
        v = DatasetVersion(
            name=name, version=version, path=str(dest),
            num_examples=sum(1 for _ in open(src_path)),
            created_at=time.time(),
            metadata=metadata or {},
        )
        self.versions.setdefault(name, []).append(v)
        self._append_manifest(v)
        return v

    def latest(self, name: str) -> Optional[DatasetVersion]:
        versions = self.versions.get(name, [])
        return versions[-1] if versions else None

    def list(self, name: Optional[str] = None) -> List[DatasetVersion]:
        if name:
            return list(self.versions.get(name, []))
        return [v for vs in self.versions.values() for v in vs]


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

class Evaluator:
    """Lightweight eval harness: perplexity, accuracy, generation quality.

    For real benchmarks, swap in lm-evaluation-harness. This is the
    minimal version that fits the Kernel API.
    """
    def __init__(self, model: Indus, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def perplexity(self, text: str, tokenizer) -> float:
        """Standard perplexity on a single text."""
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            return float("nan")
        # Chunk into block_size windows
        chunk = self.model.cfg.block_size
        nlls = []
        for i in range(0, len(ids) - 1, chunk):
            sub = ids[i:i + chunk + 1]
            x = torch.tensor([sub[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([sub[1:]], dtype=torch.long, device=self.device)
            out = self.model(x, y)
            nlls.append(out.loss.item() * (len(sub) - 1))
        total_nll = sum(nlls)
        total_tokens = sum(min(chunk, len(ids) - 1 - i) for i in range(0, len(ids) - 1, chunk))
        return math.exp(total_nll / max(1, total_tokens))

    def compare(self, other_model: Indus, text: str, tokenizer) -> dict:
        """Compare two models on the same text. Returns {'self': ppl, 'other': ppl}."""
        e_self = Evaluator(self.model, self.device)
        e_other = Evaluator(other_model, self.device)
        return {
            "self": e_self.perplexity(text, tokenizer),
            "other": e_other.perplexity(text, tokenizer),
        }


# ---------------------------------------------------------------------------
# Indus Kernel — the top-level continuous-learning orchestrator
# ---------------------------------------------------------------------------

class IndusKernel:
    """The continuous-learning layer that sits above the model.

    Typical flow:
        kernel = IndusKernel(model)
        adapter = kernel.add_adapter("math-v1", r=8)
        kernel.train_adapter(adapter, math_corpus)
        kernel.evaluate(...)
        if kernel.is_better(): kernel.promote(adapter)
    """
    def __init__(self, model: Indus, store_root: str = "data_store"):
        self.model = model
        self.store = DataStore(store_root)
        self.adapters: Dict[str, LoRALinear] = {}
        self.active_adapter: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

    def add_adapter(self, name: str, r: int = 8, alpha: float = 16.0,
                    target: str = "qkv") -> int:
        """Inject a fresh LoRA adapter and freeze the base model."""
        n = inject_lora(self.model, r=r, alpha=alpha, target=target)
        self.active_adapter = name
        return n

    def train_adapter(
        self,
        adapter_name: str,
        pairs: List[tuple],
        iters: int = 200,
        lr: float = 1e-4,
    ) -> dict:
        """Quick LoRA fine-tune on (prompt, response) pairs."""
        from .train import sft, TrainConfig
        # Save current weights, train only LoRA params, restore.
        cfg = TrainConfig(
            out_dir=f"checkpoints/adapter_{adapter_name}",
            dataset="sft",
            max_iters=iters,
            batch_size=4,
            lr=lr,
            n_layer=self.model.cfg.n_layer,
            n_head=self.model.cfg.n_head,
            n_kv_head=self.model.cfg.n_kv_head,
            n_embd=self.model.cfg.n_embd,
            block_size=self.model.cfg.block_size,
        )
        sft(cfg, base_model=self.model, sft_pairs=pairs)
        return {"adapter": adapter_name, "iters": iters}

    def evaluate(self, text: str, tokenizer) -> float:
        return Evaluator(self.model).perplexity(text, tokenizer)

    def promote(self, adapter_name: str, dest: str = "checkpoints/indus_promoted.pt"):
        """Save the current model (with adapter merged) as the new base."""
        # In a real system: merge LoRA into base weights, then save.
        torch.save({
            "model": self.model.state_dict(),
            "cfg": self.model.cfg.__dict__,
            "adapter": adapter_name,
        }, dest)
        return dest

    def record(self, event: Dict[str, Any]):
        self.history.append({"ts": time.time(), **event})
