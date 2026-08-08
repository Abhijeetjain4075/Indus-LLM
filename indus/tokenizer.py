"""BPE tokenizer wrapper for Indus.

Uses tiktoken under the hood (OpenAI's BPE implementation). Defaults
to the cl100k_base tokenizer (used by GPT-3.5/4) which is fast and
broadly compatible.
"""

from __future__ import annotations
from typing import List, Union

import tiktoken
import torch


class IndusTokenizer:
    """Thin wrapper around tiktoken for encode/decode round-trips.

    The character-level mode is for tiny demo runs where BPE would
    be overkill (e.g., Shakespeare).
    """
    def __init__(self, encoding: str = "gpt2"):
        self.encoding_name = encoding
        if encoding == "char":
            self._char_init = None  # lazy
            self.enc = None
            self.vocab_size = None
            self.eos_token_id = 0
            self.bos_token_id = 0
        else:
            self.enc = tiktoken.get_encoding(encoding)
            self.vocab_size = self.enc.max_token_value + 1
            self.eos_token_id = self.enc.eot_token
            self.bos_token_id = self.enc.encode_ordinary("<|endoftext|>")[0]

    def encode(self, text: str) -> List[int]:
        if self.enc is None:
            return self._char_encode(text)
        return self.enc.encode_ordinary(text)

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        if self.enc is None:
            return self._char_decode(ids)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.enc.decode(ids)

    # ---- char mode ----
    def _char_init_or_get(self, text: str):
        if self._char_init is None:
            chars = sorted(set(text))
            self._char_init = chars
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(chars)
            self.eos_token_id = 0
            self.bos_token_id = 0

    def _char_init_from(self, stoi: dict, itos: dict):
        """Bootstrap the char vocab from a saved mapping (for round-trip loading)."""
        self.stoi = dict(stoi)
        self.itos = dict(itos)
        self.vocab_size = len(stoi)
        self.eos_token_id = 0
        self.bos_token_id = 0
        self._char_init = sorted(self.stoi.keys())

    def _char_encode(self, text: str) -> List[int]:
        self._char_init_or_get(text)
        return [self.stoi[c] for c in text]

    def _char_decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)
