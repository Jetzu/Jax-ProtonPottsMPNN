# Origin: cc (Claude Code, 2026-10-02) - vocabulary data frozen from the upstream mpnn package by
#         scripts/util/PPJAX_export_vocab_cc.py (atomworks TokenEncoding).
# Purpose: the token/atom vocabulary the model is sized by, without importing atomworks or torch.

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_DATA = Path(__file__).with_name("data") / "PPJAX_vocab.json"

# Protonation microstate -> canonical parent, the superset table of
# mpnn.metrics.sequence_recovery._build_canonical_map. Variants absent from a vocabulary are skipped.
PARENT_OF = {
    "HID": "HIS", "HIE": "HIS", "HIS-S": "HIS", "HIS-P": "HIS", "HIS-D": "HIS", "HIS-A": "HIS",
    "ASP-P": "ASP", "ASP-D": "ASP", "ASP-A": "ASP",
    "GLU-P": "GLU", "GLU-D": "GLU", "GLU-A": "GLU",
}


@dataclass(frozen=True)
class TokenEncoding:
    """The parts of ``atomworks.ml.encoding_definitions.TokenEncoding`` the model reads."""

    tokens: Tuple[str, ...]
    atom_order: Tuple[str, ...]
    n_atoms_per_token: int
    unknown_tokens: Tuple[str, ...]

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def token_to_idx(self) -> Dict[str, int]:
        return {t: i for i, t in enumerate(self.tokens)}

    @property
    def idx_to_token(self) -> Tuple[str, ...]:
        return self.tokens

    def atom_idx(self, token: str, atom_name: str) -> int:
        """Every token in these vocabularies shares one atom layout (verified at export time)."""
        return self.atom_order.index(atom_name)

    def atom_indices(self, atom_names: Sequence[str]) -> np.ndarray:
        """``[n_tokens, len(atom_names)]`` table, the port of ``construct_X_atoms``'s lookup."""
        row = np.array([self.atom_order.index(a) for a in atom_names], dtype=np.int64)
        return np.broadcast_to(row, (self.n_tokens, row.shape[0])).copy()

    def unknown_indices(self) -> List[int]:
        t2i = self.token_to_idx
        return [t2i[t] for t in self.unknown_tokens]

    def canonical_map(self) -> np.ndarray:
        """Token index -> canonical parent index (``_build_canonical_map``)."""
        t2i = self.token_to_idx
        cmap = np.arange(self.n_tokens, dtype=np.int64)
        for variant, parent in PARENT_OF.items():
            if variant in t2i and parent in t2i:
                cmap[t2i[variant]] = t2i[parent]
        return cmap


@lru_cache(maxsize=None)
def _raw() -> dict:
    return json.loads(_DATA.read_text())


@lru_cache(maxsize=None)
def get_encoding(name: str = "v6") -> TokenEncoding:
    """``v6`` (30 tokens, neutral His = HIS-S), ``potts32`` (v3/v4) or ``mpnn`` (21 tokens)."""
    raw = _raw()
    if name not in ("v6", "potts32", "mpnn"):
        raise ValueError(f"unknown vocabulary {name!r}; have v6 / potts32 / mpnn")
    d = raw[name]
    return TokenEncoding(tuple(d["tokens"]), tuple(d["atom_order"]),
                         int(d["n_atoms_per_token"]), tuple(d["unknown_tokens"]))


def vocab_encoding(extended_vocab) -> TokenEncoding:
    """Map the engine's ``extended_vocab`` argument onto an encoding, as the torch engine does:
    an explicit vocabulary NAME selects it; a truthy legacy bool is the 32-token Potts set; a
    falsy value is the plain 21-token MPNN set."""
    if isinstance(extended_vocab, str):
        return get_encoding({"v3": "potts32", "v4": "potts32", "v6": "v6"}.get(
            extended_vocab, extended_vocab))
    return get_encoding("potts32" if extended_vocab else "mpnn")


@lru_cache(maxsize=None)
def three_to_one() -> Dict[str, str]:
    return dict(_raw()["three_to_one"])


@lru_cache(maxsize=None)
def unknown_aa() -> str:
    return str(_raw()["unknown_aa"])
