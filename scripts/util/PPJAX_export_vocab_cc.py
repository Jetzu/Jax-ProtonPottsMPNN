# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; reads the upstream mpnn package read-only.
# Purpose: freeze the mpnn TokenEncoding vocabularies (v6 / v4-32-token / plain MPNN) into a JSON
#          data file so ppjax carries the vocabulary without importing atomworks or torch.
import json, sys
from pathlib import Path
from mpnn.transforms.feature_aggregation.token_encodings import (
    MPNN_TOKEN_ENCODING, POTTS_MPNN_TOKEN_ENCODING, POTTS_MPNN_V6_TOKEN_ENCODING)
from atomworks.constants import DICT_THREE_TO_ONE, UNKNOWN_AA

def dump(enc):
    toks = [str(t) for t in enc.idx_to_token]
    atoms = sorted({a for (t, a) in enc.atom_to_idx}, key=lambda a: enc.atom_to_idx[(toks[0], a)])
    # verify the atom layout is identical for every token
    for t in toks:
        for a in atoms:
            assert enc.atom_to_idx[(t, a)] == enc.atom_to_idx[(toks[0], a)], (t, a)
    return {
        "tokens": toks,
        "atom_order": atoms,
        "n_atoms_per_token": int(enc.n_atoms_per_token),
        "n_tokens": int(enc.n_tokens),
        "unknown_tokens": [str(t) for t in enc.unknown_tokens],
    }

out = {
    "v6": dump(POTTS_MPNN_V6_TOKEN_ENCODING),
    "potts32": dump(POTTS_MPNN_TOKEN_ENCODING),
    "mpnn": dump(MPNN_TOKEN_ENCODING),
    "three_to_one": {str(k): str(v) for k, v in DICT_THREE_TO_ONE.items()},
    "unknown_aa": str(UNKNOWN_AA),
}
dst = Path(sys.argv[1] if len(sys.argv) > 1 else "src/ppjax/data/PPJAX_vocab.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print("wrote", dst, {k: (v["n_tokens"] if isinstance(v, dict) and "n_tokens" in v else "-") for k, v in out.items()})
print("v6 tokens:", out["v6"]["tokens"])
print("v6 unknown:", out["v6"]["unknown_tokens"], "n_atoms", out["v6"]["n_atoms_per_token"])
