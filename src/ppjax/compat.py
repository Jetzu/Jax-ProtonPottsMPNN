# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX after the N001 red-team.
# Purpose: surface the places where the port CANNOT be made identical to torch, instead of letting
#          them pass silently. torch's `argsort`/`topk` use an UNSTABLE sort whose tie order matches
#          neither numpy's stable nor its quicksort ordering (measured:
#          scripts/util/PPJAX_probe_torch_reductions_cc.py), so an exact tie at a decisive
#          comparison is a point where the two implementations may legitimately disagree.

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


class PPJAXCompatWarning(UserWarning):
    """A place where the port can differ from the torch original for this particular input."""


def warn_on_ties(values: np.ndarray, where: str, k: Optional[int] = None) -> int:
    """Warn when ``values`` contains exact duplicates that a selection could break either way.

    ``k`` restricts the check to the decision boundary: only duplicates spanning the k-th / (k+1)-th
    place can change which items are selected. Returns the number of tied groups found.
    """
    v = np.asarray(values).reshape(-1)
    if v.size < 2:
        return 0
    order = np.argsort(v, kind="stable")
    s = v[order]
    tied = s[:-1] == s[1:]
    if k is not None and 0 < k < v.size:
        # only a tie straddling the cut matters for a top-k selection
        tied = tied & (np.arange(v.size - 1) >= k - 1) & (np.arange(v.size - 1) <= k - 1)
    n = int(tied.sum())
    if n:
        warnings.warn(
            f"{where}: {n} exact tie(s) in the ranked values. torch breaks ties with an UNSTABLE "
            f"sort that this port cannot reproduce, so the selection here may differ from the "
            f"torch original for this input.", PPJAXCompatWarning, stacklevel=2)
    return n


def warn_on_near_tie(values: np.ndarray, k: int, where: str, atol: float = 1e-4) -> bool:
    """Warn when a top-``k`` selection is decided by less than ``atol``.

    Unlike :func:`warn_on_ties` this does NOT require bitwise equality: when the k-th and (k+1)-th
    ranked values are closer than the port's float32 agreement with torch, the two implementations
    can legitimately select different items. Measured instance: ``greedy_energy_block`` re-ranks
    every designable position by its single-residue improvement at every step, and on the shipped
    PD-L1 example the 3rd/4th improvements differ by 3.05e-05 at step 12 - which is what makes that
    one reference case diverge (scripts/analysis/PPJAX_measure_greedy_flip_cc.py).
    """
    v = np.sort(np.asarray(values).reshape(-1), kind="stable")
    if k <= 0 or k >= v.size:
        return False
    gap = float(abs(v[k] - v[k - 1]))
    if gap < atol:
        warnings.warn(
            f"{where}: the top-{k} cut is decided by {gap:.3e}, below the port's float32 agreement "
            f"with torch (~{atol:.0e}). The two implementations may select different items here, "
            f"which changes the rest of the trajectory.", PPJAXCompatWarning, stacklevel=2)
        return True
    return False


def check_knn_tie_boundary(boundary: np.ndarray) -> int:
    """Warn when a residue's k-th and (k+1)-th neighbour distance are exactly equal.

    ``boundary`` is the ``[..., 2]`` pair produced by :func:`ppjax.features.knn_graph`. A tie there
    means ``torch.topk`` and ``jax.lax.top_k`` may keep different neighbours, which changes the
    model's graph - not just its rounding.
    """
    b = np.asarray(boundary).reshape(-1, 2)
    n = int((b[:, 0] == b[:, 1]).sum())
    if n:
        warnings.warn(
            f"kNN graph: {n} residue(s) have an exact distance tie at the k-th/(k+1)-th cut. "
            f"torch.topk and jax.lax.top_k may keep different neighbours there, which changes the "
            f"model graph for this input.", PPJAXCompatWarning, stacklevel=2)
    return n
