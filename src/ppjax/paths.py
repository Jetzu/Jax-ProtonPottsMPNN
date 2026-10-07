# Origin: cc (Claude Code, 2026-10-07) - written for PPJAX; no upstream.
# Purpose: locate the upstream ProtonPottsMPNN checkout and the v6 checkpoint without hard-coding
#          one machine's layout. The golden reference files record the ABSOLUTE paths they were
#          generated from, which is useful provenance but useless on anyone else's machine, so the
#          recorded path is only ever the last fallback.

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

CKPT_RELPATH = Path("checkpoints") / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"


def repo_root() -> Path:
    """The checkout this package lives in.

    In the JAX fork of ProtonPottsMPNN the whole upstream tree sits alongside ``src/ppjax``, so the
    checkpoint is already in the repo at ``checkpoints/...`` and needs no environment variable.
    """
    return Path(__file__).resolve().parents[2]


def upstream_root() -> Path:
    """The unmodified ProtonPottsMPNN checkout.

    ``PROTON_ROOT`` is upstream's own convention for this; it wins. Otherwise ``~/ProtonPottsMPNN``.
    """
    return Path(os.environ.get("PROTON_ROOT") or (Path.home() / "ProtonPottsMPNN")).expanduser()


def resolve_checkpoint(recorded: Optional[str] = None) -> str:
    """The v6 checkpoint, in order of preference.

    1. ``PROTON_CKPT`` - an explicit path to the ``.ckpt``.
    2. ``PROTON_ROOT`` / the default upstream checkout, at the standard relative path.
    3. this checkout itself - the case where the port lives inside the ProtonPottsMPNN tree.
    4. ``recorded`` - the absolute path a golden reference file was generated from. Correct on the
       machine that produced it, meaningless elsewhere, so it is tried last.

    Raises with all three candidates named, rather than letting a loader fail on a path the caller
    never chose. The checkpoint is NOT redistributed with this port; get it from upstream.
    """
    candidates = []
    env_ckpt = os.environ.get("PROTON_CKPT")
    if env_ckpt:
        candidates.append(Path(env_ckpt).expanduser())
    candidates.append(upstream_root() / CKPT_RELPATH)
    candidates.append(repo_root() / CKPT_RELPATH)
    if recorded:
        candidates.append(Path(recorded))
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        "could not find the ProtonPottsMPNN v6 checkpoint. It is not redistributed with this port - "
        "get it from https://github.com/christian-creator/ProtonPottsMPNN and point PROTON_CKPT at "
        "the .ckpt, or PROTON_ROOT at the checkout. Tried: "
        + ", ".join(str(c) for c in candidates))
