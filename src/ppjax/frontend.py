# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: the structure front end. Parsing, hydrogen placement, HBPLUS and the FLAML protonation
#          labeller are C / scikit-learn code with no JAX counterpart, so ppjax consumes their
#          OUTPUT: either a .npz of featurised arrays (torch-free), or, when the upstream `mpnn`
#          package happens to be importable, a direct bridge to prepare_potts_input.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Arrays a featurisation must carry for the model to run at all.
REQUIRED_FEATURE_KEYS = ("X", "X_m", "S", "R_idx", "chain_labels", "residue_mask")
# Arrays that may legitimately be absent: `designed_residue_mask` defaults to all valid residues,
# `temperature` is only read by the decoder field, and `symmetry_equivalence_group` is None for
# every structure this pipeline produces (and is refused by the port anyway).
OPTIONAL_FEATURE_KEYS = ("designed_residue_mask", "temperature", "symmetry_equivalence_group")
FEATURE_KEYS = REQUIRED_FEATURE_KEYS + OPTIONAL_FEATURE_KEYS
TOKEN_KEYS = ("token_res_id", "token_res_name", "token_chain_id")
REGION_NAMES = ("interface", "core", "surface")


@dataclass
class FeaturisedBackbone:
    """Everything one backbone contributes, independent of the model."""

    features: Dict[str, np.ndarray]
    token_res_id: np.ndarray
    token_res_name: np.ndarray
    token_chain_id: np.ndarray
    region_masks: Dict[str, np.ndarray]
    binder_chain: str = "A"

    def save(self, path: str) -> None:
        arrays = {f"ni_{k}": np.asarray(v) for k, v in self.features.items()}
        arrays["token_res_id"] = self.token_res_id
        arrays["token_res_name"] = np.asarray(self.token_res_name).astype("U8")
        arrays["token_chain_id"] = np.asarray(self.token_chain_id).astype("U8")
        for n, m in self.region_masks.items():
            arrays[f"region_{n}"] = m
        arrays["binder_chain"] = np.asarray(self.binder_chain)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str, binder_chain: Optional[str] = None) -> "FeaturisedBackbone":
        d = np.load(path, allow_pickle=False)
        feats = {k[3:]: d[k] for k in d.files
                 if k.startswith("ni_") and k[3:] in FEATURE_KEYS + ("structure_noise",)}
        if "structure_noise" in feats:
            feats["structure_noise"] = float(np.asarray(feats["structure_noise"]))
        missing = [k for k in REQUIRED_FEATURE_KEYS if k not in feats]
        if missing:
            raise KeyError(
                f"{path} is missing featurisation arrays {missing}; it was not written by "
                f"ppjax.frontend.featurise_structure / FeaturisedBackbone.save")
        regions = {n: d[f"region_{n}"] for n in REGION_NAMES if f"region_{n}" in d.files}
        chain = binder_chain or (str(d["binder_chain"]) if "binder_chain" in d.files else "A")
        return cls(features=feats, token_res_id=d["token_res_id"],
                   token_res_name=d["token_res_name"], token_chain_id=d["token_chain_id"],
                   region_masks=regions, binder_chain=chain)


def featurise_structure(structure, binder_chain: str = "A", extended_vocab: str = "v6",
                        structure_noise: float = 0.0) -> FeaturisedBackbone:
    """Bridge to the upstream transform pipeline (needs `mpnn` + HBPLUS_PATH on this interpreter).

    This is the ONE place ppjax touches torch, and only to read tensors straight back out as
    numpy. Prefer running it once in the upstream environment and saving a .npz.
    """
    import torch  # noqa: F401  (imported lazily: ppjax itself never needs it)
    from atomworks.ml.utils.token import get_token_starts
    from mpnn.potts_inference import prepare_potts_input

    batch = prepare_potts_input(structure, designed_chains=[binder_chain],
                                structure_noise=structure_noise, extended_vocab=extended_vocab)
    ni = batch["network_input"]["input_features"]
    feats = {k: np.asarray(ni[k].detach().cpu().numpy())
             for k in FEATURE_KEYS if ni.get(k) is not None}
    # structure_noise is a scalar in the torch network_input and is applied inside the featuriser;
    # ppjax applies it in ppjax.model.apply_structure_noise so the RNG draw keeps its place in the
    # stream. Carry it through explicitly rather than letting it be silently dropped.
    feats["structure_noise"] = float(structure_noise)
    proc = batch["atom_array"]
    token_aa = proc[get_token_starts(proc)]
    regions = _region_masks(proc, token_aa, binder_chain)
    return FeaturisedBackbone(features=feats, token_res_id=np.asarray(token_aa.res_id),
                              token_res_name=np.asarray(token_aa.res_name).astype("U8"),
                              token_chain_id=np.asarray(token_aa.chain_id).astype("U8"),
                              region_masks=regions, binder_chain=binder_chain)


# Tien et al. 2013 (PLoS ONE 8:e80635) theoretical maximum accessible surface area, A^2.
_MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223,
            "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240,
            "PRO": 159, "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}


def _region_masks(aa, token_aa, binder_chain: str, interface_dist: float = 6.0,
                  rasa_threshold: float = 0.2) -> Dict[str, np.ndarray]:
    """``_binder_region_masks``: CA-CA interface contacts + residue-level RASA core/surface.
    Fail-soft, exactly like upstream - a missing region simply yields an empty mask."""
    L = len(token_aa)
    iface = np.zeros(L, bool); core = np.zeros(L, bool); surface = np.zeros(L, bool)
    binder = np.asarray(token_aa.chain_id) == binder_chain
    pos_of = {(str(token_aa.chain_id[i]), int(token_aa.res_id[i])): i
              for i in range(L) if binder[i]}
    binder_ca = (aa.chain_id == binder_chain) & (aa.atom_name == "CA")
    target_ca = (aa.chain_id != binder_chain) & (aa.atom_name == "CA")
    try:
        from biotite.structure import CellList
        target_ca_arr = aa[target_ca]
        binder_ca_idx = np.where(binder_ca)[0]
        if target_ca_arr.array_length() and len(binder_ca_idx):
            cell_list = CellList(target_ca_arr, cell_size=float(interface_dist))
            contacts = cell_list.get_atoms(aa.coord[binder_ca], radius=float(interface_dist))
            has_contact = np.asarray(contacts != -1).reshape(len(binder_ca_idx), -1).any(axis=1)
            for k, ca_idx in enumerate(binder_ca_idx):
                if bool(has_contact[k]):
                    p = pos_of.get((str(aa.chain_id[ca_idx]), int(aa.res_id[ca_idx])))
                    if p is not None:
                        iface[p] = True
    except Exception:
        pass
    try:
        from atomworks.ml.transforms.sasa import calculate_atomwise_sasa
        sasa = calculate_atomwise_sasa(aa, probe_radius=1.4, atom_radii="ProtOr", point_number=100)
        sasa = np.nan_to_num(np.asarray(sasa, dtype=float), nan=0.0)
        for ca_idx in np.where(binder_ca)[0]:
            chain, res_id = aa.chain_id[ca_idx], aa.res_id[ca_idx]
            p = pos_of.get((str(chain), int(res_id)))
            if p is None:
                continue
            sel = (aa.chain_id == chain) & (aa.res_id == res_id)
            max_asa = _MAX_ASA.get(str(aa.res_name[ca_idx]))
            if max_asa is None or not sel.any():
                continue
            if float(sasa[sel].sum()) / max_asa < rasa_threshold:
                core[p] = True
            else:
                surface[p] = True
    except Exception:
        pass
    return {"interface": iface, "core": core, "surface": surface}
