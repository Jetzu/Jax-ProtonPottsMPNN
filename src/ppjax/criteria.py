# Origin: cc (Claude Code, 2026-10-02) - ported from mpnn.inference_engines.potts_mpnn_ph
#         (PHDesignCriteria, PlacementPin, PlacementPlan, PHDesignOutput, PHDesignSet).
# Purpose: the design-run configuration schema and output records, field-for-field identical to the
#          torch engine so configs and result tables transfer unchanged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

PLACEMENT_METHODS = ("autoregressive", "converged_mcmc", "two_phase", "converged_mcmc_combined",
                     "block_descent", "greedy_energy_block")
WHOLE_CHAIN_METHODS = ("gibbs", "mpnn_sample")
ALL_METHODS = PLACEMENT_METHODS + WHOLE_CHAIN_METHODS

DEFAULT_DEP_MAP: Dict[str, List[str]] = {
    "HIS-P": ["HID", "HIE"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"],
}
DEFAULT_FORBIDDEN_TOKENS: List[str] = ["HIS-A", "ASP-A", "GLU-A", "HIS-D"]


@dataclass
class PHDesignCriteria:
    """Criteria for a single design run (see the upstream docstring for each knob's meaning)."""

    backend: str = "potts"
    selective_source: str = "potts"
    center_protonation_types: List[str] = field(
        default_factory=lambda: ["HIS-P", "ASP-P", "GLU-P"])
    dep_map: Dict[str, List[str]] = field(default_factory=lambda: dict(DEFAULT_DEP_MAP))
    topk_sites: int = 2
    placement_seq_masked: bool = False

    method: str = "converged_mcmc"
    selective: bool = True
    combined_lambda: float = 1.0
    two_phase_frac: float = 0.5
    temperature: float = 0.1

    samples_per_site: int = 4
    cv_patience: int = 3
    cv_max: int = 50

    block_size: int = 2
    global_weight: float = 0.0
    rank_normalize: bool = False
    zscale_mode: str = "block"
    adjacent_repeat_weight: float = 0.0
    repetitive_window_weight: float = 0.0
    repetitive_window_radius: int = 2
    repetitive_window_parents: List[str] = field(default_factory=list)
    repetitive_window_gate_types: List[str] = field(default_factory=list)
    self_weight: float = 1.0
    block_max_rounds: int = 10

    seed_source: str = "inverse"
    forbidden_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_TOKENS))
    num_designs: int = 8
    record_trajectory: bool = False

    center_count: int = 1
    explicit_centers: List[Dict] = field(default_factory=list)
    center_types: List[str] = field(default_factory=list)
    placement_region: List[str] = field(default_factory=lambda: ["all"])
    placement_by: str = "scan_potts"
    placement_label: str = ""
    candidate_pool: int = 12
    n_plan_samples: int = 64
    max_plans_per_seed: int = 8
    infill_scope: str = "neighbourhood"
    sweep_order: str = "position"
    neighbour_k: int = 0
    max_mutations: int = 0

    def __post_init__(self) -> None:
        if self.method not in ALL_METHODS:
            raise ValueError(f"Unknown method '{self.method}'. Available: {ALL_METHODS}")
        if self.backend not in ("potts", "mpnn"):
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'potts' or 'mpnn'.")
        if self.seed_source not in ("inverse", "native"):
            raise ValueError(f"Unknown seed_source '{self.seed_source}'.")
        if self.method == "gibbs" and self.backend != "potts":
            raise ValueError("method='gibbs' requires backend='potts'.")
        if self.method in ("mpnn_sample", "autoregressive") and self.backend != "mpnn":
            raise ValueError(f"method='{self.method}' requires backend='mpnn'.")
        if self.selective_source not in ("potts", "decoder"):
            raise ValueError(f"Unknown selective_source '{self.selective_source}'.")
        if self.selective_source == "decoder" and self.method != "autoregressive":
            raise ValueError("selective_source='decoder' is only valid for method='autoregressive'.")
        _centre_free = (self.method == "greedy_energy_block" and self.infill_scope == "chain"
                        and not self.center_types and not self.explicit_centers)
        if self.center_count < 1 and not (self.center_count == 0 and _centre_free):
            raise ValueError(f"center_count must be >= 1 (got {self.center_count}).")
        bad = set(self.placement_region) - {"interface", "core", "surface", "all"}
        if bad:
            raise ValueError(f"Unknown placement_region {bad}.")
        if self.sweep_order not in ("position", "knn", "energy"):
            raise ValueError(f"Unknown sweep_order '{self.sweep_order}'.")
        if self.infill_scope not in ("neighbourhood", "chain"):
            raise ValueError(f"Unknown infill_scope '{self.infill_scope}'.")
        if self.placement_by not in ("random", "scan_potts", "scan_mpnn"):
            raise ValueError(f"Unknown placement_by '{self.placement_by}'.")
        if self.zscale_mode not in ("single_mutation", "block"):
            raise ValueError(f"Unknown zscale_mode '{self.zscale_mode}'.")
        if self.self_weight < 0:
            raise ValueError(f"self_weight must be >= 0 (got {self.self_weight}).")
        if self.explicit_centers and self.center_count not in (1, len(self.explicit_centers)):
            raise ValueError("center_count must equal len(explicit_centers).")
        if self.center_types:
            bad_t = set(self.center_types) - set(self.dep_map)
            if bad_t:
                raise ValueError(f"center_types {bad_t} not in dep_map keys {set(self.dep_map)}.")
            self.center_count = len(self.center_types)

    def stability_weights(self):
        return float(self.self_weight), 1.0

    @property
    def is_placement(self) -> bool:
        return self.method in PLACEMENT_METHODS

    def scheme_label(self) -> str:
        obj = "selective" if self.selective else "non_selective"
        if self.method == "converged_mcmc_combined":
            base = f"converged_mcmc_combined_l{self.combined_lambda:g}"
        elif self.method == "two_phase":
            base = f"two_phase_f{self.two_phase_frac:g}"
        elif self.method == "converged_mcmc":
            base = f"{obj}_converged"
        elif self.method == "autoregressive":
            base = "selective" if self.selective else "likelihood"
            if self.selective_source == "decoder":
                base += "_dec"
        elif self.method == "block_descent":
            base = f"block_descent_b{self.block_size}_l{self.combined_lambda:g}"
            if self.zscale_mode == "single_mutation":
                base += "_sm"
            if self.adjacent_repeat_weight:
                base += f"_ar{self.adjacent_repeat_weight:g}"
            if self.repetitive_window_weight:
                base += f"_rw{self.repetitive_window_weight:g}"
            if self.self_weight != 1.0:
                base += f"_sw{self.self_weight:g}"
        else:
            base = self.method
        if self.placement_seq_masked and self.is_placement:
            base += "_smask"
        return base

    @classmethod
    def from_params(cls, params: Optional[Dict]) -> "PHDesignCriteria":
        params = dict(params or {})
        known = set(cls.__dataclass_fields__)
        crit = cls(**{k: v for k, v in params.items() if k in known})
        crit.center_protonation_types = list(crit.center_protonation_types)
        crit.forbidden_tokens = list(crit.forbidden_tokens)
        crit.dep_map = {k: list(v) for k, v in dict(crit.dep_map).items()}
        crit.placement_region = list(crit.placement_region)
        crit.explicit_centers = [dict(c) for c in crit.explicit_centers]
        crit.repetitive_window_parents = list(crit.repetitive_window_parents)
        crit.repetitive_window_gate_types = list(crit.repetitive_window_gate_types)
        crit.center_types = list(crit.center_types)
        return crit


@dataclass
class PlacementPin:
    position: int
    protonation_type: str
    prot_idx: int
    dep_idxs: List[int]
    res_id: int


@dataclass
class PlacementPlan:
    pins: List[PlacementPin]
    designable: List[int]
    label: str
    placement_score: float = 0.0

    @property
    def seed_key(self) -> int:
        if not self.pins:
            return (self.designable[0] * 97 + self.designable[-1] * 7) if self.designable else 0
        return self.pins[0].position * 97 + sum(p.position for p in self.pins[1:]) * 7


@dataclass
class PHDesignOutput:
    binder_chain: str
    canonical_sequence: str
    extended_tokens: List[str]
    extended_vocab: Any
    scheme: str
    sample: int
    final_potts_energy: float
    seed_idx: Optional[int] = None
    protonation_type: Optional[str] = None
    site_rank: Optional[int] = None
    center_res_id: Optional[int] = None
    n_neighbours: Optional[int] = None
    selective_energy: Optional[float] = None
    n_centers: Optional[int] = None
    center_res_ids: Optional[List[int]] = None
    center_protonation_types: Optional[List[str]] = None
    selective_energies: Optional[List[float]] = None
    global_protonation_dH: Optional[float] = None
    placement_prob: Optional[float] = None
    placement_entropy: Optional[float] = None
    sequence_decoded_prob_score: Optional[float] = None
    sequence_entropy: Optional[float] = None
    method: Optional[str] = None
    backend: Optional[str] = None
    selective_source: Optional[str] = None
    placement_label: Optional[str] = None
    placement_region: Optional[str] = None
    placement_by: Optional[str] = None
    combined_lambda: Optional[float] = None
    repetitive_window_weight: Optional[float] = None
    block_size: Optional[int] = None
    sweep_order: Optional[str] = None
    neighbour_k: Optional[int] = None
    max_mutations: Optional[int] = None
    energy_trajectory: Optional[List[Dict[str, Any]]] = None

    def design_id(self) -> str:
        if self.protonation_type is not None and self.center_res_ids:
            seed_tag = "" if self.seed_idx is None else f"_seed{self.seed_idx}"
            res_tag = "-".join(str(r) for r in self.center_res_ids)
            return f"{self.protonation_type}_{self.scheme}{seed_tag}_res{res_tag}_s{self.sample}"
        return f"{self.scheme}_s{self.sample}"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "design_id": self.design_id(), "scheme": self.scheme, "seed_idx": self.seed_idx,
            "extended_vocab": self.extended_vocab,
            "extended_tokens": " ".join(self.extended_tokens),
            "potts_energy": self.final_potts_energy, "selective_energy": self.selective_energy,
            "global_protonation_dH": self.global_protonation_dH,
            "placement_prob": self.placement_prob, "placement_entropy": self.placement_entropy,
            "sequence_decoded_prob_score": self.sequence_decoded_prob_score,
            "sequence_entropy": self.sequence_entropy,
            "protonation_type": self.protonation_type, "site_rank": self.site_rank,
            "center_res_id": self.center_res_id, "n_neighbours": self.n_neighbours,
            "n_centers": self.n_centers, "center_res_ids": self.center_res_ids,
            "center_protonation_types": self.center_protonation_types,
            "selective_energies": self.selective_energies,
            "method": self.method, "backend": self.backend,
            "selective_source": self.selective_source, "placement_label": self.placement_label,
            "placement_region": self.placement_region, "placement_by": self.placement_by,
            "combined_lambda": self.combined_lambda,
            "repetitive_window_weight": self.repetitive_window_weight,
            "block_size": self.block_size, "sweep_order": self.sweep_order,
            "neighbour_k": self.neighbour_k, "max_mutations": self.max_mutations,
            "energy_trajectory": self.energy_trajectory,
        }


class PHDesignSet(list):
    def sorted_by_energy(self) -> "PHDesignSet":
        return PHDesignSet(sorted(self, key=lambda d: d.final_potts_energy))

    def deduped(self) -> "PHDesignSet":
        seen, out = set(), PHDesignSet()
        for d in self:
            did = d.design_id()
            if did not in seen:
                seen.add(did)
                out.append(d)
        return out

    def top(self, n: int) -> "PHDesignSet":
        return PHDesignSet(self.deduped().sorted_by_energy()[:n])
