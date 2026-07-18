# -*- coding: utf-8 -*-
"""Pharmacodynamics (PD) module.

Implements Emax model and inhibition models for drug-target interactions.
Supports agonists, antagonists, competitive and non-competitive inhibition.
Provides methods to modify CellSwarm-AD simulation parameters.

Emax model (Hill equation):
    E(C) = Emax * C^n / (EC50^n + C^n)

Competitive inhibition:
    E(C, I) = Emax * C / (EC50 * (1 + I/Ki) + C)

Non-competitive inhibition:
    E(C, I) = Emax / (1 + I/Ki) * C^n / (EC50^n + C^n)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class DrugType(Enum):
    """Classification of drug action."""
    AGONIST = "agonist"
    ANTAGONIST = "antagonist"
    PARTIAL_AGONIST = "partial_agonist"


class InhibitionType(Enum):
    """Type of enzyme/receptor inhibition."""
    NONE = "none"
    COMPETITIVE = "competitive"
    NON_COMPETITIVE = "non_competitive"
    UNCOMPETITIVE = "uncompetitive"
    MIXED = "mixed"


@dataclass
class PDParameters:
    """Pharmacodynamic parameters.

    Attributes:
        Emax: Maximum effect (fractional, 0-1 for inhibitors; >0 for activators)
        EC50: Concentration producing 50% of Emax (same units as PK concentration)
        n: Hill coefficient (cooperativity)
        drug_type: Agonist or antagonist
        inhibition_type: Type of inhibition (for antagonists/inhibitors)
        Ki: Inhibition constant (for competitive/non-competitive)
        baseline: Baseline effect in absence of drug
    """
    Emax: float
    EC50: float
    n: float = 1.0
    drug_type: DrugType = DrugType.AGONIST
    inhibition_type: InhibitionType = InhibitionType.NONE
    Ki: float = 0.0
    baseline: float = 0.0

    def __post_init__(self) -> None:
        if self.EC50 <= 0:
            raise ValueError("EC50 must be positive")
        if self.n <= 0:
            raise ValueError("Hill coefficient n must be positive")


@dataclass
class DrugEffect:
    """Result of PD calculation."""
    concentration: float
    effect: float           # normalised 0-1
    effect_absolute: float  # Emax-scaled
    inhibition_fraction: float  # fraction of target inhibited
    param_modifications: Dict[str, float] = field(default_factory=dict)


class PDModel:
    """Pharmacodynamic model for drug-target interactions.

    Encapsulates Emax/Hill equations, inhibition models, and
    parameter-modification logic for the CellSwarm-AD ODE layer.
    """

    def __init__(self, params: PDParameters) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Core effect equations
    # ------------------------------------------------------------------

    def emax_effect(self, C: float) -> float:
        """Hill / Emax equation: E = Emax * C^n / (EC50^n + C^n)."""
        if C <= 0:
            return 0.0
        p = self.params
        Cn = C ** p.n
        EC50n = p.EC50 ** p.n
        return p.Emax * Cn / (EC50n + Cn)

    def competitive_inhibition_effect(
        self, C: float, inhibitor_conc: float
    ) -> float:
        """Effect with competitive inhibitor present.

        Apparent EC50 is increased: EC50_app = EC50 * (1 + I/Ki)
        """
        if C <= 0:
            return 0.0
        p = self.params
        if p.Ki <= 0:
            raise ValueError("Ki must be positive for competitive inhibition")
        EC50_app = p.EC50 * (1 + inhibitor_conc / p.Ki)
        Cn = C ** p.n
        EC50n = EC50_app ** p.n
        return p.Emax * Cn / (EC50n + Cn)

    def non_competitive_inhibition_effect(
        self, C: float, inhibitor_conc: float
    ) -> float:
        """Effect with non-competitive inhibitor present.

        Emax is reduced: Emax_app = Emax / (1 + I/Ki)
        """
        if C <= 0:
            return 0.0
        p = self.params
        if p.Ki <= 0:
            raise ValueError("Ki must be positive for non-competitive inhibition")
        Emax_app = p.Emax / (1 + inhibitor_conc / p.Ki)
        Cn = C ** p.n
        EC50n = p.EC50 ** p.n
        return Emax_app * Cn / (EC50n + Cn)

    def uncompetitive_inhibition_effect(
        self, C: float, inhibitor_conc: float
    ) -> float:
        """Effect with uncompetitive inhibitor.

        Both EC50 and Emax are reduced by factor (1 + I/Ki).
        """
        if C <= 0:
            return 0.0
        p = self.params
        if p.Ki <= 0:
            raise ValueError("Ki must be positive")
        factor = 1 + inhibitor_conc / p.Ki
        Emax_app = p.Emax / factor
        EC50_app = p.EC50 / factor
        Cn = C ** p.n
        EC50n = EC50_app ** p.n
        return Emax_app * Cn / (EC50n + Cn)

    def effect(
        self,
        C: float,
        inhibitor_conc: float = 0.0,
    ) -> float:
        """Compute drug effect at concentration C.

        Dispatches to appropriate equation based on inhibition_type.
        For antagonists, returns the inhibition fraction (0 = no inhibition,
        1 = full inhibition).
        """
        p = self.params
        if p.inhibition_type == InhibitionType.NONE:
            raw = self.emax_effect(C)
        elif p.inhibition_type == InhibitionType.COMPETITIVE:
            raw = self.competitive_inhibition_effect(C, inhibitor_conc)
        elif p.inhibition_type == InhibitionType.NON_COMPETITIVE:
            raw = self.non_competitive_inhibition_effect(C, inhibitor_conc)
        elif p.inhibition_type == InhibitionType.UNCOMPETITIVE:
            raw = self.uncompetitive_inhibition_effect(C, inhibitor_conc)
        else:
            raw = self.emax_effect(C)
        return float(np.clip(raw, 0.0, p.Emax))

    def compute_effect(self, C: float, inhibitor_conc: float = 0.0) -> DrugEffect:
        """Full DrugEffect object at concentration C."""
        eff = self.effect(C, inhibitor_conc)
        inhibition_frac = eff / self.params.Emax if self.params.Emax > 0 else 0.0
        return DrugEffect(
            concentration=C,
            effect=eff / self.params.Emax if self.params.Emax > 0 else 0.0,
            effect_absolute=eff,
            inhibition_fraction=inhibition_frac,
        )

    # ------------------------------------------------------------------
    # CellSwarm-AD parameter modification
    # ------------------------------------------------------------------

    def modify_amyloid_params(
        self,
        C: float,
        base_k_prod: float,
        base_k_clear: float,
    ) -> Dict[str, float]:
        """Modify amyloid ODE parameters based on drug effect.

        Anti-Abeta antibodies:
        - Reduce k_prod (production inhibition)
        - Increase k_clear (enhanced clearance)

        Args:
            C: Drug plasma concentration (same units as EC50)
            base_k_prod: Baseline amyloid production rate
            base_k_clear: Baseline amyloid clearance rate

        Returns:
            Modified parameter dict for AmyloidODE
        """
        eff = self.effect(C)
        p = self.params

        if p.drug_type == DrugType.ANTAGONIST:
            # Inhibit production proportionally to effect
            new_k_prod = base_k_prod * (1.0 - eff)
            # Boost clearance
            new_k_clear = base_k_clear * (1.0 + eff)
        else:
            # Agonists increase clearance
            new_k_prod = base_k_prod
            new_k_clear = base_k_clear * (1.0 + eff)

        return {
            "k_prod": max(0.0, new_k_prod),
            "k_clear": max(0.0, new_k_clear),
        }

    def modify_tau_params(
        self,
        C: float,
        base_k_phospho: float,
        base_k_dephos: float,
    ) -> Dict[str, float]:
        """Modify tau ODE parameters (phosphorylation/dephosphorylation).

        Args:
            C: Drug plasma concentration
            base_k_phospho: Baseline phosphorylation rate
            base_k_dephos: Baseline dephosphorylation rate

        Returns:
            Modified parameter dict
        """
        eff = self.effect(C)
        new_k_phospho = base_k_phospho * (1.0 - eff * 0.8)  # max 80% reduction
        new_k_dephos = base_k_dephos * (1.0 + eff * 0.5)
        return {
            "k_phospho": max(0.0, new_k_phospho),
            "k_dephos": max(0.0, new_k_dephos),
        }

    def modify_cholinergic_params(
        self,
        C: float,
        base_ache_activity: float = 1.0,
    ) -> Dict[str, float]:
        """Modify cholinergic parameters (AChE inhibitor effect).

        Donepezil-like: increase ACh availability by inhibiting AChE.

        Returns:
            Modified params including ACh level multiplier
        """
        eff = self.effect(C)
        # AChE inhibited -> ACh rises
        ache_remaining = base_ache_activity * (1.0 - eff)
        ach_level = 1.0 / max(ache_remaining, 0.01)  # inverse relationship
        return {
            "ache_activity": max(0.0, ache_remaining),
            "ach_level": min(ach_level, 10.0),  # cap at 10x
        }

    def modify_nmda_params(
        self,
        C: float,
        base_nmda_conductance: float = 1.0,
    ) -> Dict[str, float]:
        """Modify NMDA receptor parameters (Memantine-like antagonism).

        Uncompetitive NMDA block reduces excitotoxicity.

        Returns:
            Modified NMDA conductance
        """
        eff = self.effect(C)
        new_conductance = base_nmda_conductance * (1.0 - eff * 0.9)
        return {
            "nmda_conductance": max(0.0, new_conductance),
            "excitotoxicity_index": 1.0 - eff,
        }

    # ------------------------------------------------------------------
    # Dose-response curve generation
    # ------------------------------------------------------------------

    def dose_response_curve(
        self,
        c_min: float = 1e-4,
        c_max: float = 100.0,
        n_points: int = 200,
        log_scale: bool = True,
    ) -> Dict[str, List[float]]:
        """Generate dose-response curve data.

        Args:
            c_min: Minimum concentration
            c_max: Maximum concentration
            n_points: Number of points
            log_scale: Use logarithmic spacing

        Returns:
            Dict with 'concentrations' and 'effects' lists
        """
        if log_scale:
            concs = list(np.logspace(np.log10(c_min), np.log10(c_max), n_points))
        else:
            concs = list(np.linspace(c_min, c_max, n_points))
        effects = [self.effect(c) for c in concs]
        return {"concentrations": concs, "effects": effects}
