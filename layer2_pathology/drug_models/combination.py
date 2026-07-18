# -*- coding: utf-8 -*-
"""Combination therapy models.

Implements Bliss independence and Loewe additivity for drug combinations.
Detects synergy, additivity, and antagonism.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

from .pharmacodynamics import PDModel


class InteractionType(Enum):
    """Drug interaction classification."""
    SYNERGISTIC = "synergistic"
    ADDITIVE = "additive"
    ANTAGONISTIC = "antagonistic"


@dataclass
class CombinationResult:
    """Result of combination analysis."""
    drug_a: str
    drug_b: str
    effect_a: float
    effect_b: float
    expected_combined: float    # predicted by null model
    observed_combined: float    # actual (or simulated)
    interaction_type: InteractionType
    combination_index: float    # CI < 1 synergy, =1 additive, >1 antagonism

    def to_dict(self) -> Dict:
        return {
            "drug_a": self.drug_a,
            "drug_b": self.drug_b,
            "effect_a": round(self.effect_a, 4),
            "effect_b": round(self.effect_b, 4),
            "expected": round(self.expected_combined, 4),
            "observed": round(self.observed_combined, 4),
            "interaction": self.interaction_type.value,
            "CI": round(self.combination_index, 4),
        }


def bliss_independence(effect_a: float, effect_b: float) -> float:
    """Bliss independence model: E_AB = E_A + E_B - E_A * E_B.

    Assumes drugs act on independent targets.
    Effects should be fractional (0-1).
    """
    if not (0 <= effect_a <= 1 and 0 <= effect_b <= 1):
        raise ValueError("Effects must be in [0, 1] for Bliss model")
    return effect_a + effect_b - effect_a * effect_b


def loewe_additivity(
    conc_a: float, conc_b: float,
    ec50_a: float, ec50_b: float,
    emax_a: float = 1.0, emax_b: float = 1.0,
    n_a: float = 1.0, n_b: float = 1.0,
) -> float:
    """Loewe additivity combination index.

    CI = (Ca/ICx_a) + (Cb/ICx_b)
    CI < 1: synergy, CI = 1: additive, CI > 1: antagonism

    Uses the isobologram approach.
    """
    # Effect of each drug alone at given concentration
    def hill(c, ec50, emax, n):
        if c <= 0:
            return 0.0
        return emax * c**n / (ec50**n + c**n)

    ea = hill(conc_a, ec50_a, emax_a, n_a)
    eb = hill(conc_b, ec50_b, emax_b, n_b)
    e_combo = max(ea, eb)  # approximate combined effect

    if e_combo <= 0:
        return 1.0  # no effect, additive by default

    # Inverse Hill to get ICx concentrations
    def inverse_hill(e, ec50, emax, n):
        if e <= 0 or e >= emax:
            return float('inf')
        return ec50 * (e / (emax - e)) ** (1.0 / n)

    icx_a = inverse_hill(e_combo, ec50_a, emax_a, n_a)
    icx_b = inverse_hill(e_combo, ec50_b, emax_b, n_b)

    if icx_a == float('inf') or icx_b == float('inf'):
        return 1.0

    ci = conc_a / icx_a + conc_b / icx_b
    return ci


class CombinationModel:
    """Analyze drug combinations for synergy/antagonism."""

    def __init__(self, pd_a: PDModel, pd_b: PDModel, name_a: str = "Drug A", name_b: str = "Drug B"):
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.name_a = name_a
        self.name_b = name_b

    def analyze_bliss(self, conc_a: float, conc_b: float, observed: float = None) -> CombinationResult:
        """Analyze combination using Bliss independence.

        Args:
            conc_a: Concentration of drug A
            conc_b: Concentration of drug B
            observed: Observed combined effect (if None, uses Bliss prediction)
        """
        ea = self.pd_a.effect(conc_a) / self.pd_a.params.Emax if self.pd_a.params.Emax > 0 else 0
        eb = self.pd_b.effect(conc_b) / self.pd_b.params.Emax if self.pd_b.params.Emax > 0 else 0

        expected = bliss_independence(ea, eb)
        if observed is None:
            observed = expected

        if observed > expected * 1.05:
            interaction = InteractionType.SYNERGISTIC
        elif observed < expected * 0.95:
            interaction = InteractionType.ANTAGONISTIC
        else:
            interaction = InteractionType.ADDITIVE

        ci = expected / observed if observed > 0 else 1.0

        return CombinationResult(
            drug_a=self.name_a,
            drug_b=self.name_b,
            effect_a=ea,
            effect_b=eb,
            expected_combined=expected,
            observed_combined=observed,
            interaction_type=interaction,
            combination_index=ci,
        )

    def analyze_loewe(self, conc_a: float, conc_b: float) -> CombinationResult:
        """Analyze combination using Loewe additivity."""
        pa = self.pd_a.params
        pb = self.pd_b.params

        ea = self.pd_a.effect(conc_a) / pa.Emax if pa.Emax > 0 else 0
        eb = self.pd_b.effect(conc_b) / pb.Emax if pb.Emax > 0 else 0

        ci = loewe_additivity(
            conc_a, conc_b,
            pa.EC50, pb.EC50,
            pa.Emax, pb.Emax,
            pa.n, pb.n,
        )

        if ci < 0.9:
            interaction = InteractionType.SYNERGISTIC
        elif ci > 1.1:
            interaction = InteractionType.ANTAGONISTIC
        else:
            interaction = InteractionType.ADDITIVE

        expected = bliss_independence(ea, eb)

        return CombinationResult(
            drug_a=self.name_a,
            drug_b=self.name_b,
            effect_a=ea,
            effect_b=eb,
            expected_combined=expected,
            observed_combined=expected,
            interaction_type=interaction,
            combination_index=ci,
        )

    def checkerboard(self, concs_a: List[float], concs_b: List[float]) -> List[List[float]]:
        """Generate checkerboard CI matrix for dose combinations."""
        matrix = []
        for ca in concs_a:
            row = []
            for cb in concs_b:
                result = self.analyze_loewe(ca, cb)
                row.append(result.combination_index)
            matrix.append(row)
        return matrix
