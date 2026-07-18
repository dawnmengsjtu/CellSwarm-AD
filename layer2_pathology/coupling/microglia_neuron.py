# -*- coding: utf-8 -*-
"""Microglia → Neuron coupling via cytokine signaling.

Reference: Tang & Le (2016) Mol Neurobiol - Microglial polarization in AD.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class MicrogliaNeuronCoupling:
    """M1 microglia damage neurons (TNF-α), M2 protect (TGF-β).

    Formula: net_effect = k_damage * [TNFa] - k_protect * [TGFb]
    """
    k_damage: float = 0.05    # TNF-α damage coefficient (calibrated to match Sobol sensitivity)
    k_protect: float = 0.1    # TGF-β protection coefficient

    def compute_net_effect(self, tnf_alpha: float, tgf_beta: float) -> float:
        """Compute net effect on neuron viability.

        Args:
            tnf_alpha: TNF-α concentration (pro-inflammatory)
            tgf_beta: TGF-β concentration (anti-inflammatory)

        Returns:
            Net effect (positive = damage, negative = protection)
        """
        if tnf_alpha < 0:
            raise ValueError(f"TNF-α must be non-negative, got {tnf_alpha}")
        if tgf_beta < 0:
            raise ValueError(f"TGF-β must be non-negative, got {tgf_beta}")
        return self.k_damage * tnf_alpha - self.k_protect * tgf_beta

    def compute_from_state(self, microglia_state: str) -> Dict[str, float]:
        """Estimate cytokine levels from microglia activation state.

        Args:
            microglia_state: 'M0', 'M1', or 'M2'

        Returns:
            Dict with tnf_alpha and tgf_beta levels
        """
        cytokines = {
            "M0": {"tnf_alpha": 0.1, "tgf_beta": 0.1},
            "M1": {"tnf_alpha": 0.8, "tgf_beta": 0.05},
            "M2": {"tnf_alpha": 0.05, "tgf_beta": 0.6},
        }
        return cytokines.get(microglia_state, cytokines["M0"])

    def apply(self, viability: float, microglia_state: str, dt: float) -> float:
        """Apply coupling: update neuron viability based on microglia state.

        Args:
            viability: Current neuron viability (0-1)
            microglia_state: 'M0', 'M1', or 'M2'
            dt: Timestep

        Returns:
            Updated viability (clamped to [0, 1])
        """
        cytokines = self.compute_from_state(microglia_state)
        effect = self.compute_net_effect(cytokines["tnf_alpha"], cytokines["tgf_beta"])
        return max(0.0, min(1.0, viability - effect * dt))
