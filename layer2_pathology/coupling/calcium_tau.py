# -*- coding: utf-8 -*-
"""Calcium → Tau phosphorylation coupling via GSK-3β/CDK5 activation.

Reference: Hernández et al. (2010) Neurochem Int - Calcium-dependent kinase activation.
"""
from dataclasses import dataclass


@dataclass
class CalciumTauCoupling:
    """Calcium overload activates GSK-3β/CDK5, promoting tau phosphorylation.

    Formula: k_phos_eff = k_phos_base * (1 + alpha * max(0, [Ca] - Ca_threshold))
    """
    k_phos_base: float = 0.05   # Baseline phosphorylation rate
    alpha: float = 2.0           # Calcium sensitivity coefficient
    ca_threshold: float = 0.2    # Calcium threshold for kinase activation
    # PP2A-mediated dephosphorylation
    # Liu et al. (2005) J Biol Chem - PP2A is the major tau phosphatase in brain
    k_dephos: float = 0.03       # Dephosphorylation rate (PP2A), s⁻¹

    def effective_phosphorylation_rate(self, calcium: float) -> float:
        """Compute effective tau phosphorylation rate given calcium level.

        Args:
            calcium: Cytosolic calcium concentration

        Returns:
            Effective phosphorylation rate
        """
        if calcium < 0:
            raise ValueError(f"Calcium must be non-negative, got {calcium}")
        excess = max(0.0, calcium - self.ca_threshold)
        return self.k_phos_base * (1.0 + self.alpha * excess)

    def apply(self, p_tau: float, calcium: float, dt: float) -> float:
        """Apply coupling: update phosphorylated tau level.

        Args:
            p_tau: Current phosphorylated tau
            calcium: Calcium concentration
            dt: Timestep

        Returns:
            Updated phosphorylated tau
        """
        phos_rate = self.effective_phosphorylation_rate(calcium)
        dephos_rate = self.k_dephos * p_tau
        return p_tau + (phos_rate - dephos_rate) * dt
