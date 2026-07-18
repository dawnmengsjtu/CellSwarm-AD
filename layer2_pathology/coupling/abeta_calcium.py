# -*- coding: utf-8 -*-
"""Aβ → Calcium coupling via NMDA receptor-mediated influx.

Reference: Bhatt et al. (2009) J Neurosci - Aβ oligomers enhance NMDA receptor currents.
"""
from dataclasses import dataclass


@dataclass
class AbetaCalciumCoupling:
    """Aβ oligomers increase calcium influx through NMDA receptors.

    Formula: Ca_influx = k_abeta_ca * [Aβ_oligo] / (K_m + [Aβ_oligo])
    (Michaelis-Menten saturation kinetics)
    """
    k_abeta_ca: float = 0.5    # Max calcium influx rate
    K_m: float = 0.1           # Half-saturation constant for Aβ oligomers
    # Ca²⁺ clearance parameters (PMCA pump + SERCA + passive leak)
    # Bhatt et al. (2009); Bhargava et al. (2013) Cell Calcium - PMCA/SERCA kinetics
    k_clearance: float = 0.3   # Ca²⁺ clearance rate (PMCA + SERCA), s⁻¹
    ca_rest: float = 0.1       # Resting Ca²⁺ concentration (μM)

    def compute_influx(self, abeta_oligomer: float) -> float:
        """Compute calcium influx from Aβ oligomer concentration.

        Args:
            abeta_oligomer: Aβ oligomer concentration (non-negative)

        Returns:
            Calcium influx rate
        """
        if abeta_oligomer < 0:
            raise ValueError(f"Aβ oligomer concentration must be non-negative, got {abeta_oligomer}")
        return self.k_abeta_ca * abeta_oligomer / (self.K_m + abeta_oligomer)

    def apply(self, calcium: float, abeta_oligomer: float, dt: float) -> float:
        """Apply coupling: update calcium level.

        Args:
            calcium: Current calcium concentration
            abeta_oligomer: Aβ oligomer concentration
            dt: Timestep

        Returns:
            Updated calcium concentration
        """
        influx = self.compute_influx(abeta_oligomer)
        clearance = self.k_clearance * (calcium - self.ca_rest)
        return calcium + (influx - clearance) * dt
