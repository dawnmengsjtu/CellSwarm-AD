# -*- coding: utf-8 -*-
"""Aβ → Microglia activation coupling via TLR4/TREM2.

Reference: Heneka et al. (2015) Lancet Neurol - Neuroinflammation in AD.
"""
from dataclasses import dataclass


@dataclass
class AbetaMicrogliaCoupling:
    """Aβ activates microglia through TLR4, modulated by TREM2.

    Formula: activation = k_act * [Aβ] * (1 - trem2_inhibition)
    """
    k_act: float = 0.3              # Activation rate constant
    trem2_inhibition: float = 0.2   # TREM2-mediated inhibition (0-1)
    # IκBα-mediated negative feedback on NF-κB
    # Hoffmann et al. (2002) Science - The IκB–NF-κB signaling module
    k_nfkb_decay: float = 0.2      # NF-κB decay rate via IκBα re-synthesis, s⁻¹

    def compute_activation(self, abeta: float) -> float:
        """Compute microglial activation signal from Aβ.

        Args:
            abeta: Total Aβ concentration (non-negative)

        Returns:
            Activation signal strength
        """
        if abeta < 0:
            raise ValueError(f"Aβ must be non-negative, got {abeta}")
        return self.k_act * abeta * (1.0 - self.trem2_inhibition)

    def apply(self, nfkb: float, abeta: float, dt: float) -> float:
        """Apply coupling: update NF-κB activity.

        Args:
            nfkb: Current NF-κB activity level
            abeta: Aβ concentration
            dt: Timestep

        Returns:
            Updated NF-κB activity (clamped to [0, 1])
        """
        signal = self.compute_activation(abeta)
        decay = self.k_nfkb_decay * nfkb
        return max(0.0, min(1.0, nfkb + (signal - decay) * dt))
