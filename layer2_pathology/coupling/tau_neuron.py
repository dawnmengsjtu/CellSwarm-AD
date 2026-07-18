# -*- coding: utf-8 -*-
"""Tau → Neuron damage coupling via synaptic dysfunction.

Reference: Ittner & Götz (2011) Nat Rev Neurosci - Tau-mediated neurodegeneration.
"""
from dataclasses import dataclass


@dataclass
class TauNeuronCoupling:
    """Phosphorylated tau causes neuronal damage (Hill function).

    Formula: viability_loss = k_damage * [pTau]^n / (K_d^n + [pTau]^n)
    """
    k_damage: float = 0.1   # Max damage rate
    K_d: float = 0.5         # Half-max damage concentration
    n: float = 2.0           # Hill coefficient (cooperativity)

    def compute_damage(self, p_tau: float) -> float:
        """Compute viability loss rate from phosphorylated tau.

        Args:
            p_tau: Phosphorylated tau concentration (non-negative)

        Returns:
            Viability loss rate
        """
        if p_tau < 0:
            raise ValueError(f"p-Tau must be non-negative, got {p_tau}")
        p_tau_n = p_tau ** self.n
        return self.k_damage * p_tau_n / (self.K_d ** self.n + p_tau_n)

    def apply(self, viability: float, p_tau: float, dt: float) -> float:
        """Apply coupling: update neuron viability.

        Args:
            viability: Current neuron viability (0-1)
            p_tau: Phosphorylated tau concentration
            dt: Timestep

        Returns:
            Updated viability (clamped to [0, 1])
        """
        damage = self.compute_damage(p_tau)
        return max(0.0, min(1.0, viability - damage * dt))
