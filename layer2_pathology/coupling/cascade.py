# -*- coding: utf-8 -*-
"""Complete AD pathology cascade integrating all coupling mechanisms.

Cascade: Aβ → Ca²⁺ → tau → neuron death
         Aβ → microglia → neuron damage/protection
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

from .abeta_calcium import AbetaCalciumCoupling
from .calcium_tau import CalciumTauCoupling
from .tau_neuron import TauNeuronCoupling
from .abeta_microglia import AbetaMicrogliaCoupling
from .microglia_neuron import MicrogliaNeuronCoupling


@dataclass
class CascadeState:
    """State of the full AD cascade."""
    abeta_oligomer: float = 0.0
    calcium: float = 0.1
    p_tau: float = 0.0
    viability: float = 1.0
    nfkb: float = 0.0
    microglia_state: str = "M0"
    time: float = 0.0
    # Feedback loop parameters (for step_with_feedback)
    abeta_production_rate: float = 0.1  # nM/s, baseline Aβ production
    abeta_clearance_rate: float = 0.02  # s⁻¹, baseline clearance
    k_tau_to_abeta: float = 0.05  # tau → Aβ feedback strength

    def to_dict(self) -> Dict:
        return {
            "abeta_oligomer": round(self.abeta_oligomer, 6),
            "calcium": round(self.calcium, 6),
            "p_tau": round(self.p_tau, 6),
            "viability": round(self.viability, 6),
            "nfkb": round(self.nfkb, 6),
            "microglia_state": self.microglia_state,
            "time": round(self.time, 4),
        }


@dataclass
class ADCascade:
    """Full AD pathology cascade integrating all coupling modules.

    Two parallel pathways:
    1. Amyloid pathway: Aβ → Ca²⁺ → tau → neuron death
    2. Inflammatory pathway: Aβ → microglia → neuron damage/protection
    """
    abeta_calcium: AbetaCalciumCoupling = field(default_factory=AbetaCalciumCoupling)
    calcium_tau: CalciumTauCoupling = field(default_factory=CalciumTauCoupling)
    tau_neuron: TauNeuronCoupling = field(default_factory=TauNeuronCoupling)
    abeta_microglia: AbetaMicrogliaCoupling = field(default_factory=AbetaMicrogliaCoupling)
    microglia_neuron: MicrogliaNeuronCoupling = field(default_factory=MicrogliaNeuronCoupling)
    state: CascadeState = field(default_factory=CascadeState)

    def _update_microglia_state(self) -> None:
        """Update microglia polarization based on NF-κB activity."""
        if self.state.nfkb >= 0.6:
            self.state.microglia_state = "M1"
        elif self.state.nfkb >= 0.3:
            self.state.microglia_state = "M2"
        else:
            self.state.microglia_state = "M0"

    def step(self, abeta_oligomer: float, dt: float = 0.01) -> Dict:
        """Run one step of the full cascade.

        Args:
            abeta_oligomer: Current Aβ oligomer concentration
            dt: Timestep

        Returns:
            Current cascade state as dict
        """
        if abeta_oligomer < 0:
            raise ValueError(f"Aβ oligomer must be non-negative, got {abeta_oligomer}")
        if dt <= 0:
            raise ValueError(f"Timestep must be positive, got {dt}")

        self.state.abeta_oligomer = abeta_oligomer

        # Pathway 1: Aβ → Ca²⁺ → tau → neuron
        self.state.calcium = self.abeta_calcium.apply(
            self.state.calcium, abeta_oligomer, dt
        )
        # Physiological clamp: cytosolic Ca²⁺ range 0.05-5.0 μM
        # (pathological Ca²⁺ can reach several μM in AD neurons)
        self.state.calcium = max(0.05, min(5.0, self.state.calcium))

        self.state.p_tau = self.calcium_tau.apply(
            self.state.p_tau, self.state.calcium, dt
        )
        # Physiological clamp: p-tau fraction 0-1
        self.state.p_tau = max(0.0, min(1.0, self.state.p_tau))

        viability_after_tau = self.tau_neuron.apply(
            self.state.viability, self.state.p_tau, dt
        )

        # Pathway 2: Aβ → microglia → neuron
        self.state.nfkb = self.abeta_microglia.apply(
            self.state.nfkb, abeta_oligomer, dt
        )
        # Clamp NF-κB activity 0-1
        self.state.nfkb = max(0.0, min(1.0, self.state.nfkb))

        self._update_microglia_state()
        viability_after_microglia = self.microglia_neuron.apply(
            viability_after_tau, self.state.microglia_state, dt
        )

        # Clamp viability 0-1
        self.state.viability = max(0.0, min(1.0, viability_after_microglia))
        self.state.time += dt

        return self.state.to_dict()

    def run(self, abeta_oligomer: float, steps: int = 100, dt: float = 0.01) -> list:
        """Run the cascade for multiple steps.

        Args:
            abeta_oligomer: Constant Aβ oligomer concentration
            steps: Number of steps
            dt: Timestep

        Returns:
            List of state dicts at each step
        """
        trajectory = []
        for _ in range(steps):
            state = self.step(abeta_oligomer, dt)
            trajectory.append(state.copy())
        return trajectory

    def get_state(self) -> Dict:
        """Get current cascade state."""
        return self.state.to_dict()

    def step_with_feedback(self, dt: float = 0.01) -> Dict:
        """Run one step with tau → Aβ feedback loop.

        In this version, Aβ is an internal state variable that evolves according to:
        d[Aβ]/dt = production - clearance + k_tau_to_abeta * p_tau * Aβ

        The feedback term (k_tau_to_abeta * p_tau * Aβ) represents tau-mediated
        promotion of Aβ aggregation (Ittner & Götz 2011, Rapoport et al. 2002).

        Args:
            dt: Timestep

        Returns:
      Current cascade state as dict
        """
        if dt <= 0:
            raise ValueError(f"Timestep must be positive, got {dt}")

        # Update Aβ with feedback: d[Aβ]/dt = production - clearance + feedback
        abeta_production = self.state.abeta_production_rate
        abeta_clearance = self.state.abeta_clearance_rate * self.state.abeta_oligomer
        tau_feedback = self.state.k_tau_to_abeta * self.state.p_tau * self.state.abeta_oligomer

        d_abeta_dt = abeta_production - abeta_clearance + tau_feedback
        self.state.abeta_oligomer += d_abeta_dt * dt

        # Clamp Aβ to non-negative
        self.state.abeta_oligomer = max(0.0, self.state.abeta_oligomer)

        # Pathway 1: Aβ → Ca²⁺ → tau → neuron
        self.state.calcium = self.abeta_calcium.apply(
            self.state.calcium, self.state.abeta_oligomer, dt
        )
        self.state.calcium = max(0.05, min(5.0, self.state.calcium))

        self.state.p_tau = self.calcium_tau.apply(
            self.state.p_tau, self.state.calcium, dt
        )
        self.state.p_tau = max(0.0, min(1.0, self.state.p_tau))

        viability_after_tau = self.tau_neuron.apply(
            self.state.viability, self.state.p_tau, dt
        )

        # Pathway 2: Aβ → microglia → neuron
        self.state.nfkb = self.abeta_microglia.apply(
            self.state.nfkb, self.state.abeta_oligomer, dt
        )
        self.state.nfkb = max(0.0, min(1.0, self.state.nfkb))

        self._update_microglia_state()
        viability_after_microglia = self.microglia_neuron.apply(
            viability_after_tau, self.state.microglia_state, dt
        )

        self.state.viability = max(0.0, min(1.0, viability_after_microglia))
        self.state.time += dt

        return self.state.to_dict()

    def run_with_feedback(self, steps: int = 100, dt: float = 0.01) -> list:
        """Run the cascade with tau → Aβ feedback for multiple steps.

        Args:
            steps: Number of steps
            dt: Timestep

        Returns:
            List of state dicts at each step
        """
        trajectory = []
        for _ in range(steps):
            state = self.step_with_feedback(dt)
            trajectory.append(state.copy())
        return trajectory

    def reset(self) -> None:
        """Reset cascade to initial state."""
        self.state = CascadeState()
