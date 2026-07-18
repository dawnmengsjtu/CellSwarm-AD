# -*- coding: utf-8 -*-
"""Microglia agent with activation states and phagocytosis."""
from dataclasses import dataclass, field
from typing import Dict
from enum import Enum
import random


class ActivationState(Enum):
    M0 = "resting"
    M1 = "pro_inflammatory"
    M2 = "anti_inflammatory"


@dataclass
class MicrogliaAgent:
    """Simulates microglial cell behavior in AD.

    NF-κB dynamics: activation by Aβ with natural decay.
    State transitions: probabilistic based on NF-κB level.
    M0 (resting) → M1 (pro-inflammatory) at high NF-κB
    M0 (resting) → M2 (anti-inflammatory) at moderate NF-κB
    M1 ↔ M2 transitions possible with probability.
    """
    agent_id: str = "MG_0"
    activation_state: ActivationState = ActivationState.M0
    nfkb_activity: float = 0.0
    phagocytosis_rate: float = 0.3
    cytokine_production: Dict[str, float] = field(default_factory=dict)

    def step(self, abeta_concentration: float = 0.0, dt: float = 0.01) -> Dict:
        """Advance one time step.

        Args:
            abeta_concentration: Abeta concentration (must be non-negative)
            dt: Timestep (must be positive)

        Returns:
            Current cell state

        Raises:
            ValueError: If parameters are invalid
        """
        if abeta_concentration < 0:
            raise ValueError(f"Abeta concentration must be non-negative, got {abeta_concentration}")
        if dt <= 0:
            raise ValueError(f"Timestep must be positive, got {dt}")

        # NF-κB dynamics: activation by Aβ + natural decay + noise
        activation = abeta_concentration * 0.1 * dt
        decay = self.nfkb_activity * 0.05 * dt  # 5% decay per unit time
        noise = random.gauss(0, 0.01 * dt)

        self.nfkb_activity += activation - decay + noise
        self.nfkb_activity = max(0.0, min(1.0, self.nfkb_activity))

        # Probabilistic state transitions based on NF-κB level
        r = random.random()

        if self.nfkb_activity > 0.6:
            # High NF-κB: mostly M1, some M2
            if r < 0.75:
                self.activation_state = ActivationState.M1
            elif r < 0.95:
                self.activation_state = ActivationState.M2
            else:
                self.activation_state = ActivationState.M0
        elif self.nfkb_activity > 0.3:
            # Moderate NF-κB: mixed M1/M2
            if r < 0.3:
                self.activation_state = ActivationState.M1
            elif r < 0.7:
                self.activation_state = ActivationState.M2
            else:
                self.activation_state = ActivationState.M0
        elif self.nfkb_activity > 0.1:
            # Low NF-κB: mostly M0, some M2
            if r < 0.1:
                self.activation_state = ActivationState.M1
            elif r < 0.3:
                self.activation_state = ActivationState.M2
            else:
                self.activation_state = ActivationState.M0
        else:
            # Very low NF-κB: resting
            if r < 0.05:
                self.activation_state = ActivationState.M2
            else:
                self.activation_state = ActivationState.M0

        # Update functional outputs based on state
        if self.activation_state == ActivationState.M1:
            self.cytokine_production = {
                "tnf_alpha": self.nfkb_activity * 0.8,
                "il1_beta": self.nfkb_activity * 0.6
            }
            self.phagocytosis_rate = 0.1 + random.gauss(0, 0.02)
        elif self.activation_state == ActivationState.M2:
            self.cytokine_production = {
                "il10": self.nfkb_activity * 0.3,
                "tgf_beta": self.nfkb_activity * 0.2
            }
            self.phagocytosis_rate = 0.6 + random.gauss(0, 0.05)
        else:
            self.cytokine_production = {}
            self.phagocytosis_rate = 0.3 + random.gauss(0, 0.03)

        self.phagocytosis_rate = max(0.0, min(1.0, self.phagocytosis_rate))

        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "activation_state": self.activation_state.value,
            "nfkb_activity": round(self.nfkb_activity, 4),
            "phagocytosis_rate": round(self.phagocytosis_rate, 4),
            "cytokine_production": self.cytokine_production,
        }
