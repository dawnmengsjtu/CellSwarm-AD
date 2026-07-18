# -*- coding: utf-8 -*-
"""Astrocyte agent with reactivity tracking."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class AstrocyteAgent:
    """Simulates astrocyte behavior in AD context."""
    agent_id: str = "AS_0"
    reactivity: float = 0.0
    glutamate_uptake: float = 0.8
    calcium_wave: float = 0.0

    def step(self, abeta_concentration: float = 0.0, cytokine_level: float = 0.0, dt: float = 0.01) -> Dict:
        """Advance one time step."""
        self.reactivity += (cytokine_level + abeta_concentration * 0.1) * 0.15 * dt
        self.reactivity = min(1.0, self.reactivity)
        self.glutamate_uptake = max(0.1, 0.8 - self.reactivity * 0.5)
        self.calcium_wave = self.reactivity * 0.3
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "reactivity": round(self.reactivity, 4),
            "glutamate_uptake": round(self.glutamate_uptake, 4),
            "calcium_wave": round(self.calcium_wave, 4),
        }
