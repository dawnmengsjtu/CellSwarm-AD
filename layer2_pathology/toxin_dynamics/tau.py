# -*- coding: utf-8 -*-
"""Tau protein dynamics model."""
from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class TauDynamics:
    """Models tau phosphorylation and tangle formation."""
    phosphorylation_rate: float = 0.05
    tangle_rate: float = 0.02
    clearance_rate: float = 0.01
    p_tau: float = 0.0
    tangles: float = 0.0

    def step(self, dt: float = 0.01, kinase_activity: float = 0.0) -> Dict:
        phosphorylation = (self.phosphorylation_rate + kinase_activity) * dt
        self.p_tau += phosphorylation
        tangle_formation = self.tangle_rate * self.p_tau * dt
        self.p_tau -= tangle_formation
        self.tangles += tangle_formation
        clearance = self.clearance_rate * self.p_tau * dt
        self.p_tau = max(0, self.p_tau - clearance)
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "p_tau": round(self.p_tau, 6),
            "tangles": round(self.tangles, 6),
        }
