# -*- coding: utf-8 -*-
"""Amyloid cascade ODE model."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AmyloidODE:
    """ODE model for amyloid cascade dynamics.

    dA/dt = k_prod - k_clear * A - k_agg * A^2
    dO/dt = k_agg * A^2 - k_plaque * O
    dP/dt = k_plaque * O
    """
    k_prod: float = 0.1
    k_clear: float = 0.03
    k_agg: float = 0.01
    k_plaque: float = 0.005
    A: float = 0.0  # monomer
    O: float = 0.0  # oligomer
    P: float = 0.0  # plaque
    trajectory: List[Dict] = field(default_factory=list)

    def derivatives(self) -> tuple:
        dA = self.k_prod - self.k_clear * self.A - self.k_agg * self.A ** 2
        dO = self.k_agg * self.A ** 2 - self.k_plaque * self.O
        dP = self.k_plaque * self.O
        return dA, dO, dP

    def step(self, dt: float = 0.01) -> Dict:
        dA, dO, dP = self.derivatives()
        self.A = max(0, self.A + dA * dt)
        self.O = max(0, self.O + dO * dt)
        self.P = max(0, self.P + dP * dt)
        state = self.get_state()
        self.trajectory.append(state)
        return state

    def run(self, steps: int = 100, dt: float = 0.01) -> List[Dict]:
        return [self.step(dt) for _ in range(steps)]

    def get_state(self) -> Dict:
        return {"A": round(self.A, 6), "O": round(self.O, 6), "P": round(self.P, 6)}
