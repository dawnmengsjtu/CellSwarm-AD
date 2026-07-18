# -*- coding: utf-8 -*-
"""Calcium signaling ODE model."""
from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass
class CalciumODE:
    """ODE model for intracellular calcium dynamics.

    dCa/dt = k_influx * stimulus - k_pump * Ca + k_leak * (Ca_er - Ca)
    dCa_er/dt = k_pump * Ca - k_leak * (Ca_er - Ca) - k_release * Ca_er
    """
    channel_rate: float = 0.1  # k_influx
    pump_rate: float = 0.05    # k_pump
    leak_rate: float = 0.01    # k_leak
    k_release: float = 0.02
    Ca: float = 0.0001   # cytosolic calcium (mM)
    Ca_er: float = 0.5   # ER calcium (mM)
    stimulus: float = 0.0
    trajectory: List[Dict] = field(default_factory=list)

    def derivatives(self) -> tuple:
        dCa = self.channel_rate * self.stimulus - self.pump_rate * self.Ca + self.leak_rate * (self.Ca_er - self.Ca)
        dCa_er = self.pump_rate * self.Ca - self.leak_rate * (self.Ca_er - self.Ca) - self.k_release * self.Ca_er
        return dCa, dCa_er

    def step(self, dt: float = 0.01) -> Dict:
        dCa, dCa_er = self.derivatives()
        self.Ca = max(0, self.Ca + dCa * dt)
        self.Ca_er = max(0, self.Ca_er + dCa_er * dt)
        state = self.get_state()
        self.trajectory.append(state)
        return state

    def run(self, steps: int = 100, dt: float = 0.01, stimulus: float = 0.0) -> List[Dict]:
        self.stimulus = stimulus
        return [self.step(dt) for _ in range(steps)]

    def get_state(self) -> Dict:
        return {"Ca_cyt": round(self.Ca, 6), "Ca_er": round(self.Ca_er, 6)}
