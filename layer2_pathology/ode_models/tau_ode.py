# -*- coding: utf-8 -*-
"""Tau phosphorylation ODE model."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TauODE:
    """ODE model for tau phosphorylation cascade.

    dT/dt = -k_phos * T * K + k_dephos * Tp
    dTp/dt = k_phos * T * K - k_dephos * Tp - k_tangle * Tp
    dNFT/dt = k_tangle * Tp
    """
    k_phos: float = 0.05
    k_dephos: float = 0.02
    k_tangle: float = 0.01
    T: float = 1.0    # normal tau
    Tp: float = 0.0   # phosphorylated tau
    NFT: float = 0.0  # neurofibrillary tangles
    kinase: float = 0.5  # kinase activity
    trajectory: List[Dict] = field(default_factory=list)

    def derivatives(self) -> tuple:
        dT = -self.k_phos * self.T * self.kinase + self.k_dephos * self.Tp
        dTp = self.k_phos * self.T * self.kinase - self.k_dephos * self.Tp - self.k_tangle * self.Tp
        dNFT = self.k_tangle * self.Tp
        return dT, dTp, dNFT

    def step(self, dt: float = 0.01) -> Dict:
        dT, dTp, dNFT = self.derivatives()
        self.T = max(0, self.T + dT * dt)
        self.Tp = max(0, self.Tp + dTp * dt)
        self.NFT = max(0, self.NFT + dNFT * dt)
        state = self.get_state()
        self.trajectory.append(state)
        return state

    def run(self, steps: int = 100, dt: float = 0.01, kinase_activity: float = 0.5) -> List[Dict]:
        self.kinase = kinase_activity
        return [self.step(dt) for _ in range(steps)]

    def get_state(self) -> Dict:
        return {"T": round(self.T, 6), "Tp": round(self.Tp, 6), "NFT": round(self.NFT, 6)}
