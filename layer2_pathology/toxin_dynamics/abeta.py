# -*- coding: utf-8 -*-
"""Amyloid-beta dynamics model."""
from dataclasses import dataclass
from typing import Dict, List
import math


@dataclass
class AbetaDynamics:
    """Models Abeta production, aggregation, and clearance."""
    production_rate: float = 0.1
    aggregation_rate: float = 0.05
    clearance_rate: float = 0.03
    monomer_conc: float = 0.0
    oligomer_conc: float = 0.0
    plaque_conc: float = 0.0

    def step(self, dt: float = 0.01, microglial_clearance: float = 0.0) -> Dict:
        production = self.production_rate * dt
        self.monomer_conc += production
        aggregation = self.aggregation_rate * self.monomer_conc * dt
        self.monomer_conc -= aggregation
        self.oligomer_conc += aggregation
        plaque_formation = self.aggregation_rate * self.oligomer_conc * 0.5 * dt
        self.oligomer_conc -= plaque_formation
        self.plaque_conc += plaque_formation
        total_clearance = (self.clearance_rate + microglial_clearance) * dt
        self.monomer_conc = max(0, self.monomer_conc - total_clearance * self.monomer_conc)
        self.oligomer_conc = max(0, self.oligomer_conc - total_clearance * 0.5 * self.oligomer_conc)
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "monomer": round(self.monomer_conc, 6),
            "oligomer": round(self.oligomer_conc, 6),
            "plaque": round(self.plaque_conc, 6),
            "total_abeta": round(self.monomer_conc + self.oligomer_conc + self.plaque_conc, 6),
        }
