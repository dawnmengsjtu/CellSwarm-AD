# -*- coding: utf-8 -*-
"""Neuron agent with viability tracking and fixed gene-weight metadata."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import random

# Dimensionless heuristic/model-assigned initialization weights for 12 AD risk
# genes.  These values encode relative model emphasis; they are not measured
# expression effects and are not precise estimates derived from ROSMAP, Jansen
# et al., or another transcriptomic/GWAS dataset.  ``NeuronAgent.step`` does not
# dynamically update them.
GENE_MAP: Dict[str, float] = {
    "APP": 1.0, "PSEN1": 0.8, "PSEN2": 0.6, "APOE4": 0.9,
    "TREM2": 0.7, "MAPT": 0.85, "BIN1": 0.5, "CLU": 0.55,
    "ABCA7": 0.45, "CD33": 0.4, "SORL1": 0.65, "ADAM10": 0.35,
}


@dataclass
class NeuronAgent:
    """Simulates a single neuron in AD context."""
    agent_id: str = "N_0"
    viability: float = 1.0
    calcium: float = 0.1
    tau_phosphorylation: float = 0.0
    # Retain the historical field name for API compatibility.  The mapping is
    # fixed per-agent model metadata, rather than dynamically simulated gene
    # expression.
    gene_expression: Dict[str, float] = field(default_factory=lambda: dict(GENE_MAP))

    def step(self, abeta_concentration: float = 0.0, dt: float = 0.01) -> Dict:
        """Advance one time step without modifying the fixed gene weights.

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

        self.calcium += abeta_concentration * 0.05 * dt
        self.tau_phosphorylation += self.calcium * 0.02 * dt
        damage = self.tau_phosphorylation * 0.1 * dt
        self.viability = max(0.0, self.viability - damage)
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "viability": round(self.viability, 4),
            "calcium": round(self.calcium, 4),
            "tau_phosphorylation": round(self.tau_phosphorylation, 4),
            "gene_expression": self.gene_expression,
        }
