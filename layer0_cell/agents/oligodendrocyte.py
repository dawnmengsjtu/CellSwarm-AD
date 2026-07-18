# -*- coding: utf-8 -*-
"""Oligodendrocyte agent with myelination and maturation tracking.

AD relevance:
- Oligodendrocyte precursor cells (OPCs) fail to differentiate in AD
  (Behrendt et al., 2013, Glia; Desai et al., 2010, Neurosci Lett)
- Aβ oligomers directly impair OPC maturation and myelination capacity
  (Horiuchi et al., 2012, J Neuroimmunol)
- TNF-α from M1 microglia inhibits OPC differentiation
  (Cunha et al., 2020, Front Cell Neurosci)
- Calcium dysregulation leads to myelin sheath decompaction
  (Baraban et al., 2018, Nat Neurosci)
- White matter degeneration is an early feature of AD
  (Nasrabady et al., 2018, Acta Neuropathol)
"""
from dataclasses import dataclass
from typing import Dict
from enum import Enum
import random


class MaturationState(Enum):
    """Oligodendrocyte lineage maturation stages."""
    OPC = "OPC"                # Oligodendrocyte precursor cell
    PRE_OL = "pre_OL"         # Pre-oligodendrocyte (intermediate)
    MATURE_OL = "mature_OL"   # Mature myelinating oligodendrocyte


@dataclass
class OligodendrocyteAgent:
    """Simulates oligodendrocyte behavior in AD context.

    Maturation: OPC → pre-OL → mature OL, gated by Aβ and TNF-α.
    Myelination capacity degrades with Aβ exposure.
    Myelin integrity degrades with elevated Ca²⁺.
    TNF-α from M1 microglia blocks OPC → pre-OL transition.
    """
    agent_id: str = "OL_0"
    maturation_state: MaturationState = MaturationState.OPC
    myelination_capacity: float = 0.8   # baseline capacity; OPCs start high
    myelin_integrity: float = 1.0       # intact myelin at start
    # Internal maturation progress: accumulates toward thresholds
    _maturation_progress: float = 0.0

    # --- Biologically-motivated parameters (see docstring refs) ---
    # Aβ effect on myelination: ~5% reduction per unit Aβ per dt
    #   (Horiuchi et al., 2012 — dose-dependent OPC toxicity)
    ABETA_MYELINATION_COEFF: float = 0.05
    # Ca²⁺ effect on myelin integrity: ~8% degradation per unit Ca²⁺ per dt
    #   (Baraban et al., 2018 — Ca²⁺-mediated myelin decompaction)
    CALCIUM_INTEGRITY_COEFF: float = 0.08
    # TNF-α inhibition of maturation: blocks ~70% of differentiation signal
    #   (Cunha et al., 2020 — TNF-α suppresses OPC differentiation)
    TNF_MATURATION_INHIBITION: float = 0.7
    # Base maturation rate per dt (slow, weeks-scale in vivo)
    BASE_MATURATION_RATE: float = 0.02
    # Maturation thresholds (progress needed to advance)
    OPC_TO_PRE_OL_THRESHOLD: float = 0.4
    PRE_OL_TO_MATURE_THRESHOLD: float = 0.8

    def step(
        self,
        abeta_concentration: float = 0.0,
        calcium_level: float = 0.1,
        tnf_alpha: float = 0.0,
        dt: float = 0.01,
    ) -> Dict:
        """Advance one time step.

        Args:
            abeta_concentration: Extracellular Aβ level (≥0)
            calcium_level: Intracellular Ca²⁺ level (≥0)
            tnf_alpha: TNF-α concentration from M1 microglia (≥0)
            dt: Timestep (>0)

        Returns:
            Current cell state dict

        Raises:
            ValueError: If parameters are invalid
        """
        if abeta_concentration < 0:
            raise ValueError(f"abeta_concentration must be non-negative, got {abeta_concentration}")
        if calcium_level < 0:
            raise ValueError(f"calcium_level must be non-negative, got {calcium_level}")
        if tnf_alpha < 0:
            raise ValueError(f"tnf_alpha must be non-negative, got {tnf_alpha}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # --- 1. Myelination capacity: degraded by Aβ exposure ---
        abeta_damage = abeta_concentration * self.ABETA_MYELINATION_COEFF * dt
        noise_myel = random.gauss(0, 0.005 * dt)
        self.myelination_capacity -= abeta_damage + noise_myel
        self.myelination_capacity = max(0.0, min(1.0, self.myelination_capacity))

        # --- 2. Myelin integrity: degraded by elevated Ca²⁺ ---
        ca_damage = calcium_level * self.CALCIUM_INTEGRITY_COEFF * dt
        noise_integ = random.gauss(0, 0.005 * dt)
        self.myelin_integrity -= ca_damage + noise_integ
        self.myelin_integrity = max(0.0, min(1.0, self.myelin_integrity))

        # --- 3. Maturation progression ---
        # Effective maturation rate: base rate suppressed by TNF-α
        tnf_suppression = 1.0 - min(1.0, tnf_alpha * self.TNF_MATURATION_INHIBITION)
        effective_rate = self.BASE_MATURATION_RATE * tnf_suppression * dt
        noise_mat = random.gauss(0, 0.003 * dt)
        self._maturation_progress += effective_rate + noise_mat
        self._maturation_progress = max(0.0, min(1.0, self._maturation_progress))

        # State transitions (unidirectional: OPC → pre-OL → mature OL)
        if (
            self.maturation_state == MaturationState.OPC
            and self._maturation_progress >= self.OPC_TO_PRE_OL_THRESHOLD
        ):
            self.maturation_state = MaturationState.PRE_OL
        elif (
            self.maturation_state == MaturationState.PRE_OL
            and self._maturation_progress >= self.PRE_OL_TO_MATURE_THRESHOLD
        ):
            self.maturation_state = MaturationState.MATURE_OL

        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "maturation_state": self.maturation_state.value,
            "myelination_capacity": round(self.myelination_capacity, 4),
            "myelin_integrity": round(self.myelin_integrity, 4),
            "maturation_progress": round(self._maturation_progress, 4),
        }
