# -*- coding: utf-8 -*-
"""Endothelial (BBB) agent with barrier integrity and Aβ transport.

AD relevance:
- Blood-brain barrier breakdown is an early biomarker of cognitive decline
  (Nation et al., 2019, Nat Med)
- LRP1-mediated Aβ efflux across BBB is impaired in AD
  (Deane et al., 2009, J Clin Invest)
- RAGE mediates Aβ influx into the brain, upregulated in AD
  (Deane et al., 2003, Nat Med)
- TNF-α disrupts tight junctions (claudin-5, occludin) via NF-κB
  (Aslam et al., 2012, J Cell Sci)
- IL-10 is protective, partially restoring barrier function
  (Mazzon et al., 2006, Shock)
- Aβ deposits in cerebral vasculature (CAA) weaken tight junctions
  (Zipfel et al., 2009, Stroke)
"""
from dataclasses import dataclass
from typing import Dict
import random


@dataclass
class EndothelialAgent:
    """Simulates brain endothelial cell (BBB unit) in AD context.

    BBB integrity is maintained by tight junctions and modulated by
    inflammation. Aβ clearance across BBB depends on barrier health
    (LRP1 efflux dominant when intact; RAGE influx when compromised).
    TNF-α weakens tight junctions; IL-10 is partially protective.
    """
    agent_id: str = "EC_0"
    bbb_integrity: float = 1.0            # overall barrier function
    tight_junction_strength: float = 1.0  # claudin-5/occludin level
    abeta_transport_rate: float = 0.5     # net Aβ efflux rate (positive = clearance)

    # --- Biologically-motivated parameters ---
    # Aβ effect on tight junctions: ~4% per unit Aβ per dt
    #   (Zipfel et al., 2009 — CAA weakens junctions dose-dependently)
    ABETA_TJ_COEFF: float = 0.04
    # TNF-α effect on BBB integrity: ~10% per unit TNF-α per dt
    #   (Aslam et al., 2012 — TNF-α opens paracellular pathway)
    TNF_INTEGRITY_COEFF: float = 0.10
    # IL-10 protective effect on BBB integrity: ~6% recovery per unit IL-10 per dt
    #   (Mazzon et al., 2006 — IL-10 partially restores barrier)
    IL10_PROTECTION_COEFF: float = 0.06
    # Natural tight junction repair rate (slow turnover, ~1% per dt)
    TJ_REPAIR_RATE: float = 0.01
    # LRP1 efflux baseline (healthy BBB clears Aβ efficiently)
    #   (Deane et al., 2009 — LRP1 is primary Aβ efflux receptor)
    LRP1_BASELINE: float = 0.6
    # RAGE influx contribution (increases when BBB is compromised)
    #   (Deane et al., 2003 — RAGE transports Aβ into brain)
    RAGE_INFLUX_COEFF: float = 0.3

    def step(
        self,
        abeta_concentration: float = 0.0,
        tnf_alpha: float = 0.0,
        il10: float = 0.0,
        dt: float = 0.01,
    ) -> Dict:
        """Advance one time step.

        Args:
            abeta_concentration: Parenchymal Aβ level (≥0)
            tnf_alpha: TNF-α concentration (≥0)
            il10: IL-10 concentration (≥0, anti-inflammatory)
            dt: Timestep (>0)

        Returns:
            Current cell state dict

        Raises:
            ValueError: If parameters are invalid
        """
        if abeta_concentration < 0:
            raise ValueError(f"abeta_concentration must be non-negative, got {abeta_concentration}")
        if tnf_alpha < 0:
            raise ValueError(f"tnf_alpha must be non-negative, got {tnf_alpha}")
        if il10 < 0:
            raise ValueError(f"il10 must be non-negative, got {il10}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # --- 1. Tight junction dynamics ---
        # Aβ deposition weakens tight junctions
        abeta_damage = abeta_concentration * self.ABETA_TJ_COEFF * dt
        # Slow natural repair
        repair = self.TJ_REPAIR_RATE * (1.0 - self.tight_junction_strength) * dt
        noise_tj = random.gauss(0, 0.005 * dt)

        self.tight_junction_strength += repair - abeta_damage + noise_tj
        self.tight_junction_strength = max(0.0, min(1.0, self.tight_junction_strength))

        # --- 2. BBB integrity ---
        # TNF-α degrades integrity; IL-10 partially restores it
        tnf_damage = tnf_alpha * self.TNF_INTEGRITY_COEFF * dt
        il10_recovery = il10 * self.IL10_PROTECTION_COEFF * (1.0 - self.bbb_integrity) * dt
        noise_bbb = random.gauss(0, 0.005 * dt)

        # Tight junction strength is a major contributor to overall integrity
        tj_pull = (self.tight_junction_strength - self.bbb_integrity) * 0.05 * dt

        self.bbb_integrity += tj_pull + il10_recovery - tnf_damage + noise_bbb
        self.bbb_integrity = max(0.0, min(1.0, self.bbb_integrity))

        # --- 3. Aβ transport rate ---
        # Intact BBB → LRP1-dominant efflux (clearance)
        # Compromised BBB → RAGE-dominant influx (Aβ entry)
        lrp1_efflux = self.LRP1_BASELINE * self.bbb_integrity
        rage_influx = self.RAGE_INFLUX_COEFF * (1.0 - self.bbb_integrity)
        noise_transport = random.gauss(0, 0.01 * dt)

        # Net transport: positive = clearance, negative = Aβ entering brain
        self.abeta_transport_rate = lrp1_efflux - rage_influx + noise_transport
        self.abeta_transport_rate = max(-1.0, min(1.0, self.abeta_transport_rate))

        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "bbb_integrity": round(self.bbb_integrity, 4),
            "tight_junction_strength": round(self.tight_junction_strength, 4),
            "abeta_transport_rate": round(self.abeta_transport_rate, 4),
        }
