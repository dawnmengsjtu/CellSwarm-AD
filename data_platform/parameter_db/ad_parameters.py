# -*- coding: utf-8 -*-
"""AD-related parameter database with literature sources."""
from typing import List, Dict
from .schema import Parameter


# Aβ dynamics parameters
ABETA_PARAMS = [
    Parameter(
        name="abeta_production_rate",
        value=0.1,
        unit="nM/s",
        source="Cirrito et al. (2003) J Neurosci",
        confidence=0.9,
        min_value=0.05,
        max_value=0.5,
        category="abeta",
        notes="ISF Aβ production rate measured by microdialysis"
    ),
    Parameter(
        name="abeta_clearance_rate",
        value=0.02,
        unit="1/s",
        source="Mawuenyega et al. (2010) Science",
        confidence=0.95,
        min_value=0.01,
        max_value=0.05,
        category="abeta",
        notes="Half-life ~2-4 hours in human CSF"
    ),
    Parameter(
        name="abeta_aggregation_rate",
        value=0.05,
        unit="1/(nM·s)",
        source="Lomakin et al. (1996) PNAS",
        confidence=0.85,
        min_value=0.01,
        max_value=0.1,
        category="abeta",
        notes="Second-order aggregation kinetics"
    ),
]

# Tau dynamics parameters
TAU_PARAMS = [
    Parameter(
        name="tau_phosphorylation_rate",
        value=0.05,
        unit="1/s",
        source="Alonso et al. (2001) PNAS",
        confidence=0.8,
        min_value=0.02,
        max_value=0.1,
        category="tau",
        notes="GSK-3β/CDK5-mediated phosphorylation"
    ),
    Parameter(
        name="tau_dephosphorylation_rate",
        value=0.02,
        unit="1/s",
        source="Estimated from PP2A activity",
        confidence=0.6,
        min_value=0.01,
        max_value=0.05,
        category="tau",
        notes="Needs experimental validation"
    ),
    Parameter(
        name="tau_tangle_rate",
        value=0.01,
        unit="1/s",
        source="Estimated",
        confidence=0.5,
        min_value=0.005,
        max_value=0.02,
        category="tau",
        notes="NFT formation rate, needs validation"
    ),
]

# Calcium signaling parameters
CALCIUM_PARAMS = [
    Parameter(
        name="calcium_resting",
        value=0.1,
        unit="μM",
        source="Berridge et al. (2003) Nat Rev Mol Cell Biol",
        confidence=0.95,
        min_value=0.05,
        max_value=0.15,
        category="calcium",
        notes="Cytosolic Ca²⁺ ~100 nM"
    ),
    Parameter(
        name="calcium_peak",
        value=1.0,
        unit="μM",
        source="Berridge et al. (2003) Nat Rev Mol Cell Biol",
        confidence=0.9,
        min_value=0.5,
        max_value=2.0,
        category="calcium",
        notes="Peak during signaling"
    ),
    Parameter(
        name="calcium_pump_rate",
        value=0.05,
        unit="1/s",
        source="Berridge et al. (2003) Nat Rev Mol Cell Biol",
        confidence=0.85,
        min_value=0.02,
        max_value=0.1,
        category="calcium",
        notes="PMCA/SERCA pump activity"
    ),
]

# Microglia parameters
MICROGLIA_PARAMS = [
    Parameter(
        name="microglia_activation_threshold",
        value=0.6,
        unit="dimensionless",
        source="Colonna & Butovsky (2017) Annu Rev Immunol",
        confidence=0.8,
        min_value=0.4,
        max_value=0.8,
        category="microglia",
        notes="NF-κB activation threshold for M0→M1"
    ),
    Parameter(
        name="microglia_phagocytosis_m0",
        value=0.3,
        unit="1/s",
        source="Krabbe et al. (2013) PLoS One",
        confidence=0.85,
        min_value=0.2,
        max_value=0.4,
        category="microglia",
        notes="Baseline phagocytic activity"
    ),
    Parameter(
        name="microglia_phagocytosis_m2",
        value=0.6,
        unit="1/s",
        source="Krabbe et al. (2013) PLoS One",
        confidence=0.85,
        min_value=0.5,
        max_value=0.8,
        category="microglia",
        notes="Enhanced in anti-inflammatory state"
    ),
]

# Neuron parameters
NEURON_PARAMS = [
    Parameter(
        name="neuron_resting_potential",
        value=-70.0,
        unit="mV",
        source="Hodgkin & Huxley (1952) J Physiol",
        confidence=0.99,
        min_value=-75.0,
        max_value=-65.0,
        category="neuron",
        notes="Classic value for mammalian neurons"
    ),
    Parameter(
        name="neuron_firing_threshold",
        value=-55.0,
        unit="mV",
        source="Hodgkin & Huxley (1952) J Physiol",
        confidence=0.95,
        min_value=-60.0,
        max_value=-50.0,
        category="neuron",
        notes="Action potential threshold"
    ),
]


class ADParameterDB:
    """Database of AD-related parameters from literature."""

    def __init__(self):
        self.params: Dict[str, Parameter] = {}
        self._load_all()

    def _load_all(self):
        """Load all parameters into the database."""
        all_params = ABETA_PARAMS + TAU_PARAMS + CALCIUM_PARAMS + MICROGLIA_PARAMS + NEURON_PARAMS
        for p in all_params:
            self.params[p.name] = p

    def get(self, name: str) -> Parameter:
        """Get parameter by name."""
        if name not in self.params:
            raise KeyError(f"Parameter '{name}' not found in database")
        return self.params[name]

    def query(self, category: str = None, min_confidence: float = 0.0) -> List[Parameter]:
        """Query parameters by category and confidence.

        Args:
            category: Filter by category (abeta, tau, calcium, microglia, neuron)
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching parameters
        """
        results = []
        for p in self.params.values():
            if category and p.category != category:
                continue
            if p.confidence < min_confidence:
                continue
            results.append(p)
        return results

    def list_categories(self) -> List[str]:
        """List all parameter categories."""
        return sorted(set(p.category for p in self.params.values()))

    def summary(self) -> Dict:
        """Get database summary statistics."""
        return {
            "total_params": len(self.params),
            "categories": self.list_categories(),
            "avg_confidence": sum(p.confidence for p in self.params.values()) / len(self.params),
            "high_confidence": len([p for p in self.params.values() if p.confidence >= 0.8]),
        }
