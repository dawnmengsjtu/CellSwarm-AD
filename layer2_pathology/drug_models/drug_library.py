# -*- coding: utf-8 -*-
"""AD drug library with real clinical parameters.

Four FDA-approved/pipeline drugs:
- Aducanumab (Aduhelm): anti-Aβ monoclonal antibody
- Lecanemab (Leqembi): anti-Aβ monoclonal antibody
- Donepezil (Aricept): AChE inhibitor
- Memantine (Namenda): NMDA receptor antagonist
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .pharmacokinetics import (
    PKParameters,
    RouteOfAdministration,
    TwoCompartmentPKParameters,
    TwoCompartmentPKModel,
)
from .pharmacodynamics import PDParameters, DrugType, InhibitionType


@dataclass
class Drug:
    """Complete drug profile with PK and PD parameters."""
    name: str
    generic_name: str
    brand_name: str
    mechanism: str
    target: str                    # molecular target
    pk_params: Union[PKParameters, TwoCompartmentPKParameters]
    pd_params: PDParameters
    approved: bool = True
    year_approved: int = 0
    clinical_dose: str = ""        # typical clinical dosing
    notes: str = ""

    def summary(self) -> Dict:
        pk = self.pk_params
        if isinstance(pk, TwoCompartmentPKParameters):
            half_life_h = round(pk.terminal_half_life, 1)
        else:
            half_life_h = round(0.693 / pk.ke, 1)
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "target": self.target,
            "approved": self.approved,
            "clinical_dose": self.clinical_dose,
            "half_life_h": half_life_h,
            "EC50": self.pd_params.EC50,
        }


# --- Real drug definitions ---

ADUCANUMAB = Drug(
    name="Aducanumab",
    generic_name="aducanumab-avwa",
    brand_name="Aduhelm",
    mechanism="Anti-Aβ monoclonal antibody (targets aggregated Aβ)",
    target="Aβ plaques",
    pk_params=TwoCompartmentPKParameters(
        dose=700.0,         # mg (10 mg/kg × 70 kg), IV infusion
        Vc=3.1,             # L (central compartment)
        Vp=2.8,             # L (peripheral compartment)
        ke=0.017798,        # 1/h — derived from t½α≈1.5d, t½β≈21d
        k12=0.001344,       # 1/h — central→peripheral
        k21=0.001488,       # 1/h — peripheral→central
        ka=0.0,             # IV, no absorption phase
        F=1.0,
        route=RouteOfAdministration.IV_INFUSION,
        t_infusion=1.0,     # 1-hour infusion
    ),
    pd_params=PDParameters(
        Emax=0.7,           # max 70% plaque reduction (EMERGE trial)
        EC50=1.0,           # μg/mL — adjusted for effective brain exposure
        n=1.5,
        drug_type=DrugType.ANTAGONIST,
    ),
    approved=True,
    year_approved=2021,
    clinical_dose="10 mg/kg IV q4w (after titration)",
    notes="Accelerated approval 2021, controversial efficacy. ARIA risk ~35%."
)

LECANEMAB = Drug(
    name="Lecanemab",
    generic_name="lecanemab-irmb",
    brand_name="Leqembi",
    mechanism="Anti-Aβ monoclonal antibody (targets soluble protofibrils)",
    target="Aβ protofibrils",
    pk_params=TwoCompartmentPKParameters(
        dose=700.0,         # mg (10 mg/kg × 70 kg), IV infusion
        Vc=3.0,             # L (central compartment)
        Vp=2.5,             # L (peripheral compartment)
        ke=0.022229,        # 1/h — derived from t½α≈1d, t½β≈6d
        k12=0.005212,       # 1/h — central→peripheral
        k21=0.006254,       # 1/h — peripheral→central
        ka=0.0,
        F=1.0,
        route=RouteOfAdministration.IV_INFUSION,
        t_infusion=1.0,
    ),
    pd_params=PDParameters(
        Emax=0.59,          # 59% plaque reduction (Clarity AD)
        EC50=0.8,           # μg/mL — effective brain concentration lower than plasma
        n=1.2,
        drug_type=DrugType.ANTAGONIST,
    ),
    approved=True,
    year_approved=2023,
    clinical_dose="10 mg/kg IV q2w",
    notes="Full approval 2023. 27% slowing of cognitive decline (Clarity AD). ARIA ~13%."
)

DONEPEZIL = Drug(
    name="Donepezil",
    generic_name="donepezil hydrochloride",
    brand_name="Aricept",
    mechanism="Reversible AChE inhibitor (increases synaptic ACh)",
    target="Acetylcholinesterase (AChE)",
    pk_params=PKParameters(
        dose=10.0,          # mg oral
        Vd=840.0,           # L (~12 L/kg)
        ke=0.0099,          # t1/2 ~70h → ke = ln2/70 = 0.0099
        ka=1.5,             # oral absorption, Tmax ~3h requires higher ka
        F=1.0,              # ~100% bioavailability
        route=RouteOfAdministration.ORAL,
    ),
    pd_params=PDParameters(
        Emax=0.8,           # max 80% AChE inhibition
        EC50=0.015,         # μg/mL (15 ng/mL)
        n=1.0,
        drug_type=DrugType.ANTAGONIST,
        inhibition_type=InhibitionType.COMPETITIVE,
        Ki=0.012,
    ),
    approved=True,
    year_approved=1996,
    clinical_dose="5-10 mg PO daily",
    notes="First-line symptomatic treatment. Modest cognitive benefit (2-4 ADAS-cog points)."
)

MEMANTINE = Drug(
    name="Memantine",
    generic_name="memantine hydrochloride",
    brand_name="Namenda",
    mechanism="Uncompetitive NMDA receptor antagonist (reduces excitotoxicity)",
    target="NMDA receptor",
    pk_params=PKParameters(
        dose=20.0,          # mg oral (target dose)
        Vd=630.0,           # L (~9-11 L/kg)
        ke=0.011,           # t1/2 ~60-80h
        ka=0.8,
        F=1.0,              # ~100% bioavailability
        route=RouteOfAdministration.ORAL,
    ),
    pd_params=PDParameters(
        Emax=0.6,           # moderate NMDA block
        EC50=0.5,           # μg/mL
        n=1.0,
        drug_type=DrugType.ANTAGONIST,
        inhibition_type=InhibitionType.UNCOMPETITIVE,
        Ki=0.7,
    ),
    approved=True,
    year_approved=2003,
    clinical_dose="10 mg PO BID or 28 mg XR daily",
    notes="Moderate-to-severe AD. Often combined with Donepezil."
)


# --- Drug Library ---

DRUG_LIBRARY: Dict[str, Drug] = {
    "aducanumab": ADUCANUMAB,
    "lecanemab": LECANEMAB,
    "donepezil": DONEPEZIL,
    "memantine": MEMANTINE,
}


class DrugLibrary:
    """Searchable drug library."""

    def __init__(self):
        self.drugs = dict(DRUG_LIBRARY)

    def get(self, name: str) -> Drug:
        key = name.lower()
        if key not in self.drugs:
            raise KeyError(f"Drug '{name}' not found. Available: {list(self.drugs.keys())}")
        return self.drugs[key]

    def list_drugs(self) -> List[str]:
        return list(self.drugs.keys())

    def add(self, drug: Drug) -> None:
        self.drugs[drug.name.lower()] = drug

    def by_mechanism(self, keyword: str) -> List[Drug]:
        return [d for d in self.drugs.values() if keyword.lower() in d.mechanism.lower()]

    def by_target(self, target: str) -> List[Drug]:
        return [d for d in self.drugs.values() if target.lower() in d.target.lower()]


def get_drug(name: str) -> Drug:
    """Convenience function to get a drug by name."""
    return DRUG_LIBRARY[name.lower()]
