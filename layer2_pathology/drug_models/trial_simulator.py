# -*- coding: utf-8 -*-
"""Clinical trial simulator for AD drug interventions.

Simulates virtual clinical trials with:
- Patient population with heterogeneous parameters
- Treatment arms (drug, placebo, combination)
- Cognitive decline trajectories using CellSwarm-AD cascade
- Statistical analysis (t-test, effect size)
- Stage-specific baseline MMSE
- Realistic effect sizes (Cohen's d ~ 0.2-0.5)
- Randomized allocation from a single population
- Dropout simulation
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import math

import numpy as np

from ..coupling.cascade import ADCascade
from .pharmacokinetics import PKModel, PKParameters, TwoCompartmentPKModel, TwoCompartmentPKParameters
from .pharmacodynamics import PDModel


# Stage-specific MMSE distributions (mean, std)
STAGE_MMSE = {
    "mild": (24.0, 2.0),       # MMSE 20-28
    "moderate": (17.0, 2.5),   # MMSE 10-24
    "severe": (8.0, 3.0),      # MMSE 0-14
}


@dataclass
class Patient:
    """Virtual patient with individual parameters."""
    patient_id: int
    age: float
    apoe4_copies: int = 0          # 0, 1, or 2
    baseline_abeta: float = 0.5    # baseline Aβ oligomer level
    baseline_viability: float = 1.0
    disease_stage: str = "mild"    # mild, moderate, severe
    cognitive_score: float = 26.0  # MMSE-like score (0-30)

    def __post_init__(self):
        # APOE4 increases Aβ burden
        self.baseline_abeta *= (1.0 + 0.3 * self.apoe4_copies)
        # Disease stage affects baseline
        stage_map = {"mild": 1.0, "moderate": 1.3, "severe": 1.6}
        self.baseline_abeta *= stage_map.get(self.disease_stage, 1.0)


@dataclass
class TrialArm:
    """Treatment arm in a clinical trial."""
    name: str
    pk_model: Optional[Union[PKModel, TwoCompartmentPKModel]] = None
    pd_model: Optional[PDModel] = None
    is_placebo: bool = False
    dosing_interval_h: float = 24.0  # hours between doses
    # Multi-drug combination: list of (pk_model, pd_model) tuples.
    # When non-empty, Bliss independence is applied across all drugs.
    # Takes precedence over the single pk_model/pd_model fields.
    drugs: list = field(default_factory=list)


@dataclass
class TrialResult:
    """Results from a simulated clinical trial."""
    arm_name: str
    n_patients: int
    mean_cognitive_change: float
    std_cognitive_change: float
    mean_viability: float
    cognitive_scores: List[float] = field(default_factory=list)
    viability_scores: List[float] = field(default_factory=list)
    n_dropouts: int = 0

    def to_dict(self) -> Dict:
        return {
            "arm": self.arm_name,
            "n": self.n_patients,
            "mean_change": round(self.mean_cognitive_change, 3),
            "std_change": round(self.std_cognitive_change, 3),
            "mean_viability": round(self.mean_viability, 4),
            "n_dropouts": self.n_dropouts,
        }


class TrialSimulator:
    """Simulate virtual AD clinical trials.

    Uses CellSwarm-AD cascade to model disease progression
    and drug effects on each virtual patient.
    """

    def __init__(self, duration_weeks: int = 78, seed: int = 42,
                 dropout_rate: float = 0.15):
        self.duration_weeks = duration_weeks
        self.rng = np.random.default_rng(seed)
        self.dropout_rate = dropout_rate  # fraction over full trial duration

    def generate_population(
        self,
        n: int,
        age_range: Tuple[float, float] = (55, 85),
        apoe4_freq: float = 0.25,
        stage: str = "mild",
        stage_distribution: Optional[Dict[str, float]] = None,
    ) -> List[Patient]:
        """Generate a virtual patient population.

        Args:
            n: Number of patients.
            age_range: (min_age, max_age).
            apoe4_freq: APOE4 allele frequency.
            stage: Default disease stage (used when stage_distribution is None).
            stage_distribution: Optional dict mapping stage names to proportions,
                e.g. {"mild": 0.6, "moderate": 0.3, "severe": 0.1}.
        """
        if stage_distribution is None:
            stage_distribution = {stage: 1.0}

        # Build cumulative distribution for stage sampling
        stages = list(stage_distribution.keys())
        probs = np.array([stage_distribution[s] for s in stages], dtype=float)
        probs /= probs.sum()  # normalise

        patients = []
        for i in range(n):
            # Sample stage
            pat_stage = self.rng.choice(stages, p=probs)

            age = self.rng.uniform(*age_range)
            apoe4 = int(self.rng.binomial(2, apoe4_freq))

            # Stage-specific baseline MMSE
            mean, std = STAGE_MMSE.get(pat_stage, (24.0, 2.0))
            baseline_mmse = self.rng.normal(mean, std)
            baseline_mmse = max(0.0, min(30.0, baseline_mmse))

            patients.append(Patient(
                patient_id=i,
                age=age,
                apoe4_copies=apoe4,
                disease_stage=pat_stage,
                cognitive_score=baseline_mmse,
            ))
        return patients

    def _simulate_patient(
        self,
        patient: Patient,
        arm: TrialArm,
        steps_per_week: int = 9,
    ) -> Tuple[float, float, bool]:
        """Simulate one patient through the trial.

        Returns:
            (cognitive_change, final_viability, dropped_out)
        """
        cascade = ADCascade()
        cascade.state.viability = patient.baseline_viability

        total_steps = self.duration_weeks * steps_per_week
        dt = 0.01

        # Pre-compute drug effect (steady-state approximation)
        drug_effect = 0.0
        if not arm.is_placebo:
            if arm.drugs:
                # Multi-drug arm: Bliss independence
                # E_combo = 1 - prod(1 - Ei)  for each drug i
                survival = 1.0
                for pk_m, pd_m in arm.drugs:
                    css_i = pk_m.steady_state_concentration(arm.dosing_interval_h)
                    ei = pd_m.effect(css_i)
                    survival *= (1.0 - ei)
                drug_effect = 1.0 - survival
            elif arm.pk_model and arm.pd_model:
                # Single-drug arm (backward compatible)
                css = arm.pk_model.steady_state_concentration(arm.dosing_interval_h)
                drug_effect = arm.pd_model.effect(css)

        # Per-step dropout probability
        dropout_prob_per_step = self.dropout_rate / total_steps if total_steps > 0 else 0.0
        dropped_out = False
        completed_steps = total_steps

        for step in range(total_steps):
            # Dropout check
            if self.dropout_rate > 0 and self.rng.random() < dropout_prob_per_step:
                dropped_out = True
                completed_steps = step
                break

            abeta = patient.baseline_abeta

            # Add patient-level noise (small)
            noise = self.rng.normal(0, 0.002)
            abeta = max(0, abeta + noise)

            cascade.step(abeta, dt)

        # Map viability to cognitive decline
        viability_loss = 1.0 - cascade.state.viability
        ptau_burden = min(cascade.state.p_tau, 1.0)

        # Scale by fraction of trial actually completed
        fraction_completed = completed_steps / total_steps if total_steps > 0 else 1.0

        # MMSE decline model calibrated to clinical data:
        # Mild AD placebo: ~3-4 points/year
        years = self.duration_weeks / 52.0
        base_decline = 3.5 * years * fraction_completed

        # Pathology score: weighted combination of viability loss, p-tau, and calcium
        # Higher calcium is pathological in AD (excitotoxicity)
        calcium_norm = min(cascade.state.calcium / 5.0, 1.0)
        pathology_score = (0.5 * viability_loss
                          + 0.3 * ptau_burden
                          + 0.2 * calcium_norm)

        # Drug effect: reduce pathology score proportionally
        # This models the net clinical benefit without perturbing cascade dynamics
        # (avoids paradoxical M2 microglia switching from abeta reduction)
        #
        # Calibration constant k = 0.34, derived from Clarity AD trial data:
        #   van Dyck et al., NEJM 2023 (lecanemab, N=1795, 78 weeks)
        #   Placebo CDR-SB decline: 1.66 pts; Lecanemab: 1.21 pts -> 27.1% slowing
        #   Lecanemab PD effect E = 0.888 at therapeutic Css (Gibiansky et al.,
        #     Clin Pharmacol Ther 2023: EC50=3400 ng/mL, Css=27000 ng/mL, Emax model)
        #   Mean pathology score PS_0 ~ 0.45 for mild AD in this population
        #   k = slowing * (0.05 + 0.95*PS_0) / (E * 0.95 * PS_0)
        #     = 0.271 * 0.4775 / (0.888 * 0.4275) = 0.34
        CALIBRATION_K = 0.34  # dimensionless; derived from Clarity AD RCT
        if drug_effect > 0:
            pathology_score *= (1.0 - drug_effect * CALIBRATION_K)

        cognitive_change = -base_decline * (0.05 + 0.95 * pathology_score)

        # Realistic inter-patient noise (calibrated for Cohen's d ~ 0.2-0.4)
        cognitive_change += self.rng.normal(0, 0.9)

        return (cognitive_change, cascade.state.viability, dropped_out)

    def run_arm(
        self,
        patients: List[Patient],
        arm: TrialArm,
    ) -> TrialResult:
        """Run one treatment arm."""
        cognitive_changes = []
        viabilities = []
        n_dropouts = 0

        for patient in patients:
            cog_change, viability, dropped = self._simulate_patient(patient, arm)
            cognitive_changes.append(cog_change)
            viabilities.append(viability)
            if dropped:
                n_dropouts += 1

        return TrialResult(
            arm_name=arm.name,
            n_patients=len(patients),
            mean_cognitive_change=float(np.mean(cognitive_changes)),
            std_cognitive_change=float(np.std(cognitive_changes)),
            mean_viability=float(np.mean(viabilities)),
            cognitive_scores=cognitive_changes,
            viability_scores=viabilities,
            n_dropouts=n_dropouts,
        )

    def run_trial(
        self,
        arms: List[TrialArm],
        n_per_arm: int = 50,
        **pop_kwargs,
    ) -> Dict[str, TrialResult]:
        """Run a complete trial with multiple arms.

        Generates a single pooled population and randomly allocates
        patients to arms (proper randomisation).

        Args:
            arms: List of treatment arms
            n_per_arm: Patients per arm
            **pop_kwargs: Passed to generate_population

        Returns:
            Dict mapping arm name to TrialResult
        """
        total_n = n_per_arm * len(arms)
        all_patients = self.generate_population(total_n, **pop_kwargs)
        self.rng.shuffle(all_patients)

        results = {}
        for i, arm in enumerate(arms):
            patients = all_patients[i * n_per_arm : (i + 1) * n_per_arm]
            results[arm.name] = self.run_arm(patients, arm)
        return results

    @staticmethod
    def compare_arms(result_a: TrialResult, result_b: TrialResult) -> Dict:
        """Compare two trial arms (basic t-test approximation).

        Returns:
            Dict with effect size, mean difference, and significance estimate
        """
        diff = result_a.mean_cognitive_change - result_b.mean_cognitive_change
        pooled_std = math.sqrt(
            (result_a.std_cognitive_change ** 2 + result_b.std_cognitive_change ** 2) / 2
        )
        cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

        # Approximate t-statistic
        se = pooled_std * math.sqrt(1 / result_a.n_patients + 1 / result_b.n_patients)
        t_stat = diff / se if se > 0 else 0.0

        return {
            "arm_a": result_a.arm_name,
            "arm_b": result_b.arm_name,
            "mean_diff": round(diff, 3),
            "cohens_d": round(cohens_d, 3),
            "t_statistic": round(t_stat, 3),
            "significant_approx": abs(t_stat) > 1.96,
        }
