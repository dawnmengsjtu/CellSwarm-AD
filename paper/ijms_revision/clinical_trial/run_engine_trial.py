"""Authoritative IJMS virtual-trial analysis driven by CellSwarm-AD.

Unlike the deprecated illustrative generator in ``run_clinical_trial.py``, this
script obtains every longitudinal outcome from the repository ADCascade and
PK/PD implementations.  The only longitudinal extension is observation noise:
independent Gaussian increments form a Brownian path with SD 0.9 at week 78,
matching the endpoint noise declared by TrialSimulator.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from layer2_pathology.coupling.cascade import ADCascade
from layer2_pathology.drug_models.drug_library import DONEPEZIL, LECANEMAB, MEMANTINE
from layer2_pathology.drug_models.pharmacodynamics import PDModel
from layer2_pathology.drug_models.pharmacokinetics import PKModel, TwoCompartmentPKModel
from layer2_pathology.drug_models.trial_simulator import Patient, TrialArm, TrialSimulator

ARMS = ("placebo", "lecanemab", "donepezil", "donepezil+memantine")
VISITS = (0, 13, 26, 39, 52, 65, 78)
N_PER_ARM = 200
STEPS_PER_WEEK = 9
DT = 0.01
CALIBRATION_K = 0.34
DROPOUT_FRACTION = 0.15


def arm_definitions() -> dict[str, TrialArm]:
    done_pk, mem_pk = PKModel(DONEPEZIL.pk_params), PKModel(MEMANTINE.pk_params)
    return {
        "placebo": TrialArm("placebo", is_placebo=True),
        "lecanemab": TrialArm("lecanemab", TwoCompartmentPKModel(LECANEMAB.pk_params),
                              PDModel(LECANEMAB.pd_params), dosing_interval_h=336.0),
        "donepezil": TrialArm("donepezil", done_pk, PDModel(DONEPEZIL.pd_params),
                              dosing_interval_h=24.0),
        "donepezil+memantine": TrialArm(
            "donepezil+memantine", dosing_interval_h=24.0,
            drugs=[(done_pk, PDModel(DONEPEZIL.pd_params)),
                   (mem_pk, PDModel(MEMANTINE.pd_params))]),
    }


def pd_effect(arm: TrialArm) -> float:
    if arm.is_placebo:
        return 0.0
    if arm.drugs:
        residual = 1.0
        for pk, pd_model in arm.drugs:
            residual *= 1.0 - pd_model.effect(pk.steady_state_concentration(arm.dosing_interval_h))
        return 1.0 - residual
    concentration = arm.pk_model.steady_state_concentration(arm.dosing_interval_h)
    return float(arm.pd_model.effect(concentration))


def dropout_schedule(rng: np.random.Generator) -> np.ndarray:
    first_missing = np.full(N_PER_ARM, 999, dtype=int)
    selected = rng.choice(N_PER_ARM, int(N_PER_ARM * DROPOUT_FRACTION), replace=False)
    first_missing[selected] = rng.choice(VISITS[1:], len(selected), replace=True)
    return first_missing


def simulate_patient_visits(patient: Patient, arm: TrialArm, rng: np.random.Generator,
                            first_missing_week: int) -> list[dict]:
    cascade = ADCascade()
    cascade.state.viability = patient.baseline_viability
    drug = pd_effect(arm)
    noise = 0.0
    previous_fraction = 0.0
    rows = []
    visit_steps = {week * STEPS_PER_WEEK: week for week in VISITS[1:]}

    def add_row(week: int, change: float, viability: float, ptau: float, calcium: float) -> None:
        observed = week < first_missing_week
        rows.append({
            "patient_id": patient.patient_id, "treatment": arm.name,
            "visit_week": week, "observed": observed,
            "first_missing_week": None if first_missing_week == 999 else first_missing_week,
            "age": patient.age, "apoe4_copies": patient.apoe4_copies,
            "baseline_abeta": patient.baseline_abeta,
            "baseline_score": patient.cognitive_score,
            "cognitive_change": change if observed else np.nan,
            "cognitive_score": patient.cognitive_score + change if observed else np.nan,
            "viability": viability, "p_tau": ptau, "calcium": calcium,
        })

    add_row(0, 0.0, cascade.state.viability, cascade.state.p_tau, cascade.state.calcium)
    total_steps = 78 * STEPS_PER_WEEK
    for step in range(1, total_steps + 1):
        abeta = max(0.0, patient.baseline_abeta + rng.normal(0.0, 0.002))
        cascade.step(abeta, DT)
        if step not in visit_steps:
            continue
        week = visit_steps[step]
        fraction = week / 78.0
        noise += rng.normal(0.0, 0.9 * math.sqrt(fraction - previous_fraction))
        previous_fraction = fraction
        pathology = (0.5 * (1.0 - cascade.state.viability)
                     + 0.3 * min(cascade.state.p_tau, 1.0)
                     + 0.2 * min(cascade.state.calcium / 5.0, 1.0))
        pathology *= 1.0 - CALIBRATION_K * drug
        base_decline = 3.5 * (week / 52.0)
        change = -base_decline * (0.05 + 0.95 * pathology) + noise
        add_row(week, float(change), cascade.state.viability,
                cascade.state.p_tau, cascade.state.calcium)
    return rows


def simulate_trial(seed: int) -> pd.DataFrame:
    simulator = TrialSimulator(duration_weeks=78, seed=seed, dropout_rate=0.0)
    patients = simulator.generate_population(N_PER_ARM * len(ARMS), stage="mild")
    simulator.rng.shuffle(patients)
    definitions = arm_definitions()
    rows = []
    for arm_index, arm_name in enumerate(ARMS):
        schedule = dropout_schedule(simulator.rng)
        group = patients[arm_index * N_PER_ARM:(arm_index + 1) * N_PER_ARM]
        for i, patient in enumerate(group):
            patient.patient_id = f"{arm_index + 1}-{i + 1:03d}"
            rows.extend(simulate_patient_visits(patient, definitions[arm_name], simulator.rng,
                                                int(schedule[i])))
    return pd.DataFrame(rows)


def fit_gee(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    observed = data.loc[data.observed].copy()
    observed["treatment"] = pd.Categorical(observed.treatment, categories=ARMS)
    model = smf.gee("cognitive_change ~ baseline_score + C(treatment) * C(visit_week)",
                    groups="patient_id", data=observed, family=sm.families.Gaussian(),
                    cov_struct=sm.cov_struct.Exchangeable())
    result = model.fit(cov_type="robust")
    ci = result.conf_int()
    table = pd.DataFrame({"term": result.params.index, "estimate": result.params.values,
                          "robust_se": result.bse.values, "p_value": result.pvalues.values,
                          "ci95_low": ci[0].values, "ci95_high": ci[1].values})
    endpoint_terms = table[table.term.str.contains("T.78]") & table.term.str.contains(":")].copy()
    metadata = {"model": "Gaussian GEE", "formula": model.formula,
                "cluster": "patient_id", "working_correlation": "exchangeable",
                "robust_se": True, "n_observations": int(result.nobs),
                "estimated_working_correlation": float(result.cov_struct.dep_params)}
    return endpoint_terms, metadata


def endpoint_effects(data: pd.DataFrame, replicate: int, seed: int) -> list[dict]:
    endpoint = data[(data.visit_week == 78) & data.observed]
    placebo = endpoint.loc[endpoint.treatment == "placebo", "cognitive_change"].to_numpy()
    rows = []
    for arm in ARMS[1:]:
        active = endpoint.loc[endpoint.treatment == arm, "cognitive_change"].to_numpy()
        pooled = math.sqrt(((len(placebo)-1)*placebo.var(ddof=1)+(len(active)-1)*active.var(ddof=1)) /
                           (len(placebo)+len(active)-2))
        rows.append({"replicate": replicate, "seed": seed, "treatment": arm,
                     "n_randomized_per_arm": N_PER_ARM, "n_week78_placebo": len(placebo),
                     "n_week78_active": len(active), "placebo_mean": placebo.mean(),
                     "active_mean": active.mean(), "difference": active.mean()-placebo.mean(),
                     "cohens_d": (active.mean()-placebo.mean())/pooled})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=HERE / "engine_results")
    args = parser.parse_args()
    if args.replicates < 20:
        parser.error("at least 20 independent replicates are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary = simulate_trial(args.seed)
    gee, gee_metadata = fit_gee(primary)
    monte_rows = []
    for rep in range(args.replicates):
        rep_seed = args.seed + 10_000 + rep
        monte_rows.extend(endpoint_effects(simulate_trial(rep_seed), rep + 1, rep_seed))
    monte = pd.DataFrame(monte_rows)
    summary = monte.groupby("treatment").agg(
        n_replicates=("replicate", "count"), mean_d=("cohens_d", "mean"),
        sd_d=("cohens_d", "std"), q025_d=("cohens_d", lambda x: x.quantile(.025)),
        q975_d=("cohens_d", lambda x: x.quantile(.975)),
        mean_difference=("difference", "mean")).reset_index()
    primary.to_csv(args.output_dir / "primary_patient_visits.csv", index=False)
    gee.to_csv(args.output_dir / "primary_gee_week78_terms.csv", index=False)
    monte.to_csv(args.output_dir / "monte_carlo_replicates.csv", index=False)
    summary.to_csv(args.output_dir / "monte_carlo_summary.csv", index=False)
    metadata = {
        "authority": "CellSwarm-AD ADCascade plus repository PK/PD models",
        "primary_trial": {"seed": args.seed, "n_per_arm": N_PER_ARM, "n_total": 800,
                          "visits": list(VISITS), "dropout": "exact 15% MCAR per arm"},
        "cognitive_mapping": {"source": "TrialSimulator._simulate_patient",
                              "calibration_k": CALIBRATION_K,
                              "observation_noise": "Gaussian independent increments; week-78 SD=0.9"},
        "combination": {"drugs": ["donepezil", "memantine"],
                        "rule": "Bliss independence, explicit structural assumption",
                        "pd_effect": pd_effect(arm_definitions()["donepezil+memantine"])},
        "gee": gee_metadata,
        "monte_carlo": {"replicates": args.replicates, "pooled_as_one_trial": False},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
