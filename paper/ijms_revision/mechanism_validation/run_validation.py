"""Reproducible IJMS mechanism-validation analyses.

All outputs are written below this file's directory.  The implementation imports
the model but does not modify or monkey-patch repository source files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from layer2_pathology.coupling.cascade import ADCascade
from layer2_pathology.drug_models.drug_library import DONEPEZIL, LECANEMAB, MEMANTINE
from layer2_pathology.drug_models.pharmacodynamics import PDModel
from layer2_pathology.drug_models.pharmacokinetics import PKModel, TwoCompartmentPKModel
from layer2_pathology.drug_models.trial_simulator import Patient, TrialArm, TrialSimulator

RAW = HERE / "raw"
SUMMARY = HERE / "summary"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def arms() -> dict[str, TrialArm]:
    lec_pk = TwoCompartmentPKModel(LECANEMAB.pk_params)
    done_pk, mem_pk = PKModel(DONEPEZIL.pk_params), PKModel(MEMANTINE.pk_params)
    return {
        "placebo": TrialArm("placebo", is_placebo=True),
        "lecanemab": TrialArm("lecanemab", lec_pk, PDModel(LECANEMAB.pd_params),
                              dosing_interval_h=336.0),
        "donepezil_memantine": TrialArm(
            "donepezil_memantine",
            drugs=[(done_pk, PDModel(DONEPEZIL.pd_params)),
                   (mem_pk, PDModel(MEMANTINE.pd_params))],
            dosing_interval_h=24.0,
        ),
    }


def hc3_ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> list[dict]:
    """OLS with HC3 covariance; t reference uses residual degrees of freedom."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    xtx_inv = np.linalg.inv(x.T @ x)
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    adjusted = resid / np.maximum(1.0 - leverage, 1e-12)
    meat = x.T @ (x * adjusted[:, None] ** 2)
    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(covariance))
    df = len(y) - x.shape[1]
    critical = stats.t.ppf(0.975, df)
    out = []
    for name, estimate, stderr in zip(names, beta, se):
        t_value = estimate / stderr
        out.append({
            "term": name, "estimate": float(estimate), "std_error_hc3": float(stderr),
            "ci95_low": float(estimate - critical * stderr),
            "ci95_high": float(estimate + critical * stderr),
            "t": float(t_value), "df": int(df),
            "p_value": float(2 * stats.t.sf(abs(t_value), df)),
        })
    return out


def run_apoe4(n_per_arm: int, seed: int) -> None:
    sim = TrialSimulator(duration_weeks=78, seed=seed, dropout_rate=0.15)
    population = sim.generate_population(2 * n_per_arm, stage="mild")
    sim.rng.shuffle(population)
    arm_defs = arms()
    rows: list[dict] = []
    for treatment, patients in (("placebo", population[:n_per_arm]),
                                ("lecanemab", population[n_per_arm:])):
        arm = arm_defs[treatment]
        for patient in patients:
            change, viability, dropout = sim._simulate_patient(patient, arm)
            rows.append({
                "patient_id": f"{treatment}_{patient.patient_id}", "treatment": treatment,
                "treatment_lecanemab": int(treatment == "lecanemab"),
                "apoe4_copies": patient.apoe4_copies, "age": patient.age,
                "baseline_cognitive_score": patient.cognitive_score,
                "baseline_abeta": patient.baseline_abeta,
                # The simulator returns change at the patient's final observed
                # time.  It is a week-78 outcome only for trial completers.
                "observed_change_at_exit": change,
                "week78_change": None if dropout else change,
                "final_viability": viability, "dropped_out": int(dropout),
                "completed_week78": int(not dropout),
            })
    write_csv(RAW / "apoe4_patient_level.csv", rows)
    analysis_rows = [r for r in rows if r["completed_week78"] == 1]
    y = np.array([r["week78_change"] for r in analysis_rows], float)
    trt = np.array([r["treatment_lecanemab"] for r in analysis_rows], float)
    apoe = np.array([r["apoe4_copies"] for r in analysis_rows], float)
    baseline = np.array([r["baseline_cognitive_score"] for r in analysis_rows], float)
    age = np.array([r["age"] for r in analysis_rows], float)
    x = np.column_stack([np.ones(len(analysis_rows)), trt, apoe, trt * apoe,
                         baseline - baseline.mean(), age - age.mean()])
    model = hc3_ols(y, x, ["intercept", "lecanemab", "apoe4_copies",
                           "lecanemab_x_apoe4_copies", "baseline_cognitive_centered",
                           "age_centered"])
    counts = {}
    for treatment in ("placebo", "lecanemab"):
        for copies in (0, 1, 2):
            randomized = [r for r in rows
                          if r["treatment"] == treatment and r["apoe4_copies"] == copies]
            completed = [r for r in randomized if r["completed_week78"] == 1]
            vals = [r["week78_change"] for r in completed]
            counts[f"{treatment}_apoe{copies}"] = {
                "n_randomized": len(randomized),
                "n_week78_completers": len(completed),
                "week78_mean": float(np.mean(vals)),
            }
    randomized_by_treatment = {
        treatment: sum(r["treatment"] == treatment for r in rows)
        for treatment in ("placebo", "lecanemab")
    }
    completed_by_treatment = {
        treatment: sum(r["treatment"] == treatment and r["completed_week78"] == 1
                       for r in rows)
        for treatment in ("placebo", "lecanemab")
    }
    write_json(SUMMARY / "apoe4_interaction.json", {
        "formula": "week78_change ~ lecanemab * apoe4_copies + centered_baseline_cognitive + centered_age",
        "covariance": "HC3", "seed": seed, "n_per_arm": n_per_arm,
        "analysis_population": "Patients with an observed week-78 outcome (trial completers)",
        "participant_flow": {
            "n_randomized_total": len(rows),
            "n_randomized_by_treatment": randomized_by_treatment,
            "n_week78_completers_total": len(analysis_rows),
            "n_week78_completers_by_treatment": completed_by_treatment,
            "n_dropped_before_week78_total": len(rows) - len(analysis_rows),
            "n_dropped_before_week78_by_treatment": {
                treatment: randomized_by_treatment[treatment] - completed_by_treatment[treatment]
                for treatment in ("placebo", "lecanemab")
            },
        },
        "dropout_handling": (
            "Patients who dropped before week 78 are retained in the raw randomized cohort; "
            "their observed_change_at_exit is recorded, week78_change is missing, and they are "
            "excluded from the completer-only week-78 interaction analysis."
        ),
        "interpretation": "Interaction is the change in lecanemab effect per additional APOE4 allele.",
        "coefficients": model, "cell_counts_and_means": counts,
    })


def drug_effect(arm: TrialArm, enabled: bool) -> float:
    if not enabled or arm.is_placebo:
        return 0.0
    survival = 1.0
    for pk, pd in arm.drugs:
        survival *= 1.0 - pd.effect(pk.steady_state_concentration(arm.dosing_interval_h))
    return 1.0 - survival


def simulate_ablation_patient(sim: TrialSimulator, patient: Patient, arm: TrialArm,
                              inflammation: bool, pkpd: bool) -> tuple[float, float, bool]:
    """Mirror TrialSimulator while selectively disabling real cascade/PKPD modules."""
    cascade = ADCascade()
    cascade.state.viability = patient.baseline_viability
    total_steps, dt = sim.duration_weeks * 9, 0.01
    completed = total_steps
    dropped = False
    for step in range(total_steps):
        if sim.dropout_rate > 0 and sim.rng.random() < sim.dropout_rate / total_steps:
            completed, dropped = step, True
            break
        cascade.step(max(0.0, patient.baseline_abeta + sim.rng.normal(0, 0.002)), dt)
        if not inflammation:
            # ADCascade NF-kB/microglial inflammatory state module.
            cascade.state.nfkb = 0.0
            cascade.state.microglia_state = "M0"
    pathology = (0.5 * (1.0 - cascade.state.viability)
                 + 0.3 * min(cascade.state.p_tau, 1.0)
                 + 0.2 * min(cascade.state.calcium / 5.0, 1.0))
    pathology *= 1.0 - 0.34 * drug_effect(arm, pkpd)
    decline = -3.5 * (78 / 52) * (completed / total_steps) * (0.05 + 0.95 * pathology)
    return decline + sim.rng.normal(0, 0.9), cascade.state.viability, dropped


def mean_diff_ci(placebo: np.ndarray, treated: np.ndarray) -> dict:
    diff = float(treated.mean() - placebo.mean())
    se = math.sqrt(placebo.var(ddof=1) / len(placebo) + treated.var(ddof=1) / len(treated))
    df = (placebo.var(ddof=1) / len(placebo) + treated.var(ddof=1) / len(treated)) ** 2 / (
        (placebo.var(ddof=1) / len(placebo)) ** 2 / (len(placebo) - 1)
        + (treated.var(ddof=1) / len(treated)) ** 2 / (len(treated) - 1))
    critical = stats.t.ppf(0.975, df)
    return {"mean_difference": diff, "ci95_low": diff - critical * se,
            "ci95_high": diff + critical * se,
            "p_value": float(2 * stats.t.sf(abs(diff / se), df)), "welch_df": float(df)}


def run_ablation(n_per_arm: int, replicates: int, seed: int) -> None:
    conditions = {
        "full": (True, True, True),
        "no_patient_heterogeneity": (False, True, True),
        "no_inflammation": (True, False, True),
        "no_pkpd": (True, True, False),
    }
    rows = []
    for condition, (heterogeneity, inflammation, pkpd) in conditions.items():
        for rep in range(replicates):
            sim = TrialSimulator(78, seed + rep, 0.15)
            patients = sim.generate_population(2 * n_per_arm, stage="mild")
            if not heterogeneity:
                mean_abeta = float(np.mean([p.baseline_abeta for p in patients]))
                for p in patients:
                    p.baseline_abeta = mean_abeta
            sim.rng.shuffle(patients)
            for treatment, subset in (("placebo", patients[:n_per_arm]),
                                      ("donepezil_memantine", patients[n_per_arm:])):
                arm = arms()[treatment]
                for p in subset:
                    change, viability, dropout = simulate_ablation_patient(
                        sim, p, arm, inflammation, pkpd)
                    rows.append({"condition": condition, "replicate": rep,
                                 "patient_id": p.patient_id, "treatment": treatment,
                                 "apoe4_copies": p.apoe4_copies, "baseline_abeta": p.baseline_abeta,
                                 "week78_change": change, "final_viability": viability,
                                 "dropped_out": int(dropout)})
    write_csv(RAW / "ablation_patient_level.csv", rows)
    summary = []
    for condition in conditions:
        placebo = np.array([r["week78_change"] for r in rows if r["condition"] == condition and r["treatment"] == "placebo"])
        treated = np.array([r["week78_change"] for r in rows if r["condition"] == condition and r["treatment"] == "donepezil_memantine"])
        summary.append({"condition": condition, "comparison": "donepezil_memantine_vs_placebo",
                        "n_per_group": len(placebo), "placebo_mean": float(placebo.mean()),
                        "treated_mean": float(treated.mean()), **mean_diff_ci(placebo, treated)})
    write_csv(SUMMARY / "ablation_summary.csv", summary)
    write_json(SUMMARY / "ablation_module_map.json", {
        "combination": {"drugs": ["donepezil", "memantine"],
                        "rule": "Bliss independence using each drug's repository PKModel/PDModel"},
        "conditions": {
            "full": "ADCascade + patient APOE4/A-beta heterogeneity + repository PK/PD",
            "no_patient_heterogeneity": "Trial-level only: baseline_abeta flattened; this is not a tissue-spatial ablation",
            "no_inflammation": "ADCascade state.nfkb reset to 0 and microglia_state reset to M0 after each step",
            "no_pkpd": "Combined repository PK/PD effect set to zero; cascade and heterogeneity retained",
        },
        "spatial_separation": "The trial simulator does not call layer1_tissue; tissue spatial ablation is reported by grid_convergence.py outputs.",
    })


def simulate_grid(n: int, physical_length: float = 1.0, final_time: float = 0.01,
                  diffusion: float = 0.01, source_radius: float = 0.08,
                  roi_radius: float = 0.15, source_density: float = 10.0,
                  decay: float = 0.1, enabled: bool = True) -> tuple[np.ndarray, dict]:
    dx = physical_length / n
    dt_max = 0.2 * dx * dx / diffusion if enabled else final_time
    steps = max(1, math.ceil(final_time / dt_max))
    dt = final_time / steps
    axis = (np.arange(n) + 0.5) * dx - physical_length / 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    source = radius <= source_radius
    roi = radius <= roi_radius
    c = np.zeros((n, n), dtype=float)
    started = time.perf_counter()
    alpha = diffusion * dt / (dx * dx) if enabled else 0.0
    for _ in range(steps):
        if enabled:
            padded = np.pad(c, 1, mode="edge")  # zero-flux boundary
            lap = (padded[1:-1, 2:] + padded[1:-1, :-2] + padded[2:, 1:-1]
                   + padded[:-2, 1:-1] - 4 * c)
            c += alpha * lap
        c += dt * (source_density * source - decay * c)
    metrics = {"grid_n": n, "physical_length": physical_length, "dx": dx, "dt": dt,
               "steps": steps, "diffusion_coefficient": diffusion if enabled else 0.0,
               "diffusion_cfl": alpha, "source_radius": source_radius,
               "represented_source_area": float(source.sum() * dx * dx),
               "roi_radius": roi_radius, "represented_roi_area": float(roi.sum() * dx * dx),
               "roi_mean": float(c[roi].mean()), "total_mass": float(c.sum() * dx * dx),
               "peak": float(c.max()), "runtime_seconds": time.perf_counter() - started}
    return c, metrics


def run_grid() -> None:
    fields, rows = {}, []
    for n in (100, 200, 400):
        fields[n], metric = simulate_grid(n)
        rows.append(metric)
    ref = fields[400]
    for metric in rows:
        n = metric["grid_n"]
        factor = 400 // n
        ref_restricted = ref.reshape(n, factor, n, factor).mean(axis=(1, 3))
        denom = np.linalg.norm(ref_restricted)
        metric["relative_l2_vs_400"] = float(np.linalg.norm(fields[n] - ref_restricted) / denom) if n != 400 else 0.0
        metric["roi_relative_error_vs_400"] = abs(metric["roi_mean"] - rows[-1]["roi_mean"]) / abs(rows[-1]["roi_mean"])
        metric["mass_relative_error_vs_400"] = abs(metric["total_mass"] - rows[-1]["total_mass"]) / abs(rows[-1]["total_mass"])
    _, no_spatial = simulate_grid(400, enabled=False)
    write_csv(RAW / "grid_metrics.csv", rows)
    write_json(SUMMARY / "grid_convergence.json", {
        "scheme": "cell-centered explicit finite difference, zero-flux boundary",
        "scaling": "dx=L/N; diffusion factor=D*dt/dx^2; physical source and ROI are radius masks; integrals multiply by dx^2",
        "reference": "N=400 restricted by conservative block averaging",
        "metrics": rows,
        "tissue_spatial_ablation": {"full_spatial_n400": rows[-1], "no_diffusion_n400": no_spatial,
                                    "meaning": "Independent Layer-1 tissue field test; not claimed to enter the trial simulator."},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", choices=["apoe4", "ablation", "grid", "all"])
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--n-per-arm", type=int, default=300)
    parser.add_argument("--ablation-n-per-arm", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()
    if args.analysis in ("apoe4", "all"):
        run_apoe4(args.n_per_arm, args.seed)
    if args.analysis in ("ablation", "all"):
        run_ablation(args.ablation_n_per_arm, args.replicates, args.seed)
    if args.analysis in ("grid", "all"):
        run_grid()


if __name__ == "__main__":
    main()
