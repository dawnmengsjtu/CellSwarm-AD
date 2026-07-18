# IJMS mechanism validation

This directory is an isolated, reproducible replacement for the earlier mechanism analyses. It does not modify manuscript DOCX files, reviewer-response files, main-trial outputs, repository configs, or model source.

## Analyses

1. **APOE4 interaction** uses patient-level placebo and lecanemab simulations. The prespecified model is `week78_change ~ lecanemab * APOE4_copies + centered baseline cognition + centered age`, estimated by OLS with HC3 robust covariance. The interaction coefficient, 95% CI, and two-sided p-value are in `summary/apoe4_interaction.json`; every modeled patient is in `raw/apoe4_patient_level.csv`. APOE4 is modeled additively per allele because the virtual population may contain few two-copy patients; no genotype categories are merged after inspecting results.
2. **Ablation** uses one combination throughout: repository PK/PD models for donepezil + memantine combined by Bliss independence. Trial ablations map to real code states: patient Aβ heterogeneity, `ADCascade` NF-κB/microglial inflammatory state, and PK/PD effect. The trial simulator does not call Layer 1, so spatial removal is not mislabelled: patient heterogeneity is a trial-level ablation, while the independent diffusion/no-diffusion result is a tissue-spatial ablation.
3. **Grid convergence** fixes the square physical domain at L=1 and changes `dx=L/N` for N=100/200/400. The explicit diffusion coefficient is scaled as `D*dt/dx²`; `dt` is refined to keep CFL ≤0.2. Source and ROI are fixed-radius physical masks, source is a rate density, and area integrals use `dx²`. Field errors use conservative block restriction of N=400; ROI and mass errors are also reported.

The constants are declared in code and are not selected based on the resulting significance, effect direction, or convergence values. The simulation seed is fixed and recorded.

## Reproduce

From the repository root on Windows PowerShell:

```powershell
python paper/ijms_revision/mechanism_validation/run_validation.py all
python -m pytest paper/ijms_revision/mechanism_validation/test_validation.py -q
```

Run components separately with `apoe4`, `ablation`, or `grid`. Useful explicit options:

```powershell
python paper/ijms_revision/mechanism_validation/run_validation.py apoe4 --seed 20260716 --n-per-arm 300
python paper/ijms_revision/mechanism_validation/run_validation.py ablation --seed 20260716 --ablation-n-per-arm 100 --replicates 3
```

## Output inventory

- `raw/apoe4_patient_level.csv`: patient-level genotype, allocation, covariates, outcome, viability, dropout.
- `summary/apoe4_interaction.json`: formula, HC3 coefficient table, interaction CI/p, genotype-arm counts.
- `raw/ablation_patient_level.csv`: patient-level results for every ablation condition.
- `summary/ablation_summary.csv`: donepezil+memantine versus placebo mean differences, Welch CI/p.
- `summary/ablation_module_map.json`: auditable mapping from ablation labels to model modules and the spatial/trial separation.
- `raw/grid_metrics.csv`: dx, dt, CFL, represented areas, physical metrics, runtime, and errors.
- `summary/grid_convergence.json`: convergence metadata plus independent tissue spatial/no-diffusion comparison.

Limitations: these are deterministic virtual-population experiments conditional on the existing model calibration. The APOE4 interaction is not clinical evidence. Pooled ablation patients across simulation replicates improve Monte Carlo precision but are not independent biological cohorts; replicate identity remains in the raw file.
