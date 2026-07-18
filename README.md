# CellSwarm-AD

CellSwarm-AD is a multiscale agent-based research framework for Alzheimer's disease simulation and virtual-trial methodology. This repository is the reviewer-facing reproducibility release accompanying the IJMS revision.

## Author

Xuanlin Meng

## Model scope

- **Layer 0 — cell agents:** neurons, microglia, astrocytes, oligodendrocytes, and endothelial cells.
- **Layer 1 — tissue module:** spatial placement, amyloid-beta diffusion, chemotaxis, and local tissue readouts.
- **Layer 2 — pathology and pharmacology:** the amyloid-beta–calcium–tau–NF-kappaB–viability cascade, PK/PD models, and the virtual-trial engine.
- **Layer 3 — experiment orchestration:** prompt construction and deterministic mock responses for a software proof of concept.

The quantitative virtual-trial path calls the Layer 2 cascade and PK/PD modules; it does not call the Layer 1 tissue grid. Spatial and clinical analyses are therefore evaluated separately. Layer 3 did not generate, select, modify, or interpret any quantitative result reported in the manuscript, and no live OpenAI or Anthropic backend was used.

## Reproducibility contents

This release includes:

- the complete Layer 0–3 source trees and `main.py`;
- deterministic configuration and seeds;
- patient-level primary-trial and Monte Carlo outputs;
- APOE4, ablation, and grid-convergence analyses;
- processed inputs and generators for the final manuscript figures;
- code-faithful supplementary parameter, prompt, and agent-equation materials;
- the repository and revision-specific test suites.

The primary clinical analysis is one simulated trial with 200 patients per arm. Twenty additional trials quantify Monte Carlo variability and are not pooled. Donepezil plus memantine is a distinct simulated arm under an imposed Bliss-independence rule; the repository does not claim pharmacological synergy. Clarity AD supplies calibration rather than independent validation, and published ADNI summaries are used only as contextual comparison.

## Installation

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on macOS/Linux.

## Verification

```bash
python -m pytest -q
python main.py
```

Release audit on 2026-07-18: **168 tests passed**. Two NumPy `trapz` deprecation warnings remain and do not affect test outcomes.

## Reproduce the revision analyses

Run from the repository root:

```bash
python paper/ijms_revision/clinical_trial/run_engine_trial.py --replicates 20
python paper/ijms_revision/mechanism_validation/run_validation.py all
python paper/ijms_revision/figures_final/generate_figures_1_to_3.py
python paper/ijms_revision/figures/figure4_pkpd/generate_figure4_pkpd.py
python paper/ijms_revision/figures_final/generate_figure5_nature9.py
```

Precomputed patient-level and summary outputs are included, so figures can be regenerated without rerunning the slower simulations.

## Evidence boundaries

- Gene weights are fixed heuristic/model-assigned metadata, not measured transcriptomic effects.
- The current parameter provenance and sensitivity analyses do not establish formal parameter identifiability.
- APOE4 endpoint analysis uses true week-78 completers; dropout-at-exit values are not relabelled as week-78 outcomes.
- Figure 4's Bliss surface visualizes an imposed structural rule and is not an estimated drug interaction.

Questions about this reproducibility release can be submitted through [GitHub Issues](https://github.com/dawnmengsjtu/CellSwarm-AD/issues).
