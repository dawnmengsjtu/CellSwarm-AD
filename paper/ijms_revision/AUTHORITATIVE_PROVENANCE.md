# Authoritative provenance for the IJMS V4 revision

Public repository: https://github.com/dawnmengsjtu/CellSwarm-AD

## Claim-to-evidence chain

| Manuscript claim | Script | Primary evidence | Statistical or display output |
|---|---|---|---|
| 78-week four-arm virtual trial | `clinical_trial/run_engine_trial.py` | `clinical_trial/engine_results/primary_patient_visits.csv` | `primary_gee_week78_terms.csv` |
| Monte Carlo variability | same | `monte_carlo_replicates.csv` (one row per trial and active arm) | `monte_carlo_summary.csv` |
| Donepezil plus memantine | repository PK/PD models and Bliss rule | `clinical_trial/engine_results/provenance.json` | primary and replicate outputs |
| APOE4 interaction | `mechanism_validation/run_validation.py` | `raw/apoe4_patient_level.csv` | `summary/apoe4_interaction.json`; 600 randomized and 510 true week-78 completers |
| Patient-level ablation | same | `raw/ablation_patient_level.csv` | `summary/ablation_summary.csv` |
| Grid convergence | same | fixed-domain solver output | `summary/grid_convergence.json` |
| Parameter provenance | `supplement/extract_runtime_parameters.py` | executable configuration and defaults | selected and complete CSV inventories |
| Layer 3 prompts | repository prompt constructors | code-faithful supplement | deterministic mock output; no live-LLM quantitative result |
| Figures 1–3 | `figures_final/generate_figures_1_to_3.py` | repository processed-data tables and arrays | 600-dpi PNG and vector PDF |
| Figure 4 | `figures/figure4_pkpd/generate_figure4_pkpd.py` | repository PK/PD implementation | panel h visualizes the imposed Bliss surface |
| Figure 5 | `figures_final/generate_figure5_nature9.py` | primary patient visits, 20 independent trials, and separate APOE4 patient-level data | nine panels a–i; `figure5_panel_statistics.json` records plotted statistics |
| Supplementary Tables S4/S6/S7 | `supplement/build_supplement_latex.py` | runtime inventory and executable agent/prompt code | seven-page XeLaTeX PDF |

## Evidence boundaries

- Clarity AD supplies calibration and is not claimed as independent validation.
- Published ADNI natural-history summaries are a contextual comparison only; the discrepancy is disclosed.
- `TrialSimulator` calls the pathology cascade and PK/PD modules but not the Layer 1 tissue grid. Patient-trial and tissue-grid analyses are therefore reported separately.
- The 12 gene weights are fixed, dimensionless, heuristic/model-assigned metadata. They are not transcriptomic effect estimates and are not dynamically updated.
- Layer 3 uses a deterministic mock backend in the reported demonstration. No live LLM generated, selected, modified, or interpreted quantitative results.
- Bliss independence is an imposed structural combination rule, not an estimated interaction or evidence of synergy.
- Figure 5 panel g uses only patients with an observed week-78 endpoint. Dropouts retain `observed_change_at_exit`, have missing `week78_change`, and are excluded from the completer endpoint model.
- Figure 5 display jitter changes x positions only and never changes outcomes or statistics. Baseline-amyloid and viability panels are reported without predictive or efficacy claims.
- The old six-panel Figure 5 and `generate_figure5_final.py` are superseded audit history and are excluded from the V4 package and reproducibility archive.
- The garbled legacy supplementary DOCX is superseded by the XeLaTeX PDF; the `.tex` source is retained.
- Parameter provenance and sensitivity analyses do not establish formal parameter identifiability.

## Reproduction

Run from the repository root:

```powershell
python paper/ijms_revision/clinical_trial/run_engine_trial.py --replicates 20
python paper/ijms_revision/mechanism_validation/run_validation.py all --n-per-arm 300 --ablation-n-per-arm 100 --replicates 3
python paper/ijms_revision/figures_final/generate_figures_1_to_3.py
python paper/ijms_revision/figures/figure4_pkpd/generate_figure4_pkpd.py
python paper/ijms_revision/figures_final/generate_figure5_nature9.py
python paper/ijms_revision/supplement/extract_runtime_parameters.py
python paper/ijms_revision/supplement/build_supplement_latex.py
xelatex -interaction=nonstopmode -halt-on-error paper/ijms_revision/supplement/latex/IJMS_Supplementary_Materials.tex
python -m pytest -q
```
