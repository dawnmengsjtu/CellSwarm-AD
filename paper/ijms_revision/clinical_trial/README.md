# IJMS authoritative clinical-trial chain

The only manuscript-authoritative clinical pipeline is `run_engine_trial.py` and
the only authoritative output directory is `engine_results/`. It calls
`ADCascade` and the repository PK/PD implementations, uses one primary
n=200-per-arm trial, and keeps 20 Monte Carlo trials separate.

`run_clinical_trial.py`, `results/`, and the CSV/JSON files in this directory
root are retained only as deprecated audit artifacts. They use an illustrative
hard-coded trajectory generator and must not be cited, plotted, or described as
CellSwarm-AD engine output.

Reproduce from repository root:

```powershell
python paper/ijms_revision/clinical_trial/run_engine_trial.py --replicates 20
python paper/ijms_revision/figures_final/generate_figure5_nature9.py
python -m pytest paper/ijms_revision/clinical_trial/test_engine_trial.py -q
```

`make_figure5.py` and `figure5_revised.png` are retained as historical
six-panel audit artifacts only. They are not the journal-facing Figure 5 and
must not be used for the revision. The authoritative generator reads both the
clinical outputs below and the separately simulated APOE4 analysis files under
`paper/ijms_revision/mechanism_validation/`.

Authoritative chain:

1. `run_engine_trial.py`
2. `engine_results/primary_patient_visits.csv`
3. `engine_results/primary_gee_week78_terms.csv`
4. `engine_results/monte_carlo_replicates.csv`
5. `engine_results/monte_carlo_summary.csv`
6. `engine_results/provenance.json`
7. `../figures_final/generate_figure5_nature9.py`
8. `../figures_final/figure5_panel_statistics.json`
9. `../figures_final/figure5_final.png` and `figure5_final.pdf`

Clarity AD supplies the fixed calibration constant and is not claimed as an
independent validation dataset. Bliss independence is a structural combination
assumption, not an empirically estimated synergy.
