"""Build the evidence-bound nine-panel IJMS clinical figure.

Every plotted observation is read from the authoritative revision outputs.  The
only random operation is deterministic horizontal jitter used to separate
overlapping observed points; it does not alter any outcome value or statistic.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "paper/ijms_revision/clinical_trial/engine_results"
APOE = ROOT / "paper/ijms_revision/mechanism_validation"
OUT = ROOT / "paper/ijms_revision/figures_final"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["placebo", "lecanemab", "donepezil", "donepezil+memantine"]
LABELS = {
    "placebo": "Placebo",
    "lecanemab": "Lecanemab",
    "donepezil": "Donepezil",
    "donepezil+memantine": "Done + Mem.",
}
COLORS = {
    "placebo": "#5F6368",
    "lecanemab": "#0072B2",
    "donepezil": "#E69F00",
    "donepezil+memantine": "#CC79A7",
}
MARKERS = {"placebo": "o", "lecanemab": "s", "donepezil": "^", "donepezil+memantine": "D"}
LINESTYLES = {"placebo": "-", "lecanemab": "--", "donepezil": "-.", "donepezil+memantine": ":"}
RNG = np.random.default_rng(20260717)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.0,
    "axes.titlesize": 6.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 6.2,
    "xtick.labelsize": 5.4,
    "ytick.labelsize": 5.4,
    "legend.fontsize": 5.2,
    "axes.linewidth": 0.55,
    "lines.linewidth": 1.05,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.3,
    "ytick.major.size": 2.3,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def clean(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)
    if grid:
        ax.grid(axis="y", color="#D8DADD", lw=0.45, alpha=0.7, zorder=0)


def letter(ax: plt.Axes, value: str, x: float = -0.17, y: float = 1.12) -> None:
    ax.text(x, y, value, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="top", color="#111111", clip_on=False)


def ci95(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, mean - half, mean + half


def jitter(position: float, n: int, width: float) -> np.ndarray:
    return position + RNG.uniform(-width, width, n)


visits = pd.read_csv(TRIAL / "primary_patient_visits.csv")
gee = pd.read_csv(TRIAL / "primary_gee_week78_terms.csv")
mc = pd.read_csv(TRIAL / "monte_carlo_replicates.csv")
apoe_raw = pd.read_csv(APOE / "raw/apoe4_patient_level.csv")
apoe_summary = json.loads((APOE / "summary/apoe4_interaction.json").read_text(encoding="utf-8"))
interaction = next(row for row in apoe_summary["coefficients"]
                   if row["term"] == "lecanemab_x_apoe4_copies")
apoe_complete = apoe_raw.loc[
    apoe_raw.completed_week78.eq(1) & apoe_raw.week78_change.notna()
].copy()
apoe_flow = apoe_summary["participant_flow"]

# Fail fast if a dropout has been mislabeled as having a week-78 outcome, or if
# the plotted completer population disagrees with the prespecified summary.
if apoe_raw.loc[apoe_raw.dropped_out.eq(1), "week78_change"].notna().any():
    raise ValueError("APOE4 dropout rows must not contain week78_change")
if apoe_raw.loc[apoe_raw.completed_week78.eq(1), "week78_change"].isna().any():
    raise ValueError("APOE4 completer rows must contain week78_change")
if len(apoe_raw) != apoe_flow["n_randomized_total"]:
    raise ValueError("APOE4 randomized count differs between raw data and summary")
if len(apoe_complete) != apoe_flow["n_week78_completers_total"]:
    raise ValueError("APOE4 completer count differs between raw data and summary")

baseline = visits.loc[visits.visit_week.eq(0)].drop_duplicates("patient_id").copy()
endpoint = visits.loc[visits.visit_week.eq(78) & visits.observed].copy()
trajectory = (visits.loc[visits.observed]
              .groupby(["treatment", "visit_week"])["cognitive_change"]
              .agg(["mean", "sem", "count"]).reset_index())

# Primary-trial Cohen's d values are recomputed directly from observed week-78 rows.
placebo_endpoint = endpoint.loc[endpoint.treatment.eq("placebo"), "cognitive_change"].to_numpy()
primary_d: dict[str, float] = {}
for arm in ARMS[1:]:
    active = endpoint.loc[endpoint.treatment.eq(arm), "cognitive_change"].to_numpy()
    pooled_sd = np.sqrt(((len(placebo_endpoint) - 1) * placebo_endpoint.var(ddof=1)
                         + (len(active) - 1) * active.var(ddof=1))
                        / (len(placebo_endpoint) + len(active) - 2))
    primary_d[arm] = float((active.mean() - placebo_endpoint.mean()) / pooled_sd)

# Panel h uses the prespecified lecanemab endpoint subset.  HC3 covariance is
# used for the slope interval and P value; baseline Aβ has three modeled strata.
lecanemab_endpoint = endpoint.loc[endpoint.treatment.eq("lecanemab")].copy()
h_model = sm.OLS(lecanemab_endpoint.cognitive_change,
                 sm.add_constant(lecanemab_endpoint.baseline_abeta)).fit(cov_type="HC3")
h_grid = np.linspace(lecanemab_endpoint.baseline_abeta.min(),
                     lecanemab_endpoint.baseline_abeta.max(), 120)
h_prediction = h_model.get_prediction(sm.add_constant(h_grid)).summary_frame(alpha=0.05)

fig = plt.figure(figsize=(183 / 25.4, 168 / 25.4), layout="constrained")
layout = fig.add_gridspec(4, 6, height_ratios=[0.78, 1.35, 1.12, 1.12])
fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.06, hspace=0.10)

# a | Baseline cohort profile: age and APOE4-carrier percentage.
ax = fig.add_subplot(layout[0, 0:2])
for yi, arm in enumerate(ARMS):
    group = baseline.loc[baseline.treatment.eq(arm)]
    mean, lo, hi = ci95(group.age)
    ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], color=COLORS[arm],
                marker=MARKERS[arm], mfc="white", mec=COLORS[arm], mew=0.8,
                ms=3.5, lw=0.9, capsize=2.0, zorder=3)
    carrier = 100 * group.apoe4_copies.gt(0).mean()
    ax.text(73.1, yi, f"{carrier:.0f}%", ha="center", va="center", fontsize=5.1)
ax.text(73.1, -0.72, "APOE4+", ha="center", va="bottom", fontsize=5.1, fontweight="bold")
ax.set_yticks(range(4), [LABELS[a] for a in ARMS])
ax.set_xlim(67.0, 74.1)
ax.set_xticks([68, 70, 72])
ax.set_xlabel("Mean age (95% CI), years")
ax.set_title("Baseline cohort profile", loc="left", pad=2)
ax.invert_yaxis()
clean(ax, grid=False); letter(ax, "a", x=-0.23)

# b | Baseline MMSE-like score distribution.
ax = fig.add_subplot(layout[0, 2:4])
for xi, arm in enumerate(ARMS, start=1):
    vals = baseline.loc[baseline.treatment.eq(arm), "baseline_score"].to_numpy()
    ax.scatter(jitter(xi, len(vals), 0.12), vals, s=2.1, color=COLORS[arm],
               alpha=0.20, edgecolors="none", zorder=1)
bp = ax.boxplot([baseline.loc[baseline.treatment.eq(a), "baseline_score"] for a in ARMS],
                positions=range(1, 5), widths=0.48, patch_artist=True,
                showfliers=False, medianprops={"color": "#111111", "lw": 0.8},
                whiskerprops={"color": "#555555", "lw": 0.65},
                capprops={"color": "#555555", "lw": 0.65})
for box, arm in zip(bp["boxes"], ARMS):
    box.set(facecolor="white", edgecolor=COLORS[arm], linewidth=0.8)
ax.set_xticks(range(1, 5), ["Pbo", "Lec", "Don", "D+M"])
ax.set_ylabel("Baseline score")
ax.set_ylim(17.5, 30.5)
ax.set_title("Baseline cognitive-score balance", loc="left", pad=2)
clean(ax, grid=True); letter(ax, "b")

# c | Regimens and observation schedule from the authoritative engine config.
ax = fig.add_subplot(layout[0, 4:6])
ax.set_xlim(-18, 81); ax.set_ylim(-0.9, 4.0)
regimen = {
    "placebo": "control",
    "lecanemab": "q2w",
    "donepezil": "q24h",
    "donepezil+memantine": "q24h joint PD",
}
for yi, arm in enumerate(ARMS[::-1]):
    ax.hlines(yi, 0, 78, color=COLORS[arm], lw=1.15, linestyle=LINESTYLES[arm])
    ax.plot([0, 78], [yi, yi], linestyle="none", marker=MARKERS[arm], ms=2.8,
            mfc="white", mec=COLORS[arm], mew=0.7)
    ax.text(-17, yi, LABELS[arm], ha="left", va="center", fontsize=5.1)
    ax.text(76.5, yi + 0.17, regimen[arm], ha="right", va="bottom", fontsize=4.8,
            color="#333333")
for week in [0, 13, 26, 39, 52, 65, 78]:
    ax.vlines(week, -0.45, 3.3, color="#B8BBC0", lw=0.35, zorder=0)
ax.set_xticks([0, 26, 52, 78])
ax.set_xlabel("Visit week")
ax.set_yticks([])
ax.text(0.99, 0.98, "n=200/arm; 15% MCAR", transform=ax.transAxes,
        ha="right", va="top", fontsize=4.9)
ax.set_title("Regimen and visit schedule", loc="left", pad=2)
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="x", direction="out", pad=1.5)
letter(ax, "c")

# d | Longitudinal observed means and 95% confidence intervals.
ax = fig.add_subplot(layout[1, 0:4])
for arm in ARMS:
    group = trajectory.loc[trajectory.treatment.eq(arm)]
    x = group.visit_week.to_numpy(); y = group["mean"].to_numpy()
    ci = 1.96 * group["sem"].to_numpy()
    ax.fill_between(x, y - ci, y + ci, color=COLORS[arm], alpha=0.10, lw=0)
    ax.plot(x, y, color=COLORS[arm], linestyle=LINESTYLES[arm], marker=MARKERS[arm],
            ms=3.0, mfc="white", mec=COLORS[arm], mew=0.75, label=LABELS[arm])
ax.axhline(0, color="#333333", lw=0.55)
ax.set_xticks([0, 13, 26, 39, 52, 65, 78])
ax.set_xlabel("Week")
ax.set_ylabel("MMSE-like change from baseline")
ax.set_ylim(-2.15, 0.55)
ax.legend(loc="lower left", ncol=2, columnspacing=0.8, handlelength=2.4,
          borderaxespad=0.2, labelspacing=0.25)
ax.set_title("Longitudinal cognitive outcome (mean and 95% CI)", loc="left", pad=2)
clean(ax, grid=True); letter(ax, "d", x=-0.10, y=1.08)

# e | Patient-level observed week-78 endpoint distribution.
ax = fig.add_subplot(layout[1, 4:6])
endpoint_values = [endpoint.loc[endpoint.treatment.eq(a), "cognitive_change"].to_numpy() for a in ARMS]
for xi, (arm, vals) in enumerate(zip(ARMS, endpoint_values), start=1):
    ax.scatter(jitter(xi, len(vals), 0.16), vals, s=2.4, color=COLORS[arm],
               alpha=0.23, edgecolors="none", zorder=1)
bp = ax.boxplot(endpoint_values, positions=range(1, 5), widths=0.48, patch_artist=True,
                showfliers=False, medianprops={"color": "#111111", "lw": 0.9},
                whiskerprops={"color": "#555555", "lw": 0.65},
                capprops={"color": "#555555", "lw": 0.65})
for box, arm in zip(bp["boxes"], ARMS):
    box.set(facecolor="white", edgecolor=COLORS[arm], linewidth=0.85)
means = [np.mean(v) for v in endpoint_values]
ax.scatter(range(1, 5), means, marker="D", s=10, facecolor="#111111", edgecolor="white",
           linewidth=0.35, zorder=4, label="mean")
ax.set_xticks(range(1, 5), ["Pbo", "Lec", "Don", "D+M"])
ax.set_ylabel("Week-78 change")
ax.set_ylim(-4.4, 1.35)
ax.text(0.98, 0.04, "n=170/arm", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0)
ax.set_title("Observed week-78 distribution", loc="left", pad=2)
clean(ax, grid=True); letter(ax, "e", x=-0.22, y=1.08)

# f | Independent-trial effect sizes with the primary trial identified.
ax = fig.add_subplot(layout[2, 0:3])
mc_values = [mc.loc[mc.treatment.eq(a), "cohens_d"].to_numpy() for a in ARMS[1:]]
violins = ax.violinplot(mc_values, positions=range(1, 4), widths=0.70,
                       showextrema=False, showmeans=False, showmedians=False)
for body, arm in zip(violins["bodies"], ARMS[1:]):
    body.set_facecolor(COLORS[arm]); body.set_edgecolor(COLORS[arm]); body.set_linewidth(0.55)
    body.set_alpha(0.16)
for xi, (arm, vals) in enumerate(zip(ARMS[1:], mc_values), start=1):
    ax.scatter(jitter(xi, len(vals), 0.13), vals, s=8, marker=MARKERS[arm],
               facecolor="white", edgecolor=COLORS[arm], linewidth=0.55, zorder=3)
    mean = vals.mean(); lo, hi = np.quantile(vals, [0.025, 0.975])
    ax.errorbar(xi, mean, yerr=[[mean - lo], [hi - mean]], fmt="_", color="#111111",
                ms=7, mew=1.0, lw=0.85, capsize=2.3, zorder=4)
    ax.scatter(xi, primary_d[arm], marker="*", s=24, facecolor=COLORS[arm],
               edgecolor="#111111", linewidth=0.35, zorder=5)
ax.axhline(0, color="#333333", lw=0.55)
ax.set_xticks(range(1, 4), ["Lecanemab", "Donepezil", "Done + Mem."])
ax.set_ylabel("Cohen's d vs placebo")
ax.set_ylim(-0.02, 0.65)
ax.legend(handles=[
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
           markeredgecolor="#555555", markersize=3.6, label="20 trials"),
    Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#777777",
           markeredgecolor="#111111", markersize=5.3, label="primary trial"),
], loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
ax.set_title("Independent-trial effect-size distribution", loc="left", pad=2)
clean(ax, grid=True); letter(ax, "f", x=-0.13, y=1.09)

# g | Separately simulated APOE4 interaction among true week-78 completers.
ax = fig.add_subplot(layout[2, 3:6])
for treatment, marker, linestyle in [("placebo", "o", "-"), ("lecanemab", "s", "--")]:
    means_g, lows_g, highs_g = [], [], []
    for copies in [0, 1, 2]:
        vals = apoe_complete.loc[(apoe_complete.treatment.eq(treatment))
                                 & (apoe_complete.apoe4_copies.eq(copies)),
                                 "week78_change"].to_numpy()
        mean, lo, hi = ci95(vals)
        means_g.append(mean); lows_g.append(lo); highs_g.append(hi)
    means_g = np.asarray(means_g); lows_g = np.asarray(lows_g); highs_g = np.asarray(highs_g)
    color = COLORS[treatment]
    ax.errorbar([0, 1, 2], means_g, yerr=[means_g - lows_g, highs_g - means_g],
                color=color, linestyle=linestyle, marker=marker, mfc="white", mec=color,
                mew=0.8, ms=3.5, capsize=2.2, label=LABELS[treatment])
ax.text(0.02, 0.04,
        f"Treatment x copy: {interaction['estimate']:.3f}\n"
        f"95% CI {interaction['ci95_low']:.3f} to {interaction['ci95_high']:.3f}; P={interaction['p_value']:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.0)
ax.text(0.98, 0.98,
        f"randomized n={apoe_flow['n_randomized_total']}; "
        f"week-78 n={apoe_flow['n_week78_completers_total']}",
        transform=ax.transAxes, ha="right", va="top", fontsize=4.9, color="#333333")
ax.set_xticks([0, 1, 2])
ax.set_xlabel("APOE4 copies")
ax.set_ylabel("Observed week-78 change (mean, 95% CI)")
ax.set_ylim(-2.45, -0.95)
ax.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.2, borderaxespad=0.2)
ax.set_title("Prespecified APOE4 interaction among week-78 completers", loc="left", pad=2)
clean(ax, grid=True); letter(ax, "g", x=-0.14, y=1.09)

# h | Baseline Aβ strata versus endpoint in the lecanemab arm.
ax = fig.add_subplot(layout[3, 0:3])
x_jitter = lecanemab_endpoint.baseline_abeta.to_numpy() + RNG.uniform(-0.008, 0.008, len(lecanemab_endpoint))
ax.scatter(x_jitter, lecanemab_endpoint.cognitive_change, s=5.5, facecolor="white",
           edgecolor=COLORS["lecanemab"], linewidth=0.45, alpha=0.62)
ax.fill_between(h_grid, h_prediction["mean_ci_lower"].to_numpy(),
                h_prediction["mean_ci_upper"].to_numpy(), color=COLORS["lecanemab"], alpha=0.13, lw=0)
ax.plot(h_grid, h_prediction["mean"].to_numpy(), color=COLORS["lecanemab"], lw=1.05)
h_ci = h_model.conf_int().iloc[1]
ax.text(0.02, 0.04,
        f"n={len(lecanemab_endpoint)}; R²={h_model.rsquared:.3f}\n"
        f"slope={h_model.params.iloc[1]:.2f} ({h_ci.iloc[0]:.2f} to {h_ci.iloc[1]:.2f}); P={h_model.pvalues.iloc[1]:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.0)
ax.set_xticks([0.50, 0.65, 0.80])
ax.set_xlabel("Baseline Aβ (modeled strata)")
ax.set_ylabel("Week-78 change")
ax.set_title("Baseline Aβ and lecanemab-arm outcome", loc="left", pad=2)
clean(ax, grid=True); letter(ax, "h", x=-0.13, y=1.09)

# i | Model-derived week-78 neuronal viability, reported descriptively.
ax = fig.add_subplot(layout[3, 3:6])
viability_values = [endpoint.loc[endpoint.treatment.eq(a), "viability"].to_numpy() for a in ARMS]
for xi, (arm, vals) in enumerate(zip(ARMS, viability_values), start=1):
    mean, lo, hi = ci95(vals)
    ax.errorbar(xi, mean, yerr=[[mean - lo], [hi - mean]], color=COLORS[arm],
                marker=MARKERS[arm], mfc="white", mec=COLORS[arm], mew=0.9,
                ms=4.0, lw=1.0, capsize=2.5, zorder=3)
    ax.text(xi, lo - 0.0011, f"{mean:.3f}", ha="center", va="top", fontsize=4.8)
ax.set_xticks(range(1, 5), ["Pbo", "Lec", "Don", "D+M"])
ax.set_ylabel("Mean neuronal viability (95% CI)")
ax.set_ylim(0.960, 0.992)
ax.text(0.98, 0.04, "expanded axis; n=170/arm", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=5.0)
ax.set_title("Model-derived week-78 viability", loc="left", pad=2)
clean(ax, grid=True); letter(ax, "i", x=-0.13, y=1.09)

# A machine-readable, reviewer-auditable record of the plotted statistics.
statistics = {
    "figure": "Figure 5, nine-panel Nature-style rebuild",
    "sources": {
        "primary_patient_visits": (TRIAL / "primary_patient_visits.csv").relative_to(ROOT).as_posix(),
        "primary_gee_week78_terms": (TRIAL / "primary_gee_week78_terms.csv").relative_to(ROOT).as_posix(),
        "monte_carlo_replicates": (TRIAL / "monte_carlo_replicates.csv").relative_to(ROOT).as_posix(),
        "apoe4_patient_level": (APOE / "raw/apoe4_patient_level.csv").relative_to(ROOT).as_posix(),
        "apoe4_interaction": (APOE / "summary/apoe4_interaction.json").relative_to(ROOT).as_posix(),
    },
    "primary_trial": {
        "n_randomized": int(baseline.patient_id.nunique()),
        "n_per_arm": {a: int((baseline.treatment == a).sum()) for a in ARMS},
        "n_week78_observed": {a: int((endpoint.treatment == a).sum()) for a in ARMS},
        "week78_mean": {a: float(endpoint.loc[endpoint.treatment.eq(a), "cognitive_change"].mean()) for a in ARMS},
        "week78_sem": {a: float(endpoint.loc[endpoint.treatment.eq(a), "cognitive_change"].sem()) for a in ARMS},
        "week78_viability_mean": {a: float(endpoint.loc[endpoint.treatment.eq(a), "viability"].mean()) for a in ARMS},
        "week78_viability_ci95": {
            a: [float(ci95(endpoint.loc[endpoint.treatment.eq(a), "viability"])[1]),
                float(ci95(endpoint.loc[endpoint.treatment.eq(a), "viability"])[2])]
            for a in ARMS
        },
        "single_trial_cohens_d": primary_d,
        "gee": gee.to_dict(orient="records"),
    },
    "monte_carlo": {
        a: {
            "n": int(len(mc.loc[mc.treatment.eq(a)])),
            "mean_d": float(mc.loc[mc.treatment.eq(a), "cohens_d"].mean()),
            "q025_d": float(mc.loc[mc.treatment.eq(a), "cohens_d"].quantile(0.025)),
            "q975_d": float(mc.loc[mc.treatment.eq(a), "cohens_d"].quantile(0.975)),
        } for a in ARMS[1:]
    },
    "apoe4_interaction": interaction,
    "apoe4_analysis_population": {
        "definition": apoe_summary["analysis_population"],
        **apoe_flow,
        "cell_counts_and_means": apoe_summary["cell_counts_and_means"],
        "dropout_handling": apoe_summary["dropout_handling"],
    },
    "baseline_abeta_lecanemab_regression": {
        "n": int(len(lecanemab_endpoint)),
        "slope": float(h_model.params.iloc[1]),
        "ci95_low": float(h_ci.iloc[0]),
        "ci95_high": float(h_ci.iloc[1]),
        "p_value": float(h_model.pvalues.iloc[1]),
        "r_squared": float(h_model.rsquared),
        "covariance": "HC3",
        "note": "Baseline Aβ consists of three modeled strata; horizontal point jitter is display-only.",
    },
    "display": {
        "width_mm": 183,
        "height_mm": 168,
        "raster_dpi": 600,
        "panel_labels": list("abcdefghi"),
        "display_only_jitter_seed": 20260717,
    },
}
(OUT / "figure5_panel_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

fig.savefig(OUT / "figure5_final.png", dpi=600, facecolor="white")
# Omit wall-clock PDF metadata so repeated runs with unchanged data are
# byte-reproducible as well as visually reproducible.
fig.savefig(OUT / "figure5_final.pdf", dpi=600, facecolor="white",
            metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
print(OUT / "figure5_final.png")
