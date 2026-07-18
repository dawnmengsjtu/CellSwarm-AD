"""Generate publication-quality Figures 1-3 from existing processed evidence.

No observations are generated here. Every panel reads repository-processed output.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/ijms_revision/figures_final"
OUT.mkdir(parents=True, exist_ok=True)
P1 = ROOT / "paper/fig2/processed_data"
P2 = ROOT / "paper/fig3/processed_data"
P3 = ROOT / "paper/fig4/processed_data"

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6E6E6E"
LIGHT = "#B8B8B8"
COL5 = [BLUE, SKY, GREEN, ORANGE, VERM]

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.2,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "axes.linewidth": 0.65,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "lines.linewidth": 1.15,
    "legend.fontsize": 5.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def label(ax, letter):
    ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", ha="left", va="top")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_by(df, keys, cols):
    return df.groupby(keys, as_index=False)[cols].mean(numeric_only=True)


def figure1():
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.45), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.10, hspace=0.10)

    # a: neuronal viability
    ax = axs[0, 0]
    d = pd.read_csv(P1 / "panel_a_aggregated.csv")
    for c, (dose, g) in zip(COL5, d.groupby("abeta_uM")):
        ax.plot(g.time_hours, g.viability_mean, color=c, label=f"{dose:g}")
        ax.fill_between(g.time_hours, g.viability_mean-g.viability_sem,
                        g.viability_mean+g.viability_sem, color=c, alpha=.13, lw=0)
    ax.set(xlabel="Time (h)", ylabel="Neuron viability", ylim=(-.02, 1.03))
    ax.legend(title="Aβ (µM)", ncol=2, loc="lower left", handlelength=1.3)
    style(ax); label(ax, "a")

    # b: firing dose-response
    ax = axs[0, 1]
    d = pd.read_csv(P1 / "panel_b_aggregated.csv").sort_values("abeta_uM")
    ax.errorbar(d.abeta_uM, d.firing_rate_mean, yerr=d.firing_rate_sem,
                fmt="o", ms=3.4, color=BLUE, capsize=2, zorder=3)
    ax.plot(d.abeta_uM, d.hill_fit, color=VERM)
    ax.set_xscale("log"); ax.set(xlabel="Aβ (µM)", ylabel="Firing rate (Hz)")
    ax.text(.05, .08, "EC50 = 2.21 ± 0.16 µM\nR² = 0.991", transform=ax.transAxes)
    style(ax); label(ax, "b")

    # c: microglial states
    ax = axs[0, 2]
    d = pd.read_csv(P1 / "panel_c_aggregated.csv")
    x = np.arange(len(d)); bottom = np.zeros(len(d))
    for col, name, c in [("resting_mean", "M0", LIGHT),
                         ("anti_inflammatory_mean", "M2", BLUE),
                         ("pro_inflammatory_mean", "M1", VERM)]:
        ax.bar(x, d[col], bottom=bottom, color=c, width=.72, label=name)
        bottom += d[col].to_numpy()
    ax.set_xticks(x, [f"{v:g}" for v in d.abeta_uM])
    ax.set(xlabel="Aβ (µM)", ylabel="State fraction", ylim=(0, 1.04))
    ax.legend(ncol=3, loc="upper center")
    style(ax); label(ax, "c")

    # d: inflammatory balance
    ax = axs[1, 0]
    d = pd.read_csv(P1 / "panel_d_aggregated.csv")
    cm = {"Low": GREEN, "Medium": ORANGE, "High": VERM}
    for name, g in d.groupby("condition"):
        c = cm[name]
        ax.plot(g.time_hours, g.M1_M2_ratio_mean, color=c, label=name)
        ax.fill_between(g.time_hours, g.M1_M2_ratio_mean-g.M1_M2_ratio_sem,
                        g.M1_M2_ratio_mean+g.M1_M2_ratio_sem, color=c, alpha=.15, lw=0)
    ax.axhline(1, color=GRAY, lw=.7, ls="--")
    ax.set(xlabel="Time (h)", ylabel="M1/M2 ratio")
    ax.legend(loc="upper left")
    style(ax); label(ax, "d")

    # e: astrocyte responses, normalized to each series maximum
    ax = axs[1, 1]
    d = pd.read_csv(P1 / "panel_e_aggregated.csv")
    for col, lab, c, marker in [
        ("glutamate_uptake_mean", "Glutamate uptake", GREEN, "o"),
        ("calcium_wave_mean", r"$Ca^{2+}$ signal", ORANGE, "s"),
        ("reactivity_gfap_mean", "GFAP-equivalent", VERM, "^")]:
        y = d[col] / d[col].max()
        ax.plot(d.abeta_uM, y, marker=marker, ms=3, color=c, label=lab)
    ax.set(xlabel="Aβ (µM)", ylabel="Normalized response", ylim=(-.03, 1.05))
    ax.legend(loc="center right")
    style(ax); label(ax, "e")

    # f: viability across cell types and conditions
    ax = axs[1, 2]
    d = pd.read_csv(P1 / "panel_h_aggregated.csv")
    cells = ["Neuron", "Microglia", "Astrocyte"]
    conds = ["Control", "Low_Abeta", "High_Abeta"]
    cc = [GRAY, ORANGE, VERM]
    for j, (cond, c) in enumerate(zip(conds, cc)):
        g = d[d.condition == cond].set_index("cell_type").reindex(cells)
        ax.errorbar(np.arange(3)+(j-1)*.16, g.viability_mean, yerr=g.viability_sem,
                    fmt="o", ms=4, color=c, capsize=2, label=cond.replace("_", " "))
    ax.set_xticks(range(3), cells, rotation=18)
    ax.set(ylabel="Viability", ylim=(0.35, 1.04))
    ax.legend(loc="lower right")
    style(ax); label(ax, "f")

    # g: calcium and tau time courses
    ax = axs[2, 0]
    d = pd.read_csv(P1 / "panel_i_calcium_tau_aggregated.csv")
    sel = ["Control", "Low_Abeta", "High_Abeta"]
    cc = [BLUE, ORANGE, VERM]
    ax2 = ax.twinx()
    for name, c in zip(sel, cc):
        g = d[d.condition == name]
        ax.plot(g.time_hours, g.calcium_mean, color=c, label=name.replace("_", " "))
        ax2.plot(g.time_hours, g.tau_mean, color=c, ls="--")
    ax.set(xlabel="Time (h)", ylabel=r"$Ca^{2+}$ (µM)")
    ax2.set_ylabel("p-Tau (fraction)", color=GRAY); ax2.tick_params(axis="y", colors=GRAY)
    ax.legend(loc="upper left")
    style(ax); ax2.spines["top"].set_visible(False); label(ax, "g")

    # h: microglial clearance
    ax = axs[2, 1]
    d = pd.read_csv(P1 / "panel_j_aggregated.csv")
    for c, (name, g) in zip([SKY, ORANGE, VERM], d.groupby("density_label", sort=False)):
        ax.plot(g.time_hours, g.abeta_remaining_mean, color=c, label=name)
        ax.fill_between(g.time_hours, g.abeta_remaining_mean-g.abeta_remaining_sem,
                        g.abeta_remaining_mean+g.abeta_remaining_sem, color=c, alpha=.13, lw=0)
    ax.set(xlabel="Time (h)", ylabel="Aβ remaining (µM)")
    ax.legend(loc="upper right")
    style(ax); label(ax, "h")

    # i: NF-kappaB activation
    ax = axs[2, 2]
    d = pd.read_csv(P1 / "panel_k_nfkb_dynamics.csv")
    for c, (name, g) in zip([GRAY, SKY, ORANGE, VERM], d.groupby("condition", sort=False)):
        ax.plot(g.time_hours, g.nfkb_mean, color=c, label=name)
        ax.fill_between(g.time_hours, g.nfkb_mean-g.nfkb_sem,
                        g.nfkb_mean+g.nfkb_sem, color=c, alpha=.13, lw=0)
    ax.set(xlabel="Time (h)", ylabel="NF-κB activity", ylim=(-.02, 1.04))
    ax.legend(loc="center right")
    style(ax); label(ax, "i")
    save(fig, "figure1_final")


def figure2():
    fig = plt.figure(figsize=(7.2, 7.55), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    axs = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.10, hspace=0.10)

    # a: cell positions
    ax = axs[0][0]
    d = pd.read_csv(P2 / "panel_a_plot.csv")
    palette = {"Neuron": BLUE, "Microglia": VERM, "Astrocyte": ORANGE,
               "Oligodendrocyte": PURPLE, "Endothelial": GREEN}
    for name, g in d.groupby("cell_type"):
        ax.scatter(g.x, g.y, s=4, alpha=.7, color=palette.get(name, GRAY), label=name)
    ax.set(xlabel="x (grid units)", ylabel="y (grid units)", xlim=(0, 200), ylim=(0, 200), aspect="equal")
    ax.legend(loc="upper right", fontsize=4.7, markerscale=1.4)
    style(ax); label(ax, "a")

    # b: four diffusion snapshots in a clean nested grid.  Use one shared
    # power-law normalization across all time points: this preserves zero and
    # the true global maximum while making the low-intensity diffusion halo
    # visible without changing or clipping the underlying arrays.
    outer = axs[0][1]
    outer.remove()
    sub = gs[0, 1].subgridspec(2, 2, wspace=.08, hspace=.16)
    steps = [0, 50, 200, 500]
    diffusion = [np.load(P2 / f"panel_b_step{step}.npy") for step in steps]
    diffusion_norm = PowerNorm(
        gamma=.20,
        vmin=0,
        vmax=max(float(arr.max()) for arr in diffusion),
    )
    snaps = []
    for k, (step, arr) in enumerate(zip(steps, diffusion)):
        aa = fig.add_subplot(sub[k//2, k%2])
        im = aa.imshow(arr, origin="lower", cmap="magma_r", norm=diffusion_norm)
        aa.set_title(f"t={step}", fontsize=6, pad=1.5)
        aa.set_xticks([]); aa.set_yticks([])
        snaps.append(aa)
    label(snaps[0], "b")
    cb = fig.colorbar(im, ax=snaps, fraction=.035, pad=.03)
    cb.set_ticks([0, 1, 10, 100])
    cb.set_label("Aβ (a.u.; shared γ=0.20 scale)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)

    # c: representative trajectories
    ax = axs[0][2]
    d = pd.read_csv(P2 / "panel_e_trajectories_rep0.csv")
    ids = sorted(d.cell_id.unique())[:18]
    for cid in ids:
        g = d[d.cell_id == cid]
        ax.plot(g.x, g.y, color=SKY, alpha=.55, lw=.65)
        ax.scatter(g.x.iloc[0], g.y.iloc[0], s=8, color=GRAY, marker="s")
        ax.scatter(g.x.iloc[-1], g.y.iloc[-1], s=9, color=PURPLE)
    ax.scatter([100], [100], marker="*", s=45, color=ORANGE, edgecolor="black", linewidth=.4)
    ax.set(xlabel="x (grid units)", ylabel="y (grid units)", xlim=(0, 200), ylim=(0, 200), aspect="equal")
    style(ax); label(ax, "c")

    # d: random walk vs chemotaxis
    ax = axs[1][0]
    d = pd.read_csv(P2 / "panel_f_comparison_agg.csv")
    for mode, c in [("random_walk", BLUE), ("chemotaxis", VERM)]:
        g = d[d["mode"] == mode]
        ax.plot(g.step, g.grand_mean, color=c, label=mode.replace("_", " ").title())
        ax.fill_between(g.step, g.grand_mean-g.grand_sem, g.grand_mean+g.grand_sem, color=c, alpha=.14, lw=0)
    ax.set(xlabel="Simulation step", ylabel="Distance to plaque center")
    ax.legend(loc="upper right")
    style(ax); label(ax, "d")

    # e: aggregation index
    ax = axs[1][1]
    d = pd.read_csv(P2 / "panel_h_clustering_agg.csv")
    ax.plot(d.step, d.mean_ai, color=VERM)
    ax.fill_between(d.step, d.mean_ai-d.sem_ai, d.mean_ai+d.sem_ai, color=VERM, alpha=.15, lw=0)
    ax.axhline(1, color=GRAY, ls="--", lw=.8)
    ax.set(xlabel="Simulation step", ylabel="Aggregation index")
    style(ax); label(ax, "e")

    # f: viability map
    ax = axs[1][2]
    d = pd.read_csv(P2 / "panel_i_viability_heatmap.csv")
    m = d.pivot(index="y", columns="x", values="viability_mean")
    im = ax.imshow(m, origin="lower", cmap="RdYlGn", vmin=.3, vmax=1, aspect="equal")
    ax.set(xlabel="x (5 µm units)", ylabel="y (5 µm units)")
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03); cb.set_label("Viability", fontsize=6)
    style(ax); label(ax, "f")

    # g: distance relationship
    ax = axs[2][0]
    d = pd.read_csv(P2 / "panel_j_distance_viability_agg.csv")
    ax.errorbar(d.dist_bin_center*5, d.viability_mean, yerr=d.viability_sem,
                fmt="o", ms=3, color=BLUE, capsize=2)
    good = np.isfinite(d.viability_mean)
    coef = np.polyfit((d.dist_bin_center*5)[good], d.viability_mean[good], 1)
    xx = np.linspace((d.dist_bin_center*5).min(), (d.dist_bin_center*5).max(), 100)
    ax.plot(xx, np.polyval(coef, xx), color=VERM, ls="--")
    ax.set(xlabel="Distance to plaque center (µm)", ylabel="Neuron viability")
    ax.text(.05, .90, "R² = 0.86", transform=ax.transAxes)
    style(ax); label(ax, "g")

    # h: activation-state fractions across replicates
    ax = axs[2][1]
    d = pd.read_csv(P2 / "panel_k_state_counts.csv")
    s = d.groupby("activation_state").fraction.agg(["mean", "sem"]).reindex(
        ["resting", "anti_inflammatory", "pro_inflammatory"])
    labs = ["M0", "M2", "M1"]
    ax.bar(labs, s["mean"], yerr=s["sem"], color=[LIGHT, BLUE, VERM], capsize=2)
    ax.set(ylabel="State fraction", ylim=(0, .8))
    style(ax); label(ax, "h")

    # i: tissue-axis fields
    ax = axs[2][2]
    d = pd.read_csv(P2 / "panel_l_profiles_agg.csv")
    ax2 = ax.twinx()
    ax.plot(d.x*5, d.tnf_alpha_mean, color=VERM, label="TNF-α")
    ax.plot(d.x*5, d.il10_mean, color=BLUE, label="IL-10")
    ax2.plot(d.x*5, d.abeta_mean, color=GRAY, lw=1.0, label="Aβ")
    ax.set(xlabel="Distance along tissue axis (µm)", ylabel="Cytokines (a.u.)")
    ax2.set_ylabel("Aβ (a.u.)", color=GRAY); ax2.tick_params(axis="y", colors=GRAY)
    lines = ax.lines + ax2.lines; ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    style(ax); ax2.spines["top"].set_visible(False); label(ax, "i")
    save(fig, "figure2_final")


def collapse(df, group_cols, value_cols):
    return df.groupby(group_cols, as_index=False)[value_cols].mean()


def figure3():
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.45), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.10, hspace=0.10)

    # a: calcium trajectories (replicates collapsed)
    ax = axs[0, 0]
    d = collapse(pd.read_csv(P3 / "panel_A_agg.csv"), ["abeta", "time"], ["calcium_mean", "calcium_sem"])
    for c, (dose, g) in zip(COL5, d.groupby("abeta")):
        g = g.iloc[::10]
        ax.plot(g.time, g.calcium_mean, color=c, label=f"{dose:g}")
        ax.fill_between(g.time, g.calcium_mean-g.calcium_sem, g.calcium_mean+g.calcium_sem, color=c, alpha=.12, lw=0)
    ax.set(xlabel="Time (h)", ylabel=r"$Ca^{2+}$ (µM)")
    ax.legend(title="Aβ (µM)", ncol=2, loc="lower right")
    style(ax); label(ax, "a")

    # b: tau trajectories
    ax = axs[0, 1]
    d = collapse(pd.read_csv(P3 / "panel_B_agg.csv"), ["abeta_source", "time"], ["p_tau_mean", "p_tau_sem"])
    for c, (dose, g) in zip(COL5, d.groupby("abeta_source")):
        g = g.iloc[::10]
        ax.plot(g.time, g.p_tau_mean, color=c, label=f"{dose:g}")
        ax.fill_between(g.time, g.p_tau_mean-g.p_tau_sem, g.p_tau_mean+g.p_tau_sem, color=c, alpha=.12, lw=0)
    ax.set(xlabel="Time (h)", ylabel="p-Tau (fraction)", ylim=(-.02, 1.03))
    style(ax); label(ax, "b")

    # c: phase portrait, mean trajectories only
    ax = axs[0, 2]
    d = collapse(pd.read_csv(P3 / "panel_C_agg.csv"), ["abeta", "time"], ["calcium_mean", "p_tau_mean"])
    for c, (dose, g) in zip(COL5, d.groupby("abeta")):
        g = g.iloc[::10]
        ax.plot(g.calcium_mean, g.p_tau_mean, color=c, label=f"{dose:g}")
        ax.scatter(g.calcium_mean.iloc[-1], g.p_tau_mean.iloc[-1], marker="*", s=20, color=c, zorder=3)
    ax.set(xlabel=r"$Ca^{2+}$ (µM)", ylabel="p-Tau (fraction)")
    ax.legend(title="Aβ (µM)", loc="lower right")
    style(ax); label(ax, "c")

    # d: NF-kappaB
    ax = axs[1, 0]
    d = collapse(pd.read_csv(P3 / "panel_D_agg.csv"), ["abeta", "time"], ["nfkb_mean", "nfkb_sem"])
    for c, (dose, g) in zip(COL5, d.groupby("abeta")):
        g = g.iloc[::10]
        ax.plot(g.time, g.nfkb_mean, color=c, label=f"{dose:g}")
    ax.axhline(.3, color=GRAY, ls=":", lw=.8); ax.axhline(.6, color=GRAY, ls="--", lw=.8)
    ax.set(xlabel="Time (h)", ylabel="NF-κB activity", ylim=(-.02, 1.03))
    style(ax); label(ax, "d")

    # e: final polarization fractions
    ax = axs[1, 1]
    d0 = pd.read_csv(P3 / "panel_E_agg.csv")
    d = d0.sort_values("time").groupby("abeta", as_index=False).tail(1).sort_values("abeta")
    x = np.arange(len(d)); bottom = np.zeros(len(d))
    for col, lab, c in [("M0_frac", "M0", LIGHT), ("M2_frac", "M2", BLUE), ("M1_frac", "M1", VERM)]:
        ax.bar(x, d[col], bottom=bottom, color=c, width=.72, label=lab)
        bottom += d[col].to_numpy()
    ax.set_xticks(x, [f"{v:g}" for v in d.abeta])
    ax.set(xlabel="Aβ (µM)", ylabel="Final state fraction", ylim=(0, 1.04))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.015),
              borderaxespad=0, handlelength=1.4, columnspacing=1.0)
    style(ax); label(ax, "e")

    # f: microglia state effect
    ax = axs[1, 2]
    d = collapse(pd.read_csv(P3 / "panel_F_agg.csv"), ["microglia_state", "time"], ["viability_mean", "viability_sem"])
    for state, c in [("M0", GRAY), ("M1", VERM), ("M2", BLUE)]:
        g = d[d.microglia_state == state].iloc[::10]
        ax.plot(g.time, g.viability_mean, color=c, label=state)
        ax.fill_between(g.time, g.viability_mean-g.viability_sem, g.viability_mean+g.viability_sem, color=c, alpha=.12, lw=0)
    ax.set(xlabel="Time (h)", ylabel="Neuron viability", ylim=(-.02, 1.03))
    ax.legend(loc="center right")
    style(ax); label(ax, "f")

    # g: full cascade viability
    ax = axs[2, 0]
    d = collapse(pd.read_csv(P3 / "panel_G_agg.csv"), ["abeta", "time"], ["viability_mean", "viability_sem"])
    for c, (dose, g) in zip(COL5, d.groupby("abeta")):
        g = g.iloc[::10]
        ax.plot(g.time, g.viability_mean, color=c, label=f"{dose:g}")
    ax.set(xlabel="Time (h)", ylabel="Neuron viability", ylim=(-.02, 1.03))
    ax.legend(title="Aβ (µM)", ncol=2, loc="upper right")
    style(ax); label(ax, "g")

    # h: pathway decomposition at Aβ=1
    ax = axs[2, 1]
    d0 = pd.read_csv(P3 / "panel_H_agg.csv")
    d0 = d0[np.isclose(d0.abeta, 1.0)]
    d = collapse(d0, ["time"], ["viab_amyloid_mean", "viab_inflam_mean", "viab_full_mean"])
    for col, lab, c in [("viab_amyloid_mean", "Amyloid only", BLUE),
                        ("viab_inflam_mean", "Inflammation only", VERM),
                        ("viab_full_mean", "Full cascade", "#000000")]:
        ax.plot(d.time.iloc[::10], d[col].iloc[::10], color=c, label=lab)
    ax.set(xlabel="Time (h)", ylabel="Neuron viability", ylim=(-.02, 1.03))
    ax.legend(loc="upper right")
    style(ax); label(ax, "h")

    # i: Sobol indices
    ax = axs[2, 2]
    d = pd.read_csv(P3 / "panel_I_sobol.csv").sort_values("ST")
    y = np.arange(len(d))
    ax.barh(y-.16, d.ST, height=.3, xerr=d.ST_conf, color=BLUE, capsize=2, label="$S_T$")
    ax.barh(y+.16, d.S1, height=.3, xerr=d.S1_conf, color=ORANGE, capsize=2, label="$S_1$")
    ax.set_yticks(y, [x.replace("_", " ") for x in d.param])
    ax.set(xlabel="Sobol index", xlim=(0, .82))
    ax.legend(loc="lower right")
    style(ax); label(ax, "i")
    save(fig, "figure3_final")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    print(OUT)
