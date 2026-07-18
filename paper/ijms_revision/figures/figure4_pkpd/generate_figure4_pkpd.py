# -*- coding: utf-8 -*-
"""Figure 4 -- Repository drug PK/PD implementation (3x3 layout, 9 panels).

Nature Methods style: Arial, 5-7pt, 0.4-0.5pt lines, 300dpi, PDF fonttype 42.

Layout:
  Row 1 (PK):  A Single-dose PK | B Multi-dose SS | C Route comparison
  Row 2 (PD):  D Dose-response   | E AChE inhibition | F NMDA blockade
  Row 3 (Int): G PK->PD effect   | H Bliss assumption | I Steady-state PD
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from scipy import stats
from pathlib import Path

# ── Nature Methods Style ───────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'axes.linewidth': 0.4,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'lines.linewidth': 0.8,
    'legend.fontsize': 5.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

PROC = Path(__file__).parent / "processed_data"
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Wong 2011)
C = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',
    'yellow': '#F0E442', 'black': '#000000', 'gray': '#999999',
}

DRUG_COLORS = {
    'aducanumab': C['blue'],
    'lecanemab': C['cyan'],
    'donepezil': C['green'],
    'memantine': C['orange'],
}
DRUG_LABELS = {
    'aducanumab': 'Aducanumab',
    'lecanemab': 'Lecanemab',
    'donepezil': 'Donepezil',
    'memantine': 'Memantine',
}

MM2IN = 1 / 25.4
fig = plt.figure(figsize=(180 * MM2IN, 190 * MM2IN))
gs = fig.add_gridspec(3, 3, hspace=0.44, wspace=0.46,
                      left=0.08, right=0.97, top=0.96, bottom=0.06)

# ════════════════════════════════════════════════════════════
# Panel A: Single-dose PK profiles (4 drugs, normalized time)
# ════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
df_a = pd.read_csv(PROC / "panel_a_pk_curves.csv")
summary_a = pd.read_csv(PROC / "panel_a_pk_summary.csv")

for _, row in summary_a.iterrows():
    name = row['drug']
    sub = df_a[df_a['drug'] == name]
    hl = row['half_life_h']
    # Normalize time to half-lives for comparison
    t_norm = sub['time_h'] / hl
    # Normalize concentration to Cmax
    cmax = row['Cmax']
    c_norm = sub['concentration'] / cmax if cmax > 0 else sub['concentration']
    mask = t_norm <= 8  # show up to 8 half-lives
    ax_a.plot(t_norm[mask], c_norm[mask], color=DRUG_COLORS[name],
              label=f"{DRUG_LABELS[name]}")

ax_a.set_xlabel(r'Time ($t_{1/2}$)')
ax_a.set_ylabel(r'$C / C_{max}$')
ax_a.set_xlim([0, 8])
ax_a.set_ylim([-0.02, 1.05])
ax_a.legend(fontsize=5, frameon=False, loc='upper right')

# ════════════════════════════════════════════════════════════
# Panel B: Multi-dose steady-state (Donepezil QD x 14 days)
# ════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])
df_b = pd.read_csv(PROC / "panel_b_multidose_curve.csv")
sum_b = pd.read_csv(PROC / "panel_b_multidose_summary.csv")

t_days = np.array(df_b['time_h']) / 24.0
ax_b.plot(t_days, df_b['concentration'] * 1000, color=C['green'], linewidth=0.8)
# Css,avg line
css = sum_b['Css_avg'].values[0] * 1000
ax_b.axhline(css, color=C['red'], linestyle='--', linewidth=0.5, alpha=0.7,
             label=f'$C_{{ss,avg}}$ = {css:.1f} ng/mL')
# Dose markers
for d in range(14):
    ax_b.axvline(d, color=C['gray'], linestyle=':', linewidth=0.2, alpha=0.4)

ax_b.set_xlabel('Time (days)')
ax_b.set_ylabel('Concentration (ng/mL)')
ax_b.set_xlim([0, t_days.max()])
ax_b.legend(fontsize=5, frameon=False, loc='upper left')

# ════════════════════════════════════════════════════════════
# Panel C: Route comparison (IV infusion vs oral)
# ════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[0, 2])
df_c = pd.read_csv(PROC / "panel_c_route_curves.csv")
sum_c = pd.read_csv(PROC / "panel_c_route_summary.csv")

for _, row in sum_c.iterrows():
    route = row['route']
    sub = df_c[df_c['route'] == route]
    t_days_c = sub['time_h'] / 24.0
    color = C['blue'] if 'IV' in route else C['orange']
    ls = '-' if 'IV' in route else '--'
    auc = row['AUC']
    ax_c.plot(t_days_c, sub['concentration'], color=color, linestyle=ls,
              label=f"{route}\nAUC={auc:.0f}")

ax_c.set_xlabel('Time (days)')
ax_c.set_ylabel('Concentration (mg/L)')
ax_c.legend(fontsize=4.5, frameon=False, loc='upper right')

# ════════════════════════════════════════════════════════════
# Panel D: Dose-response curves (4 drugs, Hill/Emax)
# ════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 0])
df_d = pd.read_csv(PROC / "panel_d_dose_response.csv")
sum_d = pd.read_csv(PROC / "panel_d_pd_summary.csv")

for _, row in sum_d.iterrows():
    name = row['drug']
    sub = df_d[df_d['drug'] == name]
    emax = row['Emax']
    # Normalize effect to fraction of Emax
    eff_norm = sub['effect'] / emax if emax > 0 else sub['effect']
    ax_d.plot(sub['concentration'], eff_norm, color=DRUG_COLORS[name],
              label=f"{DRUG_LABELS[name]} (EC$_{{50}}$={row['EC50']} mg/L)")

ax_d.set_xscale('log')
ax_d.set_xlabel('Concentration (mg/L)')
ax_d.set_ylabel(r'$E / E_{max}$')
ax_d.set_ylim([-0.02, 1.05])
ax_d.axhline(0.5, color=C['gray'], linestyle=':', linewidth=0.3, alpha=0.5)
ax_d.legend(fontsize=4.5, frameon=False, loc='lower right')

# ════════════════════════════════════════════════════════════
# Panel E: AChE competitive inhibition (Donepezil)
# ════════════════════════════════════════════════════════════
ax_e = fig.add_subplot(gs[1, 1])
df_e = pd.read_csv(PROC / "panel_e_ache_inhibition.csv")

inh_concs = sorted(df_e['donepezil_conc'].unique())
inh_colors = [C['blue'], C['cyan'], C['green'], C['orange'], C['red']]
for i, ic in enumerate(inh_concs):
    sub = df_e[df_e['donepezil_conc'] == ic]
    lbl = 'No inhibitor' if ic == 0 else f'[Don] = {ic} ug/mL'
    ax_e.plot(sub['ach_conc'], sub['effect'], color=inh_colors[i],
              linewidth=0.7, label=lbl)

ax_e.set_xlabel('[ACh] (μM)')
ax_e.set_ylabel('AChE Activity (frac.)')
ax_e.set_title('Competitive inhibition', fontsize=6, pad=3)
ax_e.legend(fontsize=4, frameon=False, loc='lower right')
ax_e.set_ylim(bottom=-0.01)

# ════════════════════════════════════════════════════════════
# Panel F: NMDA uncompetitive blockade (Memantine)
# ════════════════════════════════════════════════════════════
ax_f = fig.add_subplot(gs[1, 2])
df_f = pd.read_csv(PROC / "panel_f_nmda_blockade.csv")

inh_concs_f = sorted(df_f['memantine_conc'].unique())
for i, ic in enumerate(inh_concs_f):
    sub = df_f[df_f['memantine_conc'] == ic]
    lbl = 'No inhibitor' if ic == 0 else f'[Mem] = {ic} ug/mL'
    ax_f.plot(sub['glutamate_conc'], sub['effect'], color=inh_colors[i],
              linewidth=0.7, label=lbl)

ax_f.set_xlabel('[Glutamate] (μM)')
ax_f.set_ylabel('NMDA Response (frac.)')
ax_f.set_title('Uncompetitive inhibition', fontsize=6, pad=3)
ax_f.legend(fontsize=4, frameon=False, loc='lower right')
ax_f.set_ylim(bottom=-0.01)

# ════════════════════════════════════════════════════════════
# Panel G: PK->PD time-course (concentration & effect, dual y)
# ════════════════════════════════════════════════════════════
ax_g = fig.add_subplot(gs[2, 0])
df_g = pd.read_csv(PROC / "panel_g_pkpd_timecourse.csv")

# Show Donepezil as representative (oral, clear PK->PD lag)
sub_g = df_g[df_g['drug'] == 'donepezil']
t_days_g = np.array(sub_g['time_h']) / 24.0
cmax_g = sub_g['concentration'].max()
c_norm_g = sub_g['concentration'] / cmax_g if cmax_g > 0 else sub_g['concentration']

ax_g.plot(t_days_g, c_norm_g, color=C['green'], linewidth=0.8, label='Concentration')
ax_g.set_xlabel('Time (days)')
ax_g.set_ylabel(r'$C / C_{max}$', color=C['green'])
ax_g.tick_params(axis='y', labelcolor=C['green'])
ax_g.set_ylim([-0.02, 1.05])

ax_g2 = ax_g.twinx()
emax_g = sub_g['effect'].max()
e_norm_g = sub_g['effect'] / emax_g if emax_g > 0 else sub_g['effect']
ax_g2.plot(t_days_g, e_norm_g, color=C['red'], linewidth=0.8, linestyle='--', label='Effect')
ax_g2.set_ylabel(r'$E / E_{max}$', color=C['red'])
ax_g2.tick_params(axis='y', labelcolor=C['red'])
ax_g2.set_ylim([-0.02, 1.05])

# Combined legend
lines_g = [
    Line2D([0], [0], color=C['green'], linewidth=0.8, label='Concentration'),
    Line2D([0], [0], color=C['red'], linewidth=0.8, linestyle='--', label='Effect'),
]
ax_g.legend(handles=lines_g, fontsize=5, frameon=False, loc='upper right')
ax_g.set_title('Donepezil PK/PD', fontsize=6, pad=3)

# ════════════════════════════════════════════════════════════
# Panel H: Bliss response surface (explicit structural assumption)
# ════════════════════════════════════════════════════════════
ax_h = fig.add_subplot(gs[2, 1])
e_done = np.linspace(0, 1, 121)
e_mem = np.linspace(0, 1, 121)
X, Y = np.meshgrid(e_done, e_mem)
Z = 1 - (1 - X) * (1 - Y)
mesh = ax_h.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', vmin=0, vmax=1)
cont = ax_h.contour(X, Y, Z, levels=[0.25, 0.50, 0.75], colors='white',
                    linewidths=0.55, alpha=0.85)
ax_h.clabel(cont, inline=True, fontsize=5, fmt='%.2f')
done_ss, mem_ss = 0.6156801419, 0.1163241566
combo_ss = 1 - (1-done_ss)*(1-mem_ss)
ax_h.scatter(done_ss, mem_ss, s=28, marker='o', facecolor=C['red'],
             edgecolor='white', linewidth=0.7, zorder=4)
ax_h.annotate(f'steady state\n$E_{{combo}}$={combo_ss:.3f}',
              xy=(done_ss, mem_ss), xytext=(0.39, 0.35),
              arrowprops=dict(arrowstyle='-', lw=0.6, color='white'),
              color='white', fontsize=5.2, ha='center')
ax_h.set_xlabel('Donepezil fractional effect')
ax_h.set_ylabel('Memantine fractional effect')
ax_h.set_title('Bliss-assumed combination surface', fontsize=6, pad=3)
cb_h = fig.colorbar(mesh, ax=ax_h, fraction=0.046, pad=0.03)
cb_h.set_label(r'$E_{combo}$', fontsize=6)
cb_h.ax.tick_params(labelsize=5.5)

# ════════════════════════════════════════════════════════════
# Panel I: Repository steady-state PD effects
# ════════════════════════════════════════════════════════════
ax_i = fig.add_subplot(gs[2, 2])
pd_labels = ['Lecanemab', 'Donepezil', 'Memantine', 'Done+Mem']
# Values reproduced from repository PKModel/TwoCompartmentPKModel + PDModel
# at the dosing intervals used by the authoritative trial pipeline.
pd_values = [0.5828288643, 0.6156801419, 0.1163241566, 0.6603858252]
pd_colors = [C['cyan'], C['green'], C['orange'], C['purple']]
bars = ax_i.bar(np.arange(4), pd_values, color=pd_colors, width=0.68)
ax_i.set_xticks(np.arange(4))
ax_i.set_xticklabels(['Leca', 'Done', 'Mema', 'Done+Mem'], rotation=25, ha='right')
ax_i.set_ylabel('Fractional PD effect')
ax_i.set_ylim(0, 0.75)
for bar, value in zip(bars, pd_values):
    ax_i.text(bar.get_x()+bar.get_width()/2, value+0.02, f'{value:.3f}',
              ha='center', va='bottom', fontsize=5.5)
ax_i.text(0.02, 0.98, 'steady-state approximation\n(repository PK/PD parameters)',
          transform=ax_i.transAxes, ha='left', va='top', fontsize=5.2)

# ── Panel labels ───────────────────────────────────────────
labels = 'ABCDEFGHI'
all_axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i]
for letter, ax in zip(labels, all_axes):
    ax.text(-0.18, 1.12, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

# ── Save ───────────────────────────────────────────────────
for fmt in ['png', 'pdf']:
    outpath = OUT / f'figure4_pkpd_revised.{fmt}'
    fig.savefig(outpath, dpi=600, bbox_inches='tight')
    print(f'Saved: {outpath}')
plt.close(fig)
print('Figure 4 complete.')
