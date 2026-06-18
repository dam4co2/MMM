# %% Imports
import sys
sys.path.insert(0, "..")

from src.logging_utils import Tee
_log = Tee("../results/07_robeson.txt")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# %% Load raw dataset
df_raw = pd.read_csv("../data/dataset.csv", index_col=0)
print(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"Relevant columns present: {[c for c in ['CP','CS','RP','RS'] if c in df_raw.columns]}")

# %% Compute permeability targets (control and MMM)
def compute_permeability_targets(df):
    """Compute log-permeability targets and CO2/N2 selectivity from raw ratio columns.

    CP : absolute CO2 permeability of the control membrane
    CS : absolute CO2/N2 selectivity of the control membrane
    RP : MMM-to-control ratio for CO2 permeability
    RS : MMM-to-control ratio for CO2/N2 selectivity

    N2 permeability is derived as PN2 = PCO2 / S for both control and MMM.
    """
    df = df.copy()
    pco2_control = df["CP"]
    pco2_mmm = df["CP"] * df["RP"]
    s_control = df["CS"]
    s_mmm = df["CS"] * df["RS"]
    pn2_control = pco2_control / s_control
    pn2_mmm = pco2_mmm / s_mmm
    df["PCO2_Control"] = np.log10(pco2_control)
    df["PCO2_MMM"] = np.log10(pco2_mmm)
    df["PN2_Control"] = np.log10(pn2_control)
    df["PN2_MMM"] = np.log10(pn2_mmm)
    df["alpha_Control"] = s_control
    df["alpha_MMM"] = s_mmm
    return df.drop(columns=["CP", "CS", "RP", "RS"])

df = compute_permeability_targets(df_raw)
print(f"\nProcessed dataset: {df.shape}")
print(f"PCO2_Control  — min: {df['PCO2_Control'].min():.2f}  max: {df['PCO2_Control'].max():.2f}  (log10 Barrer)")
print(f"PCO2_MMM      — min: {df['PCO2_MMM'].min():.2f}  max: {df['PCO2_MMM'].max():.2f}  (log10 Barrer)")
print(f"alpha_Control — min: {df['alpha_Control'].min():.1f}  max: {df['alpha_Control'].max():.1f}")
print(f"alpha_MMM     — min: {df['alpha_MMM'].min():.1f}  max: {df['alpha_MMM'].max():.1f}")

# %% Convert log10 permeability back to linear for the Robeson axes
pco2_ctrl   = 10 ** df["PCO2_Control"]
alpha_ctrl  = df["alpha_Control"]
pco2_mmm_   = 10 ** df["PCO2_MMM"]
alpha_mmm_  = df["alpha_MMM"]

# %% Robeson upper-bound parameters for CO2/N2
# Form: P_CO2 = k * alpha^n   (n < 0)
upperbounds_co2_n2 = {
    "2008": [30_967_000,    -2.89],
    "2019": [755_584_213,   -3.41],
}

# Colour palette — harmonious, publication-ready
CTRL_COLOR  = "#2C7BB6"   # steel blue  — control membranes
MMM_COLOR   = "#1A9641"   # forest green — MMM
UB_COLORS   = ["#D7191C", "#FDAE61"]   # red gradient for upper bounds (oldest → newest)
GRID_COLOR  = "#CCCCCC"

# %% Build figure
fig, ax = plt.subplots(figsize=(9, 6.5))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

# ---------- scatter: control membranes ----------
sc_ctrl = ax.scatter(
    pco2_ctrl, alpha_ctrl,
    color=CTRL_COLOR, s=55, alpha=0.70,
    edgecolors="white", linewidths=0.6,
    marker="o", zorder=4,
    label="Control membrane"
)

# ---------- scatter: MMM ----------
sc_mmm = ax.scatter(
    pco2_mmm_, alpha_mmm_,
    color=MMM_COLOR, s=55, alpha=0.70,
    edgecolors="white", linewidths=0.6,
    marker="D", zorder=4,
    label="MMM"
)

# ---------- Robeson upper bounds ----------
all_p = np.concatenate([pco2_ctrl.values, pco2_mmm_.values])
all_a = np.concatenate([alpha_ctrl.values, alpha_mmm_.values])

alpha_lo = all_a.min() * 0.25
alpha_hi = all_a.max() * 6.0
alpha_range = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), 1_500)

ub_lines = []
for i, (year, (k, n)) in enumerate(upperbounds_co2_n2.items()):
    p_ub = k * alpha_range ** n          # upper-bound permeability for each alpha
    lh, = ax.plot(
        p_ub, alpha_range,
        linestyle="--", color=UB_COLORS[i], linewidth=2.2, zorder=2,
        label=f"Robeson {year} upper bound"
    )
    ub_lines.append(lh)

# ---------- axes scale, limits, labels ----------
ax.set_xscale("log")
ax.set_yscale("log")

p_lo = all_p.min() * 0.15
p_hi = all_p.max() * 8.0
a_lo = all_a.min() * 0.25
a_hi = all_a.max() * 6.0

ax.set_xlim(p_lo, p_hi)
ax.set_ylim(a_lo, a_hi)

ax.set_xlabel(r"$P_{\mathrm{CO_2}}$ (Barrer)", fontsize=13, labelpad=8)
ax.set_ylabel(r"$\alpha_{\mathrm{CO_2/N_2}}$", fontsize=13, labelpad=8)
ax.set_title(
    r"Robeson Plot — $\mathrm{CO_2/N_2}$ Separation" "\nControl vs. Mixed-Matrix Membranes",
    fontsize=14, fontweight="bold", pad=12
)

# ---------- grid ----------
ax.grid(True, which="major", linestyle="-",  linewidth=0.5, color=GRID_COLOR, alpha=0.8, zorder=1)
ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, color=GRID_COLOR, alpha=0.5, zorder=1)
ax.tick_params(axis="both", labelsize=11)

# ---------- legend ----------
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CTRL_COLOR,
           markersize=9, markeredgecolor="white", label="Control membrane"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=MMM_COLOR,
           markersize=9, markeredgecolor="white", label="MMM"),
] + [
    Line2D([0], [0], linestyle="--", color=UB_COLORS[i], linewidth=2.2,
           label=f"Robeson {year} upper bound")
    for i, year in enumerate(upperbounds_co2_n2)
]

ax.legend(
    handles=legend_handles,
    loc="upper right", fontsize=10,
    framealpha=0.92, edgecolor="#AAAAAA",
    borderpad=0.8, handlelength=1.8
)

plt.tight_layout()
plt.savefig("../images/robeson.png", dpi=180, bbox_inches="tight")
plt.close()
print("\nSaved: ../images/robeson.png")

# =============================================================================
# %% Figure 2 — Robeson plot colored by polymer (● control, ◆ MMM, same color)
# =============================================================================

# Working frame with linear permeability/selectivity
plot_df = df[["polymer_type", "PCO2_Control", "PCO2_MMM",
              "alpha_Control", "alpha_MMM"]].copy()
plot_df["x_ctrl"] = 10 ** plot_df["PCO2_Control"]
plot_df["y_ctrl"] = plot_df["alpha_Control"]
plot_df["x_mmm"]  = 10 ** plot_df["PCO2_MMM"]
plot_df["y_mmm"]  = plot_df["alpha_MMM"]

# Top-N polymers by row count; rest → 'Other'
TOP_N = 12
top_polymers = plot_df["polymer_type"].value_counts().head(TOP_N).index.tolist()
plot_df["polymer_group"] = plot_df["polymer_type"].where(
    plot_df["polymer_type"].isin(top_polymers), other="Other"
)

print(f"\nPolymers in the colored legend ({TOP_N}):")
for p in top_polymers:
    n = (plot_df["polymer_type"] == p).sum()
    print(f"  {p}: {n} data points")
print(f"  Other: {(plot_df['polymer_group'] == 'Other').sum()} data points")

# Harmonious qualitative palette — 12 vivid colors from tab20
tab20 = plt.cm.tab20.colors
POLY_COLORS = {p: tab20[i * 2 % 20 + (1 if i >= 10 else 0)]
               for i, p in enumerate(top_polymers)}
POLY_COLORS["Other"] = "#BBBBBB"

all_p2 = np.concatenate([plot_df["x_ctrl"].values, plot_df["x_mmm"].values])
all_a2 = np.concatenate([plot_df["y_ctrl"].values, plot_df["y_mmm"].values])

fig2, ax2 = plt.subplots(figsize=(11, 7.5))
fig2.patch.set_facecolor("#FAFAFA")
ax2.set_facecolor("#FAFAFA")

# Plot order: Other first (background), then top polymers
ordered_groups = ["Other"] + top_polymers

for group in ordered_groups:
    mask = plot_df["polymer_group"] == group
    sub  = plot_df[mask]
    col  = POLY_COLORS[group]
    zord = 3 if group == "Other" else 5
    alph = 0.35 if group == "Other" else 0.70

    # MMM — diamonds (◆)
    ax2.scatter(sub["x_mmm"], sub["y_mmm"],
                color=col, s=35, alpha=alph,
                edgecolors="white", linewidths=0.4,
                marker="D", zorder=zord)

    # Control — circles (●), slightly larger and more opaque
    ax2.scatter(sub["x_ctrl"], sub["y_ctrl"],
                color=col, s=50, alpha=min(alph + 0.15, 1.0),
                edgecolors="white", linewidths=0.5,
                marker="o", zorder=zord + 1)

# --- Robeson upper bounds ---
alpha_lo2 = all_a2.min() * 0.25
alpha_hi2 = all_a2.max() * 6.0
alpha_rng2 = np.logspace(np.log10(alpha_lo2), np.log10(alpha_hi2), 1_500)

for i, (year, (k, n)) in enumerate(upperbounds_co2_n2.items()):
    p_ub2 = k * alpha_rng2 ** n
    ax2.plot(p_ub2, alpha_rng2, "--",
             color=UB_COLORS[i], linewidth=2.2, zorder=7,
             label=f"Robeson {year} upper bound")

# --- Axes ---
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(all_p2.min() * 0.15, all_p2.max() * 8.0)
ax2.set_ylim(all_a2.min() * 0.25, all_a2.max() * 6.0)
ax2.set_xlabel(r"$P_{\mathrm{CO_2}}$ (Barrer)", fontsize=13, labelpad=8)
ax2.set_ylabel(r"$\alpha_{\mathrm{CO_2/N_2}}$", fontsize=13, labelpad=8)
ax2.set_title(
    r"Robeson Plot — $\mathrm{CO_2/N_2}$ Separation" "\nColored by polymer matrix  (● control,  ◆ MMM)",
    fontsize=14, fontweight="bold", pad=12
)
ax2.grid(True, which="major", linestyle="-",  linewidth=0.5, color=GRID_COLOR, alpha=0.8, zorder=1)
ax2.grid(True, which="minor", linestyle=":",  linewidth=0.3, color=GRID_COLOR, alpha=0.5, zorder=1)
ax2.tick_params(axis="both", labelsize=11)

# --- Legend: polymer colors ---
poly_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=POLY_COLORS[p],
           markersize=8, markeredgecolor="white", label=p)
    for p in top_polymers
] + [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBBBBB",
           markersize=7, markeredgecolor="none", label="Other polymers")
]
legend2a = ax2.legend(
    handles=poly_handles,
    title="Polymer", title_fontsize=9,
    loc="upper right", fontsize=8,
    framealpha=0.93, edgecolor="#AAAAAA",
    borderpad=0.7, handlelength=1.2, ncol=1
)
ax2.add_artist(legend2a)

# --- Legend: marker types + upper bounds ---
type_ub_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
           markersize=9, markeredgecolor="white", label="Control membrane (●)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#555555",
           markersize=7, markeredgecolor="white", label="MMM (◆)"),
] + [
    Line2D([0], [0], linestyle="--", color=UB_COLORS[i], linewidth=2.2,
           label=f"Robeson {year} upper bound")
    for i, year in enumerate(upperbounds_co2_n2)
]
ax2.legend(handles=type_ub_handles, loc="lower left", fontsize=9,
           framealpha=0.93, edgecolor="#AAAAAA", borderpad=0.7, handlelength=1.8)

plt.tight_layout()
plt.savefig("../images/robeson_by_polymer.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: ../images/robeson_by_polymer.png")

# =============================================================================
# %% Figure 3 — Arrow plot: each row as a vector (control -> MMM) on Robeson
# =============================================================================
# Every data point is a (control, MMM) pair already — draw a segment for each.
# Color by polymer group. Arrow tip = MMM end.

fig3, ax3 = plt.subplots(figsize=(11, 7.5))
fig3.patch.set_facecolor("#FAFAFA")
ax3.set_facecolor("#FAFAFA")

# Draw arrows in log-space so direction is meaningful on the log-log axes.
# We use ax3.annotate with arrowprops drawn in data coordinates.
for group in ordered_groups:
    mask = plot_df["polymer_group"] == group
    sub  = plot_df[mask]
    col  = POLY_COLORS[group]
    alpha_arrow = 0.25 if group == "Other" else 0.55
    lw_arrow    = 0.6  if group == "Other" else 0.9

    for _, row in sub.iterrows():
        ax3.annotate(
            "",
            xy=(row["x_mmm"],  row["y_mmm"]),    # arrowhead = MMM
            xytext=(row["x_ctrl"], row["y_ctrl"]), # tail = control
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                lw=lw_arrow,
                alpha=alpha_arrow,
                mutation_scale=6,
            ),
            zorder=4 if group != "Other" else 2,
        )

# Scatter control points on top of arrows
for group in ordered_groups:
    mask = plot_df["polymer_group"] == group
    sub  = plot_df[mask]
    col  = POLY_COLORS[group]
    zord = 3 if group == "Other" else 6
    ax3.scatter(sub["x_ctrl"], sub["y_ctrl"],
                color=col, s=30, alpha=0.85,
                edgecolors="white", linewidths=0.5,
                marker="o", zorder=zord)

# Upper bounds
for i, (year, (k, n)) in enumerate(upperbounds_co2_n2.items()):
    p_ub3 = k * alpha_rng2 ** n
    ax3.plot(p_ub3, alpha_rng2, "--",
             color=UB_COLORS[i], linewidth=2.2, zorder=7,
             label=f"Robeson {year} upper bound")

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlim(all_p2.min() * 0.15, all_p2.max() * 8.0)
ax3.set_ylim(all_a2.min() * 0.25, all_a2.max() * 6.0)
ax3.set_xlabel(r"$P_{\mathrm{CO_2}}$ (Barrer)", fontsize=13, labelpad=8)
ax3.set_ylabel(r"$\alpha_{\mathrm{CO_2/N_2}}$", fontsize=13, labelpad=8)
ax3.set_title(
    r"Robeson Plot — $\mathrm{CO_2/N_2}$ Separation" "\nVectors: control membrane $\longrightarrow$ MMM (per data point)",
    fontsize=14, fontweight="bold", pad=12
)
ax3.grid(True, which="major", linestyle="-",  linewidth=0.5, color=GRID_COLOR, alpha=0.8, zorder=1)
ax3.grid(True, which="minor", linestyle=":",  linewidth=0.3, color=GRID_COLOR, alpha=0.5, zorder=1)
ax3.tick_params(axis="both", labelsize=11)

# Legend: polymers + upper bounds
arr_poly_handles = []
for group in top_polymers:
    col = POLY_COLORS[group]
    arr_poly_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
               markersize=8, markeredgecolor="white", label=group)
    )
arr_poly_handles.append(
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBBBBB",
           markersize=7, markeredgecolor="none", label="Other polymers")
)
legend3a = ax3.legend(
    handles=arr_poly_handles,
    title="Polymer", title_fontsize=9,
    loc="upper right", fontsize=8,
    framealpha=0.93, edgecolor="#AAAAAA",
    borderpad=0.7, handlelength=1.2, ncol=1
)
ax3.add_artist(legend3a)

ub_handles3 = [
    Line2D([0], [0], linestyle="--", color=UB_COLORS[i], linewidth=2.2,
           label=f"Robeson {year} upper bound")
    for i, year in enumerate(upperbounds_co2_n2)
] + [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
           markersize=8, markeredgecolor="white", label="Control membrane (●)"),
    Line2D([0], [0], linestyle="-", color="#555555", linewidth=1.2,
           label=r"Displacement $\to$ MMM"),
]
ax3.legend(handles=ub_handles3, loc="lower left", fontsize=9,
           framealpha=0.93, edgecolor="#AAAAAA", borderpad=0.7, handlelength=1.8)

plt.tight_layout()
plt.savefig("../images/robeson_arrows.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: ../images/robeson_arrows.png")

# =============================================================================
# %% Figure 4 — Normalized RP x RS plot (all controls at (1,1))
# =============================================================================
# x = RP  (MMM/control CO2 permeability ratio)
# y = RS  (MMM/control CO2/N2 selectivity ratio)
# Reference: (1,1) = no change from control

rp = df_raw["RP"]
rs = df_raw["RS"]
polymer_raw = df_raw["polymer_type"]
polymer_group_raw = polymer_raw.where(polymer_raw.isin(top_polymers), other="Other")

fig4, ax4 = plt.subplots(figsize=(9, 7))
fig4.patch.set_facecolor("#FAFAFA")
ax4.set_facecolor("#FAFAFA")

# Reference lines at (1,1)
ax4.axvline(1.0, color="#888888", linewidth=1.0, linestyle="--", zorder=2, alpha=0.7)
ax4.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--", zorder=2, alpha=0.7)

# Quadrant labels
ax4_xlim = (rp.min() * 0.5, rp.max() * 2.5)
ax4_ylim = (rs.min() * 0.5, rs.max() * 2.5)

quad_kw = dict(fontsize=8, alpha=0.45, ha="center", va="center",
               fontstyle="italic", color="#444444")
ax4.text(ax4_xlim[0] * 2.5, ax4_ylim[1] * 0.65,
         "higher selectivity\nlower permeability", **quad_kw)
ax4.text(ax4_xlim[1] * 0.55, ax4_ylim[1] * 0.65,
         "higher selectivity\nhigher permeability", **quad_kw)
ax4.text(ax4_xlim[0] * 2.5, ax4_ylim[0] * 1.9,
         "lower selectivity\nlower permeability", **quad_kw)
ax4.text(ax4_xlim[1] * 0.55, ax4_ylim[0] * 1.9,
         "lower selectivity\nhigher permeability", **quad_kw)

# Scatter
for group in ordered_groups:
    mask = polymer_group_raw == group
    col  = POLY_COLORS[group]
    zord = 3 if group == "Other" else 5
    alph = 0.35 if group == "Other" else 0.70
    ax4.scatter(rp[mask], rs[mask],
                color=col, s=40, alpha=alph,
                edgecolors="white", linewidths=0.5,
                marker="o", zorder=zord)

# Control reference point
ax4.scatter([1.0], [1.0], color="#222222", s=220, marker="*",
            edgecolors="white", linewidths=0.8, zorder=8,
            label="Control (reference)")

ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlim(ax4_xlim)
ax4.set_ylim(ax4_ylim)
ax4.set_xlabel(r"$R_P = P_{\mathrm{CO_2}}^{\mathrm{MMM}} \,/\, P_{\mathrm{CO_2}}^{\mathrm{ctrl}}$",
               fontsize=13, labelpad=8)
ax4.set_ylabel(r"$R_S = \alpha_{\mathrm{CO_2/N_2}}^{\mathrm{MMM}} \,/\, \alpha_{\mathrm{CO_2/N_2}}^{\mathrm{ctrl}}$",
               fontsize=13, labelpad=8)
ax4.set_title(
    r"Normalized performance plot: MMM vs. control" "\n"
    r"All controls at $\bigstar\,(1,1)$ — distance shows MOF contribution",
    fontsize=14, fontweight="bold", pad=12
)
ax4.grid(True, which="major", linestyle="-",  linewidth=0.5, color=GRID_COLOR, alpha=0.8, zorder=1)
ax4.grid(True, which="minor", linestyle=":",  linewidth=0.3, color=GRID_COLOR, alpha=0.5, zorder=1)
ax4.tick_params(axis="both", labelsize=11)

# Legend
norm_poly_handles = []
for group in top_polymers:
    norm_poly_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POLY_COLORS[group],
               markersize=8, markeredgecolor="white", label=group)
    )
norm_poly_handles.append(
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBBBBB",
           markersize=7, markeredgecolor="none", label="Other polymers")
)
norm_poly_handles.append(
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#222222",
           markersize=13, markeredgecolor="white", label="Control reference (1,1)")
)
ax4.legend(
    handles=norm_poly_handles,
    title="Polymer", title_fontsize=9,
    loc="upper right", fontsize=8,
    framealpha=0.93, edgecolor="#AAAAAA",
    borderpad=0.7, handlelength=1.2, ncol=1
)

plt.tight_layout()
plt.savefig("../images/robeson_normalized.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: ../images/robeson_normalized.png")

_log.close()
