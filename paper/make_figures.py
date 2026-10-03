"""Re-renders the paper figures at publication size from the committed AEOLUS-RAMS outputs."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
for p in ("phase-6-eta", "phase-7-preventive-maintenance", "phase-8-dynamic-modeling"):
    sys.path.insert(0, str(ROOT / p))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 300,
})
# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMIL, SKY, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#7f7f7f"
W = 4.8  # LNCS text width in inches (12.2 cm)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def box(ax, x, y, w, h, title, body, fc, tsize=6.6, bsize=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.04", fc=fc, ec="#333", lw=0.6))
    ax.text(x + 0.06, y + h - 0.07, title, fontsize=tsize, fontweight="bold", va="top", ha="left")
    ax.text(x + 0.06, y + h - 0.20, body, fontsize=bsize, va="top", ha="left", linespacing=1.25)


def arrow(ax, p, q, color="#333", lw=0.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ Fig 1: architecture
def fig_architecture():
    H = 3.25
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 4.92); ax.set_ylim(0, H); ax.axis("off")
    box(ax, 0.0, 2.72, 2.25, 0.52, "Evidence", "CARE-to-Compare: 3 farms, 95 event sets\n(44 anomaly / 51 normal), 36 turbines", "#E8F1FA", bsize=5.6)
    box(ax, 2.35, 2.72, 2.45, 0.52, "Priors and parameters (P2-P8)", "Literature failure rates, repair times, costs,\nIEC 61508/61511 targets (cited in config files)", "#FDF2DC", bsize=5.6)
    ax.text(1.0, 2.60, "CHARACTERISE (P1-P4)", fontsize=6.0, fontweight="bold", ha="left", va="center", color=BLUE)
    ax.text(2.55, 2.60, "DECIDE (P5-P8)", fontsize=6.0, fontweight="bold", ha="left", va="center", color=VERMIL)
    L = [("P1  System definition + FMECA", "log tagging (34 descriptions, 44 events)\n13-component taxonomy; RPN = S x O x D"),
         ("P2  Survival analysis", "linkage, censored TBF, A/B/C tiers\nAIC, bootstrap, Gamma-Poisson update"),
         ("P3  Reliability block diagram", "13-comp. series model, R(t), IB / IC\nk-of-N farm, placeholder sensitivity"),
         ("P4  Monte Carlo availability", "renewal sim., weather-gated MTTR\n10,000 farm-years x 3 scenarios, tornado")]
    R = [("P5  Fault tree analysis", "overspeed AND of 3 OR-gates; 12 MCS\nFV importance; beta-factor CCF; MC on Q"),
         ("P6  Event tree + risk screening", "4 consequence branches, lambda = IE x P\nscreening matrix; 3 risk-reduction options"),
         ("P7  RCM / preventive maint.", "Barlow-Proschan T*, PFDavg proof-test\ninterval, CBM trigger; 13-row schedule"),
         ("P8  Dynamic reliability", "Bayesian Weibull grid, Cox PH, CBM\nthreshold sweep (synthetic data, flagged)")]
    bh = 0.5
    ys = [1.95, 1.37, 0.79, 0.21]
    for i, (t, b) in enumerate(L):
        box(ax, 0.0, ys[i], 2.25, bh, t, b, "#DCEBF7")
    for i, (t, b) in enumerate(R):
        box(ax, 2.55, ys[i], 2.25, bh, t, b, "#FBE3D6")
    for i in range(3):
        arrow(ax, (1.12, ys[i]), (1.12, ys[i + 1] + bh))
        arrow(ax, (3.67, ys[i]), (3.67, ys[i + 1] + bh))
    ax.plot([2.25, 2.40, 2.40], [ys[3] + bh / 2, ys[3] + bh / 2, ys[0] + bh / 2], color="#333", lw=0.8)
    arrow(ax, (2.40, ys[0] + bh / 2), (2.55, ys[0] + bh / 2))
    arrow(ax, (0.9, 2.72), (0.9, ys[0] + bh))
    ax.plot([4.87, 4.87], [2.72, ys[3] + bh / 2], color=ORANGE, lw=0.8, ls="--")
    ax.plot([4.8, 4.87], [2.72, 2.72], color=ORANGE, lw=0.8, ls="--")
    for yy in ys:
        arrow(ax, (4.87, yy + bh / 2), (4.80, yy + bh / 2), color=ORANGE, lw=0.7)
    ax.add_patch(FancyBboxPatch((0.0, -0.02), 4.8, 0.16, boxstyle="round,pad=0.0,rounding_size=0.03", fc="#F5F5F5", ec=GREEN, lw=0.9, ls="--"))
    ax.text(2.4, 0.06, "Cross-cutting: CSV hand-offs | cited constants | confidence tags | numerical bridges | sensitivity checks",
            fontsize=5.4, ha="center", va="center", color="#115e45")
    save(fig, "fig1_architecture.png")


# ------------------------------------------------------------------ Fig 2: FTA + ETA chain
def fig_safety_chain():
    H = 2.25
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 4.8); ax.set_ylim(0, H); ax.axis("off")
    gate = pd.read_csv(ROOT / "phase-5-fta/outputs/gate_Q_table.csv")
    g = gate[(gate.t == "365d") & (gate.node_type == "Gate")]
    gq = {n.strip().split(":")[0]: q for n, q in zip(g.node, g.Q)}
    gx = [("G1 Pitch fails to feather", f"OR gate, Q = {gq['G1']:.4f}", "encoder 0.082 | battery 0.064\nactuator 0.036", 1.56),
          ("G2 Brake fails to engage", f"OR gate, Q = {gq['G2']:.4f}", "hydraulic supply 0.108\nbrake hardware 0.032", 0.89),
          ("G3 Overspeed trip fails", f"OR gate, Q = {gq['G3']:.4f}", "fieldbus 0.006 | safety relay 0.027", 0.22)]
    for t, q, sub, y in gx:
        ax.add_patch(FancyBboxPatch((0.0, y), 1.85, 0.62, boxstyle="round,pad=0,rounding_size=0.04", fc="#DCEBF7", ec="#333", lw=0.6))
        ax.text(0.06, y + 0.57, t, fontsize=6.0, fontweight="bold", va="top")
        ax.text(0.06, y + 0.42, q, fontsize=5.6, va="top")
        ax.text(0.06, y + 0.29, sub, fontsize=5.0, va="top", color="#333", linespacing=1.2)
        arrow(ax, (1.85, y + 0.31), (2.12, 1.17))
    ax.add_patch(FancyBboxPatch((2.12, 0.65), 0.98, 1.05, boxstyle="round,pad=0,rounding_size=0.05", fc="#F4C7B8", ec=VERMIL, lw=0.9))
    ax.text(2.61, 1.64, "AND: top event", fontsize=5.8, fontweight="bold", ha="center", va="top")
    ax.text(2.61, 1.47, f"Q_top = {gq['TE']:.2e}", fontsize=5.5, ha="center", va="top")
    ax.text(2.61, 1.31, "with CCF (beta 0.1):", fontsize=5.0, ha="center", va="top", color=VERMIL)
    ax.text(2.61, 1.19, "1.21e-3", fontsize=5.2, ha="center", va="top", color=VERMIL)
    ax.text(2.61, 0.99, "12 minimal cut\nsets, all order 3", fontsize=5.0, ha="center", va="top")
    ax.text(4.05, 2.23, "Event tree: lambda_IE = 0.5 / turbine-yr", fontsize=5.8, fontweight="bold", ha="center", va="top")
    br = [("B1 safe: pitch feathers", "P = 0.8281 | 0.414 /yr", "class E, F1: acceptable", "#D6EFE3", 1.65),
          ("B2 safe: brake engages", "P = 0.1485 | 0.0742 /yr", "class D, F2: acceptable", "#D6EFE3", 1.14),
          ("B3 near-miss: trip acts", "P = 0.0226 | 0.0113 /yr", "class C, F2: ALARP zone", "#FCE7B8", 0.63),
          ("B4 catastrophic", "P = 7.69e-4 | 3.85e-4 /yr", "class A, F4: ALARP zone", "#F4C7B8", 0.12)]
    for t, p, cls, c, y in br:
        ax.add_patch(FancyBboxPatch((3.3, y), 1.5, 0.46, boxstyle="round,pad=0,rounding_size=0.04", fc=c, ec="#333", lw=0.6))
        ax.text(3.35, y + 0.42, t, fontsize=5.6, fontweight="bold", va="top")
        ax.text(3.35, y + 0.28, p, fontsize=5.0, va="top")
        ax.text(3.35, y + 0.16, cls, fontsize=5.0, va="top")
        arrow(ax, (3.10, 1.17), (3.3, y + 0.23), lw=0.6)
    save(fig, "fig2_safety_chain.png")


# ------------------------------------------------------------------ Fig 3: criticality + availability
SHORT = {"Pitch System": "Pitch", "Hydraulic System": "Hydraulic", "Cooling System": "Cooling", "Gearbox": "Gearbox",
         "Main/Rotor Bearing": "Main bearing", "Mechanical Brake": "Mech. brake", "Electrical Safety System": "Elec. safety",
         "Generator": "Generator", "Grounding/Lightning Protection": "Grounding", "Transformer": "Transformer",
         "Converter": "Converter", "SCADA/Communication": "SCADA/comm.", "Yaw System": "Yaw"}


def fig_evidence_avail():
    imp = pd.read_csv(ROOT / "phase-3-rbd/outputs/importance_table.csv")
    t = pd.read_csv(ROOT / "phase-4-monte-carlo/outputs/tornado_data.csv")
    s = pd.read_csv(ROOT / "phase-4-monte-carlo/outputs/mc_summary.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.65), gridspec_kw={"wspace": 0.95, "width_ratios": [1.0, 0.85, 0.75]})
    ax = axs[0]
    d = imp.sort_values("IC_1825d", kind="stable")
    cmap = {"fitted_tier_a": BLUE, "fitted_tier_b": SKY, "posterior_informed": GREEN, "assumed_placeholder": ORANGE}
    hmap = {"fitted_tier_a": "", "fitted_tier_b": "////", "posterior_informed": "xxxx", "assumed_placeholder": "...."}
    bars = ax.barh([SHORT[c] for c in d.component], d.IC_1825d * 100, color=[cmap[c] for c in d.confidence], height=0.7)
    for b_, c_ in zip(bars, d.confidence):
        b_.set_hatch(hmap[c_]); b_.set_edgecolor("white"); b_.set_linewidth(0)
    ax.set_xlabel("criticality importance $IC_i$ (%)"); ax.set_title("(a) Turbine failure-rate share", loc="left", fontsize=6.2)
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor="white", hatch="", label="fitted A"),
                       Patch(facecolor=SKY, edgecolor="white", hatch="////", label="fitted B"),
                       Patch(facecolor=GREEN, edgecolor="white", hatch="xxxx", label="posterior"),
                       Patch(facecolor=ORANGE, edgecolor="white", hatch="....", label="placeholder")],
              loc="lower right", frameon=False, handlelength=1.1, borderaxespad=0.1, fontsize=5.2)
    ax.tick_params(axis="y", labelsize=5.4)
    ax = axs[1]
    base = t.baseline_A_farm.iloc[0]
    lab = {"cable": "Cable MTBF*", "mttr": "MTTR mult.", "pitch": "Pitch MTBF", "hydraulic": "Hyd. MTBF"}
    t = t.sort_values("impact_total").reset_index(drop=True)
    for i, r in enumerate(t.itertuples()):
        ax.barh(i, base - r.A_farm_at_worst, left=r.A_farm_at_worst, color=VERMIL, height=0.6)
        ax.barh(i, r.A_farm_at_best - base, left=base, color=GREEN, height=0.6)
        ax.text(r.A_farm_at_best + 0.003, i, f"{r.impact_total:.3f}", va="center", fontsize=5.4)
    ax.set_yticks(range(len(t))); ax.set_yticklabels([lab[n] for n in t.sweep_name], fontsize=5.4); ax.axvline(base, color="k", lw=0.6)
    ax.set_xlim(0.83, 1.02); ax.set_xticks([0.85, 0.95]); ax.set_xlabel("mean farm availability")
    ax.set_title("(b) Sensitivity (Phase 4)", loc="left", fontsize=6.2)
    ax = axs[2]
    sc = ["baseline", "optimised", "degraded"]; x = np.arange(3); wd = 0.38  # csv keys; paper calls the 0.6x case improved access
    af = [s[(s.scenario == k) & (s.metric == "A_farm")]["mean"].iloc[0] for k in sc]
    at = [s[(s.scenario == k) & (s.metric == "A_turbine_mean")]["mean"].iloc[0] for k in sc]
    ax.bar(x - wd / 2, af, wd, color=BLUE, label="farm"); ax.bar(x + wd / 2, at, wd, color=SKY, label="turbine")
    for xi, a, b in zip(x, af, at):
        ax.text(xi - wd / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=4.8, rotation=90, va="bottom")
        ax.text(xi + wd / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=4.8, rotation=90, va="bottom")
    ax.set_ylim(0.85, 1.08); ax.set_yticks([0.85, 0.90, 0.95, 1.0]); ax.set_xticks(x); ax.set_xticklabels(["x1.0", "x0.6", "x2.5"])
    ax.set_xlabel("repair factor"); ax.set_ylabel("mean availability")
    ax.legend(loc="upper right", frameon=False, fontsize=5.0, ncol=2, columnspacing=0.8, handlelength=0.9)
    ax.set_title("(c) Scenarios", loc="left", fontsize=6.2)
    save(fig, "fig3_availability.png")


# ------------------------------------------------------------------ Fig 5: risk screening matrix
def fig_risk():
    from aeolus_rams_phase6 import config as c6
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw={"wspace": 0.55, "width_ratios": [1.1, 1]})
    ax = axs[0]
    sev, fr = c6.SEV_LABELS, c6.FREQ_LABELS
    colmap = {"UNACCEPTABLE": "#F2B8B0", "ALARP": "#FCE7B8", "ACCEPTABLE": "#CDEBDC"}
    for i, sv in enumerate(sev):
        for j, f in enumerate(fr):
            cls = c6.RISK_MATRIX[(sv, f)]
            ax.add_patch(Rectangle((j, 4 - i), 1, 1, fc=colmap[cls], ec="white", lw=1))
            ax.text(j + 0.5, 4 - i + 0.12, {"UNACCEPTABLE": "unacc.", "ALARP": "ALARP", "ACCEPTABLE": "accept."}[cls],
                    fontsize=4.2, color="#555", ha="center", va="center")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(5) + 0.5); ax.set_xticklabels(fr); ax.set_yticks(np.arange(5) + 0.5); ax.set_yticklabels(sev[::-1])
    ax.set_xlabel("frequency class (F1 = most frequent)"); ax.set_ylabel("consequence class (A = extreme)")
    # (name, x, y, colour, marker, label dx, label dy)
    pts = [("B1", 0.5, 0.5, GREEN, "o", 0.14, 0.2), ("B2", 1.5, 1.5, GREEN, "s", 0.14, 0.2), ("B3", 1.5, 2.5, ORANGE, "D", 0.14, 0.2),
           ("B4", 3.5, 4.5, VERMIL, "^", 0.14, 0.2), ("B4*", 4.5, 4.5, BLUE, "^", 0.14, 0.2), ("cable", 0.5, 1.5, PURPLE, "P", 0.14, 0.2)]
    for k, x, y, c, m, dx, dy in pts:
        ax.scatter(x, y, s=26, c=c, marker=m, edgecolor="white", lw=0.5, zorder=5)
        ax.text(x + dx, y + dy - 0.25, k, fontsize=5.5, zorder=6, fontweight="bold", va="center")
    ax.set_title("(a) Screening matrix (B4*: after Option 1)", loc="left", fontsize=6.3)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax = axs[1]
    base = 7.69e-4 * 0.5
    names = ["baseline", "+CCF (beta 0.1)", "Opt. 1: relay $Q_{G3b}$=1e-3", "Opt. 2 accumulator", "Opt. 3 encoder test", "Opt. 1+2+3"]
    vals = [base, base * 1.577175, 1.6073e-4 * 0.5, 2.9772e-4 * 0.5, 6.0352e-4 * 0.5, 4.8835e-5 * 0.5]
    cols = [GREY, VERMIL, GREEN, SKY, SKY, GREEN]
    ys = list(range(6))[::-1]
    ax.barh(ys, vals, color=cols, height=0.62)
    ax.set_xscale("log"); ax.set_xlim(1e-5, 1e-3)
    ax.axvline(1e-4, color="k", ls=":", lw=0.8); ax.text(1.06e-4, 5.55, "F4 | F5", fontsize=5.2, va="center")
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=5.5)
    for v, i in zip(vals, ys):
        ax.text(v * 1.12, i, f"{v:.1e}", va="center", fontsize=5.2)
    ax.set_xlabel("catastrophic freq. (per turbine-year)"); ax.set_title("(b) Risk-reduction options", loc="left", fontsize=6.5)
    save(fig, "fig5_risk.png")


# ------------------------------------------------------------------ Fig 6: maintenance
def fig_maintenance():
    from aeolus_rams_phase7 import config as c7
    from aeolus_rams_phase7.age_replacement import cost_rate, optimal_replacement_age
    from aeolus_rams_phase7.failure_finding import PFDavg_exact
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.5), gridspec_kw={"wspace": 0.38})
    ax = axs[0]
    for name, colr in [("Main/Rotor Bearing", BLUE), ("Gearbox", GREEN), ("Generator", ORANGE)]:
        p, c = c7.LITERATURE_WEIBULL[name], c7.REPLACEMENT_COSTS[name]
        res = optimal_replacement_age(p, c)
        T = np.linspace(2 * 365.25, 80 * 365.25, 160)
        rtf = c.Cf / p.mttf_days
        ax.plot(T / 365.25, np.array([cost_rate(t, p, c) for t in T]) / rtf, color=colr, label=f"{SHORT[name]} ($\\beta$={p.beta})")
        ax.scatter([res.T_star_years], [res.C_star / rtf], color=colr, s=12, zorder=5)
    pp = c7.WeibullParams("Pitch", c7.PITCH_BETA, c7.PITCH_ETA, "fitted_tier_a", "")
    pc = c7.ReplacementCosts("Pitch", 60000.0, 600000.0)
    T = np.linspace(0.2 * 365.25, 80 * 365.25, 200)
    ax.plot(T / 365.25, np.array([cost_rate(t, pp, pc) for t in T]) / (pc.Cf / pp.mttf_days), color=VERMIL, ls="--", label="Pitch ($\\beta$=0.73, illustration)")
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.axvline(25, color=GREY, lw=0.6, ls="-.")
    ax.set_ylim(0.3, 1.8); ax.set_xlim(0, 80)
    ax.set_xlabel("replacement age $T$ (years)"); ax.set_ylabel("$C(T)$ / run-to-failure rate")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.02, 1.0), fontsize=5.2); ax.set_title("(a) Age replacement", loc="left", fontsize=6.5)
    ax = axs[1]
    lam = c7.LAMBDA_G3b
    tau = np.logspace(0, math.log10(1825), 200)
    ax.plot(tau, [PFDavg_exact(lam, t) for t in tau], color=BLUE)
    for lab, th in [("$10^{-2}$", 1e-2), ("$10^{-3}$", 1e-3), ("$10^{-4}$", 1e-4)]:
        ax.axhline(th, color=GREY, lw=0.6, ls=":"); ax.text(1.05, th * 1.15, lab, fontsize=5.2, color=GREY)
    ax.scatter([26.4], [1e-3], color=GREEN, s=14, zorder=5)
    ax.scatter([266], [1e-2], color=GREEN, s=14, zorder=5, marker="s")
    ax.annotate("$10^{-2}$: 266 d", (266, 1e-2), (6, 2.5e-2), fontsize=5.3, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.annotate("PFD=$10^{-3}$: 26 d", (26.4, 1e-3), (60, 2.2e-4), fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.scatter([730], [0.0271], color=VERMIL, s=14, zorder=5)
    ax.annotate("assumed current\n730 d: 0.0271", (730, 0.0271), (50, 0.06), fontsize=5.3, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 0.2)
    ax.set_xlabel("proof-test interval $\\tau$ (days)"); ax.set_ylabel("PFD$_{avg}$ of relay G3b")
    ax.set_title("(b) Failure-finding interval", loc="left", fontsize=6.5)
    save(fig, "fig6_maintenance.png")


# ------------------------------------------------------------------ Fig 7: Bayesian + Cox + CBM
def fig_dynamic():
    from aeolus_rams_phase8 import synthetic_data as sd, bayesian_update as bu
    df = sd.generate_bearing_failures()
    posts, _ = bu.run_sequential_update(df)
    d = pd.read_csv(ROOT / "phase-8-dynamic-modeling/outputs/cbm_threshold_table.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.7), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1, 1]})
    ax = axs[0]
    cols = [GREY, SKY, ORANGE, GREEN]
    for i, p in enumerate(posts):
        eta = np.exp(p["log_eta_grid"]); m = p["marginal_eta"]; m = m / np.trapezoid(m, eta)
        ax.plot(eta / 1000, m * 1000, color=cols[i], ls="--" if i == 0 else "-", lw=0.9, label=f"{[0, 5, 10, 15][i]} failures")
    ax.axvline(23.75, color=VERMIL, lw=0.6, ls="-.")
    ax.set_xlabel("scale $\\eta$ (10$^3$ d)"); ax.set_ylabel("posterior density (10$^{-3}$)")
    ax.legend(frameon=False, loc="upper right", fontsize=4.9, handlelength=1.2, borderaxespad=0.1)
    ax.set_title("(a) Bayesian update", loc="left", fontsize=6.3)
    ax = axs[1]
    cx = pd.read_csv(ROOT / "phase-8-dynamic-modeling/outputs/cox_hazard_ratios.csv")
    names = ["wind", "vibration", "temp."]
    true_hr = [1.15, 1.25, 0.95]
    for i, r in enumerate(cx.itertuples()):
        y = 2 - i
        ax.plot([r.exp_coef_lower_95, r.exp_coef_upper_95], [y, y], color=BLUE, lw=1.2)
        ax.scatter([r.exp_coef], [y], color=BLUE, s=12, zorder=5)
        ax.scatter([true_hr[i]], [y + 0.22], color=VERMIL, marker="v", s=12, zorder=5)
    ax.axvline(1, color="k", lw=0.6); ax.set_xscale("log"); ax.set_xticks([0.7, 1, 1.5, 2.2]); ax.set_xticklabels(["0.7", "1", "1.5", "2.2"])
    ax.minorticks_off(); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(names); ax.set_ylim(-0.5, 2.7)
    ax.set_xlabel("hazard ratio per SD"); ax.set_title("(b) Cox PH (95% CI)", loc="left", fontsize=6.3)
    ax = axs[2]
    g = d[np.isclose(d.cp_cf_ratio, 0.10)].sort_values("k_sigma")
    ax.plot(g.k_sigma, g.cost_rate_per_day, color=BLUE, lw=0.7, alpha=0.7, label="run")
    rep = {2.0: (1305.4, 93.4), 3.0: (502.2, 67.6), 4.0: (335.1, 27.6)}
    ax.errorbar(list(rep), [v[0] for v in rep.values()], yerr=[v[1] for v in rep.values()], fmt="o", color=VERMIL, ms=2.5, capsize=1.5, lw=0.8, label="12 seeds")
    ax.axhline(600000 / 1936.0, color="k", ls=":", lw=0.8); ax.text(0.55, 340, "run-to-failure", fontsize=4.9)
    ax.set_xlim(0.4, 4.2); ax.set_ylim(0, 2500); ax.set_xlabel("threshold $k$"); ax.set_ylabel("cost rate (USD/day)")
    ax.legend(frameon=False, loc="upper right", fontsize=4.9, handlelength=1.2, borderaxespad=0.1)
    ax.set_title("(c) CBM threshold", loc="left", fontsize=6.3)
    save(fig, "fig7_dynamic.png")


if __name__ == "__main__":
    for fn in (fig_architecture, fig_safety_chain, fig_evidence_avail, fig_risk, fig_maintenance, fig_dynamic):
        fn()
        print("ok", fn.__name__)
