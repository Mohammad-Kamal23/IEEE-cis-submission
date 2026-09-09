"""Figure 3 -- calibration and ranking are different properties.
(a) measured reliability, pooled over all 18 configurations.
(b) mean rank on ECE against mean rank on AURC, over the same 18."""
import os as _os
_RES = _os.environ.get("APEX_RESULTS", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results"))
import json, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from figstyle import *

D = np.load(_os.path.join(_RES, "curve_data.npz"))
S = json.load(open("stats.json")); cfg = S["cfg"]
DOMS=["BLOODCELL","BRAINTUMOR","CHEST-XRAY","ENDOSCOPIC","RETINAL","ULTRASOUND"]
ARCHS=["ViT","ConvNeXt","MobileNetV3"]; CFG=[f"{d}|{a}" for d in DOMS for a in ARCHS]
METH=["BASE","MC_DROPOUT","TEMP_SCALE","VEC_SCALE","ISOTONIC","DIRICHLET","APEX"]

NB = 14

def reliability(m):
    """Bin INSIDE each configuration, then average bin-by-bin across the 18.

    Pooling all 90 folds into one histogram instead lets a method's
    over-confidence on one dataset cancel its under-confidence on another, which
    flatters whichever method is inconsistent rather than accurate. Binning
    per configuration is also how the ECE column of Table I is computed, so the
    figure and the table cannot disagree.
    """
    C = np.full((len(CFG), NB), np.nan); A = np.full((len(CFG), NB), np.nan)
    for j, k in enumerate(CFG):
        if f"{k}|{m}|conf" not in D:
            continue
        c = D[f"{k}|{m}|conf"]; r = D[f"{k}|{m}|corr"].astype(float)
        q = np.quantile(c, np.linspace(0, 1, NB + 1)); q[0] -= 1e-9
        for i in range(NB):
            sel = (c > q[i]) & (c <= q[i + 1])
            if sel.sum() < 15:
                continue
            C[j, i] = c[sel].mean(); A[j, i] = r[sel].mean()
    n = np.sum(~np.isnan(A), 0)
    return (np.nanmean(C, 0), np.nanmean(A, 0),
            np.nanstd(A, 0) / np.sqrt(np.maximum(n, 1)),
            float(np.nanmean(np.nanmean(np.abs(A - C), 1))))

fig = plt.figure(figsize=(WIDE, 1.78))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.285)

# ---- (a) reliability, binned within each configuration ---------------------
ax = fig.add_subplot(gs[0]); grid(ax)
ax.plot([0.35, 1], [0.35, 1], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
rel, gaps = {}, {}
for m in ["BASE", "DIRICHLET", "APEX"]:
    xs, ys, se, gap = reliability(m)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys, se = xs[ok], ys[ok], se[ok]
    st = STYLE[m]
    if m == "APEX":
        ax.fill_between(xs, ys - se, ys + se, color=APEXC, alpha=0.18, lw=0, zorder=3)
    ax.plot(xs, ys, color=st["color"], ls="-" if m == "APEX" else st["ls"],
            lw=1.6 if m == "APEX" else 1.0, marker=st["marker"] or "o",
            ms=2.9 if m == "APEX" else 2.5,
            mfc=st["color"] if m == "APEX" else "white", mew=0.7,
            zorder=6 if m == "APEX" else 4)
    rel[m] = (list(map(float, xs)), list(map(float, ys))); gaps[m] = gap
ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("observed accuracy")
ax.set_xlim(0.35, 1.005); ax.set_ylim(0.35, 1.02)
ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.text(0.030, 0.982,
        "dashed line: perfect calibration\nabove it: under-confident   below it: over-confident",
        transform=ax.transAxes, fontsize=6.0, color=MUTED, ha="left", va="top",
        linespacing=1.35)
ax.legend([plt.Line2D([], [], color=STYLE[m]["color"],
           ls="-" if m == "APEX" else STYLE[m]["ls"],
           lw=1.5 if m == "APEX" else 1.0, marker=STYLE[m]["marker"] or "o", ms=2.9,
           mfc=STYLE[m]["color"] if m == "APEX" else "white", mew=0.7)
           for m in ["BASE", "DIRICHLET", "APEX"]],
          ["Uncalibrated", "Dirichlet", "APEX"], loc="lower right",
          bbox_to_anchor=(1.03, -0.03), handletextpad=0.5)
tag(ax, "(a)", dx=-0.225)

# ---- (b) mean rank: ECE against AURC ---------------------------------------
ax = fig.add_subplot(gs[1])
rank = {}
for k in ("ECE","AURC"):
    R = np.vstack([rankdata([cfg[m][c][k] for m in METH]) for c in CFG])
    rank[k] = {m: float(R[:,i].mean()) for i,m in enumerate(METH)}
disagree = sum(1 for c in CFG
               if int(np.argmin([cfg[m][c]["ECE"] for m in METH]))
               != int(np.argmin([cfg[m][c]["AURC"] for m in METH])))

x0, x1 = 0.0, 1.0
LGAP = 0.44
lab_l = dict(zip(METH, declutter([rank["ECE"][m] for m in METH], LGAP)))
lab_r = dict(zip(METH, declutter([rank["AURC"][m] for m in METH], LGAP)))
for m in METH:
    st = STYLE[m]; hi = (m == "APEX")
    ax.plot([x0,x1], [rank["ECE"][m], rank["AURC"][m]], color=st["color"],
            lw=2.0 if hi else 1.0, alpha=1.0 if hi else 0.75, zorder=6 if hi else 3,
            solid_capstyle="round")
    for xx, kk, ha in ((x0,"ECE","right"), (x1,"AURC","left")):
        ax.scatter([xx],[rank[kk][m]], s=26 if hi else 17, facecolor=st["color"],
                   edgecolor="white", linewidth=0.7, zorder=7 if hi else 4)
    for xx, dxx, pos, txt, ha in (
            (x0, -0.055, lab_l[m], f"{st['label']}  {rank['ECE'][m]:.1f}", "right"),
            (x1, +0.055, lab_r[m], f"{rank['AURC'][m]:.1f}  {st['label']}", "left")):
        yy = rank["ECE"][m] if xx == x0 else rank["AURC"][m]
        if abs(pos - yy) > 1e-6:
            ax.plot([xx + dxx*0.72, xx + dxx*0.30], [pos, yy], color=st["color"],
                    lw=0.45, alpha=0.75, zorder=5)
        ax.text(xx + dxx, pos, txt, ha=ha, va="center", fontsize=6.2,
                color=st["color"], fontweight="bold" if hi else "normal")
ax.set_xlim(-0.80, 1.80); ax.set_ylim(7.75, 0.10)
ax.set_yticks([]); ax.set_xticks([x0,x1])
ax.set_xticklabels(["mean rank on ECE", "mean rank on AURC"], fontsize=7.3)
for s in ("left","right","top","bottom"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.axvline(x0, color=GRID, lw=0.6, zorder=1); ax.axvline(x1, color=GRID, lw=0.6, zorder=1)
ax.text(0.315, 0.985, f"in {disagree} of the 18 configurations the\n"
                      f"lowest-ECE method is not the\nlowest-AURC method",
        transform=ax.transAxes, fontsize=6.6, ha="center", va="top", color=INK,
        linespacing=1.42)
tag(ax, "(b)", dx=-0.055, dy=1.005)

fig.savefig("plots/fig3_calibration.pdf"); fig.savefig("plots/fig3_calibration.png")
rho = spearmanr([rank["ECE"][m] for m in METH],[rank["AURC"][m] for m in METH]).correlation
print("fig3 written")
print("  mean ranks ECE :", {m: round(rank['ECE'][m],2) for m in METH})
print("  mean ranks AURC:", {m: round(rank['AURC'][m],2) for m in METH})
print(f"  lowest-ECE != lowest-AURC in {disagree}/18")
print("  mean |accuracy - confidence| per configuration:",
      {k: round(v, 4) for k, v in gaps.items()})
json.dump({"mean_rank": rank, "disagree": disagree, "binned_gap": gaps,
           "reliability": {k: {"conf": v[0], "acc": v[1]} for k, v in rel.items()}},
          open("rel.json", "w"), indent=1)
