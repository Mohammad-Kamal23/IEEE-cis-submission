"""Figure 4 -- what each part contributes, and how little the gate threshold matters."""
import json, numpy as np, matplotlib.pyplot as plt
from figstyle import *

A = json.load(open("stats2.json"))
abl, tau = A["abl"], A["tau"]

ROWS = [("ABL_no_rbf",          "RBF expert"),
        ("ABL_no_knn",          "$k$-NN expert"),
        ("ABL_no_poly",         "cubic SVM"),
        ("ABL_no_descriptors",  "shape descriptors"),
        ("ABL_no_gate",         "confidence gate"),
        ("ABL_focal",           "NLL $\\rightarrow$ focal loss")]

fig = plt.figure(figsize=(COL, 2.22))
gs = fig.add_gridspec(2, 1, height_ratios=[1.58, 1.0], hspace=0.68)

# ---- (a) ablation: change in AURC and in ECE when a part is removed --------
ax = fig.add_subplot(gs[0]); grid(ax, axis="x")
y = np.arange(len(ROWS))[::-1]
dA = np.array([abl[k]["AURC"]["delta"] for k,_ in ROWS]) * 1000   # x10^-3
dE = np.array([abl[k]["ECE"]["delta"]  for k,_ in ROWS]) * 1000
pA = [abl[k]["AURC"]["p"] for k,_ in ROWS]
pE = [abl[k]["ECE"]["p"]  for k,_ in ROWS]
ax.barh(y+0.20, dA, height=0.36, color=APEXC, lw=0, zorder=3)
ax.barh(y-0.20, dE, height=0.36, color="#9ec9e2", lw=0, zorder=3)
ax.axvline(0, color="#8a8a86", lw=0.6, zorder=4)
def star(p): return "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
for i in range(len(ROWS)):
    for yy, d, p in ((y[i]+0.20, dA[i], pA[i]), (y[i]-0.20, dE[i], pE[i])):
        off = 0.55 if d >= 0 else -0.55
        ax.text(d + off, yy, star(p), va="center", ha="left" if d >= 0 else "right",
                fontsize=5.9, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels([t for _,t in ROWS])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("change vs. full APEX, worse to the right  ($\\times 10^{-3}$)")
ax.set_xlim(-6.3, 22.0)
ax.legend([plt.Rectangle((0,0),1,1,fc=APEXC,ec="none"),
           plt.Rectangle((0,0),1,1,fc="#9ec9e2",ec="none")],
          ["AURC","ECE"], loc="lower right", ncol=2,
          bbox_to_anchor=(1.015, 1.005), handlelength=1.0, handleheight=0.85,
          handletextpad=0.42, columnspacing=1.0)
tag(ax, "(a)", dx=-0.365, dy=1.03)

# ---- (b) threshold sweep ---------------------------------------------------
ax = fig.add_subplot(gs[1]); grid(ax)
ts = sorted(tau); xs = [float(t) for t in ts]
ax.plot(xs, [1000*tau[t]["AURC"] for t in ts], color=APEXC, marker="o", ms=3.1,
        mfc=APEXC, mec="white", mew=0.6, lw=1.3, zorder=5)
ax.set_xlabel("confidence gate threshold  $\\tau$")
ax.set_ylabel("AURC ($\\times 10^{-3}$)")
lo = min(1000*tau[t]["AURC"] for t in ts); hi = max(1000*tau[t]["AURC"] for t in ts)
ax.set_ylim(lo-1.35, hi+1.35)
ax.text(0.5, 0.965, f"total spread {hi-lo:.2f}$\\times 10^{{-3}}$"
                    f"  ({100*(hi-lo)/np.mean([1000*tau[t]['AURC'] for t in ts]):.1f}% of the mean)",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.3, color=MUTED)
tag(ax, "(b)", dx=-0.245, dy=1.03)

fig.savefig("plots/fig4_ablation.pdf"); fig.savefig("plots/fig4_ablation.png")
print("fig4 written")
for k,t in ROWS:
    print(f"  {t:<28} dAURC {abl[k]['AURC']['delta']*1000:+7.2f}e-3 p={abl[k]['AURC']['p']:.1e}"
          f"   dECE {abl[k]['ECE']['delta']*1000:+7.2f}e-3 p={abl[k]['ECE']['p']:.1e}")
print(f"  tau spread in AURC: {hi-lo:.5f} (x10^-3)")
