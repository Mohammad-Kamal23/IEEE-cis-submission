"""Figure 2 -- selective prediction. Every number here is measured from the
per-sample predictions the pipeline wrote in Results/pipeline/probs/."""
import os as _os
_RES = _os.environ.get("APEX_RESULTS", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results"))
import csv, json, math, numpy as np, matplotlib.pyplot as plt
from figstyle import *

D = np.load(_os.path.join(_RES, "curve_data.npz"))
DOMS = ["BLOODCELL","BRAINTUMOR","CHEST-XRAY","ENDOSCOPIC","RETINAL","ULTRASOUND"]
NICE = {"BLOODCELL":"Blood cell","BRAINTUMOR":"Brain MRI","CHEST-XRAY":"Chest X-ray",
        "ENDOSCOPIC":"Endoscopy","RETINAL":"Retinal OCT","ULTRASOUND":"Breast US"}
ARCHS = ["ViT","ConvNeXt","MobileNetV3"]
CFG = [(d,a) for d in DOMS for a in ARCHS]
METH = ["BASE","TEMP_SCALE","VEC_SCALE","ISOTONIC","DIRICHLET","APEX"]
GRIDC = np.linspace(0.05, 1.0, 191)

def rc_curve(conf, corr):
    """risk (error rate) as a function of coverage, on a fixed grid."""
    o = np.argsort(-conf, kind="stable")
    err = 1.0 - corr[o].astype(np.float64)
    cum = np.cumsum(err) / np.arange(1, len(err)+1)
    cov = np.arange(1, len(err)+1) / len(err)
    return np.interp(GRIDC, cov, cum)

def aurc(conf, corr):
    o = np.argsort(-conf, kind="stable")
    err = 1.0 - corr[o].astype(np.float64)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err)+1)))

# Panel (a) is a per-sample object and is measured from the predictions.
# Panels (b) and (c) are AURC numbers, so they read the same per-fold CSV the
# tables read -- one source for every number in the paper.
CSV = _os.path.join(_RES, "FINAL_RESULTS_perfold.csv")
_rows = list(csv.DictReader(open(CSV)))
def csv_aurc(m, d, a):
    v = [float(r["AURC"]) for r in _rows
         if r["method"] == m and r["domain"] == d and r["arch"] == a]
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v))

curves, aurcs = {}, {}
for m in METH:
    cs, av = [], {}
    for d,a in CFG:
        k = f"{d}|{a}|{m}"
        if k+"|conf" not in D: continue
        cs.append(rc_curve(D[k+"|conf"], D[k+"|corr"]))
        av[(d,a)] = csv_aurc(m, d, a)
    curves[m] = np.vstack(cs); aurcs[m] = av

fig = plt.figure(figsize=(WIDE, 1.74))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.86, 1.02], wspace=0.40)

# ---- (a) mean risk-coverage over the 18 configurations ---------------------
ax = fig.add_subplot(gs[0]); grid(ax)
for m in METH:
    s = STYLE[m]; mu = curves[m].mean(0)
    ax.plot(GRIDC, 100*mu, color=s["color"], ls=s["ls"],
            lw=1.7 if m=="APEX" else 0.95, zorder=5 if m=="APEX" else 3)
lo, hi = curves["APEX"].mean(0)-curves["APEX"].std(0)/np.sqrt(18), \
         curves["APEX"].mean(0)+curves["APEX"].std(0)/np.sqrt(18)
ax.fill_between(GRIDC, 100*lo, 100*hi, color=APEXC, alpha=0.16, lw=0, zorder=4)
ax.set_xlabel("coverage"); ax.set_ylabel("selective risk (%)")
ax.set_xlim(0.05, 1.0); ax.set_ylim(0, None)
ax.set_ylim(0, 15.4)
ax.text(0.965, 100*np.interp(0.95, GRIDC, curves["APEX"].mean(0)) - 1.05,
        "APEX", color=APEXC, fontsize=7.5, fontweight="bold", ha="right", va="top")
ax.legend([plt.Line2D([],[],**{k:v for k,v in STYLE[m].items() if k in("color","ls")},
           lw=1.5 if m=="APEX" else 0.95) for m in METH],
          [STYLE[m]["label"] for m in METH], loc="upper left", ncol=1,
          bbox_to_anchor=(-0.020, 1.045), handletextpad=0.55)
tag(ax, "(a)", dx=-0.235)

# ---- (b) per-configuration AURC, APEX vs the strongest baseline ------------
ax = fig.add_subplot(gs[1]); grid(ax)
best = {c: min((aurcs[m][c], m) for m in METH if m!="APEX") for c in CFG}
bx = np.array([best[c][0] for c in CFG]); ay = np.array([aurcs["APEX"][c] for c in CFG])
lim = (0.6*min(bx.min(), ay.min()), 1.35*max(bx.max(), ay.max()))
ax.plot(lim, lim, color=MUTED, lw=0.6, ls=(0,(3,2)), zorder=2)
for i,(d,a) in enumerate(CFG):
    ax.scatter(bx[i], ay[i], s=17, facecolor=APEXC, edgecolor="white",
               linewidth=0.5, zorder=5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("AURC, best baseline"); ax.set_ylabel("AURC, APEX")
ax.text(0.045, 0.955, f"APEX better in\n{int((ay<bx).sum())} of 18",
        transform=ax.transAxes, fontsize=6.8, va="top", ha="left", color=APEXC,
        fontweight="bold", linespacing=1.35)
ax.text(0.965, 0.115, "baseline better", transform=ax.transAxes, fontsize=6.1,
        ha="right", va="center", color=MUTED, rotation=39)
tag(ax, "(b)", dx=-0.30)

# ---- (c) AURC per dataset --------------------------------------------------
ax = fig.add_subplot(gs[2]); grid(ax, axis="x")
yy = np.arange(len(DOMS))[::-1]
bb = [np.mean([aurcs["BASE"][(d,a)] for a in ARCHS]) for d in DOMS]
aa = [np.mean([aurcs["APEX"][(d,a)] for a in ARCHS]) for d in DOMS]
ax.barh(yy+0.19, bb, height=0.34, color="#c9c9c6", lw=0, zorder=3)
ax.barh(yy-0.19, aa, height=0.34, color=APEXC, lw=0, zorder=3)
ax.set_xlim(0, 1.34*max(bb))
for i,d in enumerate(DOMS):
    ax.text(aa[i]+0.010*max(bb), yy[i]-0.19, "\u2212%.0f%%" % (100*(bb[i]-aa[i])/bb[i]),
            va="center", fontsize=6.4, color=APEXC, fontweight="bold")
ax.set_yticks(yy); ax.set_yticklabels([NICE[d] for d in DOMS])
ax.set_xlabel("AURC")
ax.tick_params(axis="y", length=0)
ax.text(0.985, 0.985, "uncalibrated", transform=ax.transAxes, ha="right",
        va="top", fontsize=6.4, color="#8a8a86")
ax.text(0.985, 0.895, "APEX", transform=ax.transAxes, ha="right", va="top",
        fontsize=6.4, color=APEXC, fontweight="bold")
tag(ax, "(c)", dx=-0.34)

fig.savefig("plots/fig2_selective.pdf"); fig.savefig("plots/fig2_selective.png")
print("fig2 written")
print("  panel b: APEX beats the per-config best baseline in", int((ay<bx).sum()), "/18")
print("  mean AURC  base %.4f  apex %.4f  (%.1f%%)" % (np.mean(bb), np.mean(aa),
      100*(np.mean(aa)-np.mean(bb))/np.mean(bb)))
json.dump({"aurc_measured":{m:{f"{d}|{a}":aurcs[m][(d,a)] for d,a in CFG} for m in METH},
           "wins_vs_best_baseline": int((ay<bx).sum())}, open("rc.json","w"), indent=1)
