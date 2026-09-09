import os as _os
_RES = _os.environ.get("APEX_RESULTS", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results"))
import csv, json, itertools, math
import numpy as np
from scipy.stats import wilcoxon

SRC = _os.path.join(_RES, "FINAL_RESULTS_perfold.csv")
rows = list(csv.DictReader(open(SRC)))
METRICS = ["BAcc","F1","AUC","ECE","dECE","SCE","Brier","BSS","NLL","Conf","Ent",
           "MTP","MCE","OCE","CCC","AURC","EAURC","Risk@90","Risk@80"]
def f(r,m):
    try: return float(r[m])
    except: return float("nan")

DOMS = ["BLOODCELL","BRAINTUMOR","CHEST-XRAY","ENDOSCOPIC","RETINAL","ULTRASOUND"]
ARCHS = ["ViT","ConvNeXt","MobileNetV3"]
CFG = [(d,a) for d in DOMS for a in ARCHS]

# config-level mean over the 5 folds
def cfg_mean(method, metric):
    out = {}
    for d,a in CFG:
        v = [f(r,metric) for r in rows if r["method"]==method and r["domain"]==d and r["arch"]==a]
        v = [x for x in v if not math.isnan(x)]
        out[(d,a)] = float(np.mean(v)) if v else float("nan")
    return out

METHODS = sorted({r["method"] for r in rows})
grand = {m:{k: float(np.nanmean(list(cfg_mean(m,k).values()))) for k in METRICS} for m in METHODS}

MAIN = ["BASE","MC_DROPOUT","TEMP_SCALE","VEC_SCALE","ISOTONIC","DIRICHLET","APEX"]
print("HEADLINE (mean over 18 configurations)")
hdr = ["BAcc","F1","AUC","ECE","dECE","SCE","MCE","Brier","BSS","NLL","AURC","EAURC","Risk@90","Risk@80","CCC"]
print(f"{'method':<12}" + "".join(f"{h:>9}" for h in hdr))
for m in MAIN:
    print(f"{m:<12}" + "".join(f"{grand[m][h]:>9.4f}" for h in hdr))

print()
print("WILCOXON  APEX vs each baseline, over 18 configurations")
res = {}
for b in MAIN[:-1]:
    res[b] = {}
    for k in ["BAcc","ECE","SCE","Brier","NLL","AURC","EAURC","Risk@90"]:
        A = cfg_mean("APEX",k); B = cfg_mean(b,k)
        pairs = [(A[c],B[c]) for c in CFG if not (math.isnan(A[c]) or math.isnan(B[c]))]
        a = np.array([p[0] for p in pairs]); bb = np.array([p[1] for p in pairs])
        lower_better = k not in ("BAcc","F1","AUC","BSS","CCC")
        wins = int(np.sum(a<bb)) if lower_better else int(np.sum(a>bb))
        try: p = float(wilcoxon(a,bb).pvalue)
        except Exception: p = 1.0
        res[b][k] = {"wins":wins,"n":len(pairs),"p":p,
                     "apex":float(a.mean()),"base":float(bb.mean())}
    r = res[b]
    print(f"  vs {b:<12} " + "  ".join(
        f"{k} {r[k]['wins']}/{r[k]['n']} p={r[k]['p']:.2e}" for k in ["BAcc","ECE","AURC","Brier","NLL"]))

json.dump({"grand":grand,"wilcoxon":res,
           "cfg":{m:{f"{d}|{a}":{k:cfg_mean(m,k)[(d,a)] for k in METRICS} for d,a in CFG} for m in METHODS}},
          open("stats.json","w"), indent=1)
print("\nwrote stats.json")
