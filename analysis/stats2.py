import json, math, numpy as np, csv
from scipy.stats import wilcoxon
S = json.load(open("stats.json"))
cfg = S["cfg"]; grand = S["grand"]
DOMS = ["BLOODCELL","BRAINTUMOR","CHEST-XRAY","ENDOSCOPIC","RETINAL","ULTRASOUND"]
ARCHS = ["ViT","ConvNeXt","MobileNetV3"]
KEYS = [f"{d}|{a}" for d in DOMS for a in ARCHS]

def paired(m1,m2,metric):
    a=np.array([cfg[m1][k][metric] for k in KEYS]); b=np.array([cfg[m2][k][metric] for k in KEYS])
    ok=~(np.isnan(a)|np.isnan(b)); a,b=a[ok],b[ok]
    lower = metric not in ("BAcc","F1","AUC","BSS","CCC")
    wins=int(np.sum(a<b)) if lower else int(np.sum(a>b))
    try: p=float(wilcoxon(a,b).pvalue)
    except Exception: p=1.0
    return a.mean(), b.mean(), wins, len(a), p

print("ABLATIONS  (variant vs APEX; positive delta = variant is worse for lower-better)")
ABL=[("ABL_focal","focal loss instead of NLL"),("ABL_no_descriptors","no handcrafted descriptors"),
     ("ABL_no_gate","no confidence gate"),("ABL_no_knn","no k-NN expert"),
     ("ABL_no_poly","no cubic-polynomial SVM"),("ABL_no_rbf","no RBF SVM")]
abl={}
for m,desc in ABL:
    abl[m]={}
    line=f"  {m:<20}"
    for k in ["BAcc","ECE","Brier","NLL","AURC"]:
        av,bv,w,n,p = paired(m,"APEX",k)
        abl[m][k]={"variant":av,"apex":bv,"delta":av-bv,"p":p,"n":n}
        line+=f"  {k} {av:.4f} ({av-bv:+.4f}, p={p:.1e})"
    print(line)

print()
print("FAIR  (single expert / equal blend, vs APEX)")
FAIR=["FAIR_equal_blend","FAIR_knn","FAIR_poly_svm","FAIR_rbf_svm"]
fair={}
for m in FAIR:
    fair[m]={}
    line=f"  {m:<20}"
    for k in ["BAcc","ECE","AURC"]:
        av,bv,w,n,p=paired(m,"APEX",k); fair[m][k]={"variant":av,"apex":bv,"delta":av-bv,"p":p}
        line+=f"  {k} {av:.4f} ({av-bv:+.4f}, p={p:.1e})"
    print(line)

print()
print("TAU SWEEP")
tau={}
for t in ["0.60","0.70","0.80","0.85","0.90","0.95"]:
    m=f"TAU_{t}"
    tau[t]={k:grand[m][k] for k in ["BAcc","ECE","AURC"]}
    print(f"  tau={t}  BAcc {grand[m]['BAcc']:.4f}  ECE {grand[m]['ECE']:.4f}  AURC {grand[m]['AURC']:.4f}")
sp=[tau[t]["AURC"] for t in tau]; print(f"  AURC range over tau: {max(sp)-min(sp):.5f}")

print()
print("PER-DATASET  AURC and BAcc (mean over 3 backbones)")
perd={}
for d in DOMS:
    ks=[f"{d}|{a}" for a in ARCHS]
    row={}
    for m in ["BASE","TEMP_SCALE","VEC_SCALE","DIRICHLET","APEX"]:
        row[m]={k: float(np.nanmean([cfg[m][kk][k] for kk in ks])) for k in ["BAcc","ECE","AURC","Brier","NLL","Risk@90"]}
    perd[d]=row
    b=row["BASE"]; x=row["APEX"]
    print(f"  {d:<12} BAcc {b['BAcc']:.4f}->{x['BAcc']:.4f}  AURC {b['AURC']:.4f}->{x['AURC']:.4f} "
          f"({100*(x['AURC']-b['AURC'])/b['AURC']:+.1f}%)  ECE {b['ECE']:.4f}->{x['ECE']:.4f}")

print()
print("PER-BACKBONE")
perа={}
for a in ARCHS:
    ks=[f"{d}|{a}" for d in DOMS]
    b={k: float(np.nanmean([cfg['BASE'][kk][k] for kk in ks])) for k in ["BAcc","ECE","AURC"]}
    x={k: float(np.nanmean([cfg['APEX'][kk][k] for kk in ks])) for k in ["BAcc","ECE","AURC"]}
    perа[a]={"BASE":b,"APEX":x}
    print(f"  {a:<13} BAcc {b['BAcc']:.4f}->{x['AURC'] and x['BAcc']:.4f}   AURC {b['AURC']:.4f}->{x['AURC']:.4f}")

json.dump({"abl":abl,"fair":fair,"tau":tau,"perdataset":perd,"perarch":perа},
          open("stats2.json","w"), indent=1)
print("\nwrote stats2.json")
