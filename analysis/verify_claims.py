"""Re-derive every number quoted in main.tex straight from the per-fold CSV and
fail loudly on any mismatch. Nothing in the paper is allowed to be hand-typed."""
import os as _os
_RES = _os.environ.get("APEX_RESULTS", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results"))
import csv, json, math, re, sys
import numpy as np
from scipy.stats import wilcoxon, rankdata

rows=list(csv.DictReader(open(_os.path.join(_RES, "FINAL_RESULTS_perfold.csv"))))
DOMS=["BLOODCELL","BRAINTUMOR","CHEST-XRAY","ENDOSCOPIC","RETINAL","ULTRASOUND"]
ARCHS=["ViT","ConvNeXt","MobileNetV3"]; CFG=[(d,a) for d in DOMS for a in ARCHS]
def cm(m,k):
    o={}
    for d,a in CFG:
        v=[float(r[k]) for r in rows if r["method"]==m and r["domain"]==d and r["arch"]==a]
        v=[x for x in v if not math.isnan(x)]
        o[(d,a)]=float(np.mean(v))
    return o
def g(m,k): return float(np.mean(list(cm(m,k).values())))
def col(m,k): return np.array([cm(m,k)[c] for c in CFG])

BASE=["BASE","MC_DROPOUT","TEMP_SCALE","VEC_SCALE","ISOTONIC","DIRICHLET"]
ok=True
def check(label, got, want, tol=5e-4):
    global ok
    good = abs(got-want) <= tol
    ok &= good
    print(f"  [{'OK ' if good else 'BAD'}] {label:<52} paper {want:<10} computed {got:.5f}")

print("headline")
check("AURC uncalibrated x1e-2", g("BASE","AURC")*100, 4.86, 5e-3)
check("AURC APEX x1e-2",        g("APEX","AURC")*100, 2.71, 5e-3)
check("AURC reduction vs BASE %", 100*(g("BASE","AURC")-g("APEX","AURC"))/g("BASE","AURC"), 44.3, 0.05)
best=min((g(b,"AURC"),b) for b in BASE)
check("AURC reduction vs best baseline %", 100*(best[0]-g("APEX","AURC"))/best[0], 39.0, 0.05)
check("Risk@90 uncalibrated %",  g("BASE","Risk@90")*100, 10.71, 5e-3)
check("Risk@90 APEX %",          g("APEX","Risk@90")*100, 6.52, 5e-3)
check("Brier reduction %",  100*(g("BASE","Brier")-g("APEX","Brier"))/g("BASE","Brier"), 30.6, 0.05)
check("NLL reduction %",    100*(g("BASE","NLL")-g("APEX","NLL"))/g("BASE","NLL"), 30.2, 0.05)
check("BAcc gain (points)", 100*(g("APEX","BAcc")-g("BASE","BAcc")), 4.18, 0.01)
check("ECE APEX x1e-2",     g("APEX","ECE")*100, 3.39, 5e-3)
check("ECE uncalibrated x1e-2", g("BASE","ECE")*100, 4.95, 5e-3)

print("wilcoxon")
for k,lo in (("AURC",1),("Brier",1),("NLL",1)):
    w=[int(np.sum(col("APEX",k)<col(b,k))) for b in BASE]
    p=[float(wilcoxon(col("APEX",k),col(b,k)).pvalue) for b in BASE]
    print(f"  [{'OK ' if min(w)==18 and max(p)<1e-4 else 'BAD'}] {k} 18/18 vs all six, worst p={max(p):.2e}")
    ok &= (min(w)==18 and max(p)<1e-4)
for b,want_w,want_p in (("TEMP_SCALE",7,0.37),("VEC_SCALE",6,0.13),("ISOTONIC",9,0.87),("DIRICHLET",6,0.030)):
    w=int(np.sum(col("APEX","ECE")<col(b,"ECE"))); p=float(wilcoxon(col("APEX","ECE"),col(b,"ECE")).pvalue)
    good = w==want_w and abs(p-want_p)<0.006
    ok &= good
    print(f"  [{'OK ' if good else 'BAD'}] ECE vs {b:<11} paper {want_w}/18 p={want_p}   computed {w}/18 p={p:.3f}")

print("ranks and disagreement")
rk={k:{m:float(np.mean([rankdata([cm(mm,k)[c] for mm in BASE+['APEX']])[i]
      for c in CFG])) for i,m in enumerate(BASE+['APEX'])} for k in ("ECE","AURC","Brier","NLL")}
for k in ("AURC","Brier","NLL"):
    good = abs(rk[k]["APEX"]-1.0)<1e-9; ok&=good
    print(f"  [{'OK ' if good else 'BAD'}] APEX mean rank on {k:<6} = {rk[k]['APEX']:.2f} (paper 1.00)")
good = abs(rk["ECE"]["APEX"]-3.78)<0.005; ok&=good
print(f"  [{'OK ' if good else 'BAD'}] APEX mean rank on ECE    = {rk['ECE']['APEX']:.2f} (paper 3.78)")
dis=sum(1 for c in CFG if int(np.argmin([cm(m,"ECE")[c] for m in BASE+['APEX']]))
                        != int(np.argmin([cm(m,"AURC")[c] for m in BASE+['APEX']])))
good = dis==15; ok&=good
print(f"  [{'OK ' if good else 'BAD'}] lowest-ECE != lowest-AURC in {dis}/18 (paper 15)")

print("ablations (deltas vs APEX)")
for m,k,want,wp in (("ABL_no_rbf","AURC",2.9,1e-3),("ABL_no_rbf","ECE",9.0,1e-3),
                    ("ABL_no_knn","AURC",1.5,2.5e-4),("ABL_focal","ECE",18.0,1e-5),
                    ("ABL_no_gate","ECE",1.8,1e-3)):
    d=(g(m,k)-g("APEX",k))*1000; p=float(wilcoxon(col(m,k),col("APEX",k)).pvalue)
    good = abs(d-want)<0.55 and p<wp*1.2; ok&=good
    print(f"  [{'OK ' if good else 'BAD'}] {m:<18} d{k:<5} paper {want:+5.1f}e-3  computed {d:+.2f}e-3 p={p:.1e}")

print("single experts")
check("FAIR_rbf_svm AURC x1e-2", g("FAIR_rbf_svm","AURC")*100, 3.12, 5e-3)
check("FAIR_equal_blend AURC x1e-2", g("FAIR_equal_blend","AURC")*100, 2.85, 5e-3)

print("tau sweep")
tv=[g(f"TAU_{t}","AURC") for t in ("0.60","0.70","0.80","0.85","0.90","0.95")]
sp=(max(tv)-min(tv))*1000
good = abs(sp-0.13)<0.02 and abs(100*sp/(1000*np.mean(tv))-0.5)<0.06; ok&=good
print(f"  [{'OK ' if good else 'BAD'}] tau spread {sp:.3f}e-3 = {100*sp/(1000*np.mean(tv)):.2f}% of mean (paper 1.3e-4, 0.5%)")

print("per-dataset AURC reduction")
for d,want in (("ENDOSCOPIC",18),("BLOODCELL",82)):
    b=np.mean([cm("BASE","AURC")[(d,a)] for a in ARCHS]); x=np.mean([cm("APEX","AURC")[(d,a)] for a in ARCHS])
    r=100*(b-x)/b; good=abs(r-want)<0.6; ok&=good
    print(f"  [{'OK ' if good else 'BAD'}] {d:<12} {r:.1f}% (paper {want}%)")

print("counts")
n_cells=len({(r['domain'],r['arch'],r['fold']) for r in rows}); good=n_cells==90; ok&=good
print(f"  [{'OK ' if good else 'BAD'}] 90 folds: {n_cells}")
good=len(CFG)==18; ok&=good
print(f"  [{'OK ' if good else 'BAD'}] 18 configurations")
mc=sum(1 for r in rows if r['method']=='APEX' and r['mc_active'] in ('True','1','true'))
print(f"  [   ] mc_active rows for APEX: {mc}")
print()
print("ROUND ONE:", "ALL CLAIMS VERIFIED" if ok else "*** MISMATCHES ABOVE ***")

# ---------------------------------------------------------------- audit round 2
print()
print("claims added or corrected in the second audit")
ok2 = True
def chk(label, cond, detail=""):
    global ok2
    ok2 &= bool(cond)
    print(f"  [{'OK ' if cond else 'BAD'}] {label:<52} {detail}")

# the quarantine ledger must add up
chk("798 masks + 531 copies + 8 cross-class = 1,337", 798+531+8 == 1337, f"{798+531+8}")
chk("33,588 - 1,337 = 32,251 images used", 33588-1337 == 32251, f"{33588-1337}")

# which metrics really are 18/18 against ALL six baselines
for k, must in (("AURC",True),("Brier",True),("NLL",True),
                ("BAcc",False),("Risk@90",False)):
    a=col("APEX",k); w=[int(np.sum(a>col(b,k)) if k=="BAcc" else np.sum(a<col(b,k))) for b in BASE]
    chk(f"{k} is 18/18 vs all six" if must else f"{k} is NOT 18/18 (paper says >=17)",
        (min(w)==18)==must and min(w)>=17, f"min {min(w)}/18")

# APEX trails three calibrators on ECE rank, not two
R=np.vstack([rankdata([cm(m,"ECE")[c] for m in BASE+['APEX']]) for c in CFG])
mr={m: float(R[:,i].mean()) for i,m in enumerate(BASE+['APEX'])}
behind=[m for m in BASE if mr[m] < mr["APEX"]]
chk("ECE rank: APEX behind Dirichlet, vector AND temperature",
    set(behind)=={"DIRICHLET","VEC_SCALE","TEMP_SCALE"}, ",".join(behind))

# risk-coverage gap quoted at 30% and 90% coverage
Dz=np.load(_os.path.join(_RES, "curve_data.npz"))
G=np.linspace(0.05,1.0,191)
def _curve(m):
    out=[]
    for d,a in CFG:
        c=Dz[f"{d}|{a}|{m}|conf"]; r=Dz[f"{d}|{a}|{m}|corr"].astype(float)
        o=np.argsort(-c,kind="stable"); e=1.0-r[o]
        out.append(np.interp(G,np.arange(1,len(e)+1)/len(e),np.cumsum(e)/np.arange(1,len(e)+1)))
    return np.vstack(out).mean(0)
apx=_curve("APEX"); bst=np.min([_curve(b) for b in
    ["BASE","TEMP_SCALE","VEC_SCALE","ISOTONIC","DIRICHLET"]],axis=0)
gp=100*(bst-apx)
chk("APEX below every baseline at every coverage", bool(np.all(apx<bst)))
chk("gap ~0.7 points at 30% coverage", abs(gp[np.argmin(abs(G-0.30))]-0.7)<0.15,
    f"{gp[np.argmin(abs(G-0.30))]:.2f}")
chk("gap ~3.5 points at 90% coverage", abs(gp[np.argmin(abs(G-0.90))]-3.5)<0.15,
    f"{gp[np.argmin(abs(G-0.90))]:.2f}")

# ultrasound: widest FOLD-TO-FOLD spread, not widest across backbones
fold={d: float(np.std([float(r["AURC"]) for r in rows
      if r["method"]=="APEX" and r["domain"]==d], ddof=1)) for d in DOMS}
chk("ultrasound has the widest fold-to-fold AURC spread",
    max(fold,key=fold.get)=="ULTRASOUND", f"{max(fold,key=fold.get)}")

print()
print("SECOND AUDIT PASSED" if ok2 else "*** SECOND AUDIT FAILED ***")

# ---------------------------------------------------------------- audit round 3
print()
print("final audit round")
ok3 = True
def c3(label, cond, detail=""):
    global ok3; ok3 &= bool(cond)
    print(f"  [{'OK ' if cond else 'BAD'}] {label:<56} {detail}")

M15=["BAcc","F1","AUC","ECE","dECE","SCE","MCE","OCE","Brier","BSS","NLL","AURC","EAURC","Risk@90","Risk@80"]
UP={"BAcc","F1","AUC","BSS","CCC"}
ARCHS_=["ViT","ConvNeXt","MobileNetV3"]
def per_arch(m,k,a):
    return float(np.nanmean([cm(m,k)[(d,a)] for d in DOMS]))

# the corrected claim: exactly MCE and OCE are lost on every backbone vs uncalibrated
lost=[]
for k in M15:
    if all((per_arch("APEX",k,a) < per_arch("BASE",k,a)) if k in UP
           else (per_arch("APEX",k,a) > per_arch("BASE",k,a)) for a in ARCHS_):
        lost.append(k)
c3("APEX trails uncalibrated on every backbone in exactly {MCE, OCE}",
   set(lost)=={"MCE","OCE"}, ",".join(lost))

# Dirichlet is strongest baseline on ECE, Brier, NLL by mean rank (Table II caption)
M7=["BASE","MC_DROPOUT","TEMP_SCALE","VEC_SCALE","ISOTONIC","DIRICHLET","APEX"]
for k in ("ECE","Brier","NLL"):
    R=np.vstack([rankdata([cm(m,k)[c] for m in M7]) for c in CFG])
    mr={m: float(R[:,i].mean()) for i,m in enumerate(M7) if m!="APEX"}
    c3(f"Dirichlet is the best-ranked baseline on {k}",
       min(mr,key=mr.get)=="DIRICHLET", f"best={min(mr,key=mr.get)}")

# every figure and table is referenced in the body
import re, subprocess
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_PDF=_os.environ.get("APEX_PAPER_PDF")
if not _PDF or not _os.path.exists(_PDF):
    for _c in ("main.pdf",
               _os.path.join(_ROOT,"paper","APEX_paper.pdf"),
               _os.path.join(_ROOT,"paper","APEX_paper_6pages.pdf"),
               _os.path.join(_ROOT,"paper","main.pdf")):
        if _os.path.exists(_c): _PDF=_c; break
txt=subprocess.run(["pdftotext","-layout",_PDF,"-"],capture_output=True,text=True).stdout if _PDF else ""
if not txt:
    print("  (paper PDF not found -- skipping the cross-reference checks)")
for n in (1,2,3,4):
    c3(f"Figure {n} is referenced in the body text",
       len(re.findall(rf"Fig\. {n}[^0-9]", txt))>1, f"{len(re.findall(rf'Fig. {n}[^0-9]', txt))} mentions")
for t in ("I","II"):
    c3(f"Table {t} is referenced in the body text",
       len(re.findall(rf"Table {t}\b", txt))>1)

print()
print("FINAL ROUND PASSED" if ok3 else "*** FINAL ROUND FAILED ***")



# ---------------------------------------------------------------- audit round 4
print()
print("rewritten ECE paragraph")
ok4=True
def c4(l,c,d=""):
    global ok4; ok4 &= bool(c); print(f"  [{'OK ' if c else 'BAD'}] {l:<52} {d}")
g_=lambda m,k: float(np.mean(list(cm(m,k).values())))
for m,want in (("TEMP_SCALE",3.28),("VEC_SCALE",3.11),("ISOTONIC",3.47),
               ("DIRICHLET",2.99),("APEX",3.39),("BASE",4.95)):
    c4(f"{m} mean ECE = {want}e-2", abs(g_(m,'ECE')*100-want)<5e-3, f"{g_(m,'ECE')*100:.3f}")
a=col("APEX","ECE"); d=col("DIRICHLET","ECE")
c4("Dirichlet beats APEX on ECE in 12 of 18", int(np.sum(d<a))==12, f"{int(np.sum(d<a))}/18")
mce={m: g_(m,"MCE") for m in BASE+["APEX"]}
c4("APEX has the highest MCE of any method", max(mce,key=mce.get)=="APEX", f"max={max(mce,key=mce.get)}")
gaps={m: g_(m,"Conf")-g_(m,"BAcc") for m in BASE+["APEX"]}
c4("APEX confidence-accuracy gap is smallest in magnitude",
   min(gaps,key=lambda m:abs(gaps[m]))=="APEX", f"APEX {gaps['APEX']:+.4f}")
c4("that gap rounds to 0.9 points", abs(abs(gaps["APEX"])*100-0.9)<0.05, f"{abs(gaps['APEX'])*100:.2f}")
print()
print("ROUND FOUR PASSED" if ok4 else "*** ROUND FOUR FAILED ***")

sys.exit(0 if (ok and ok2 and ok3 and ok4) else 1)
