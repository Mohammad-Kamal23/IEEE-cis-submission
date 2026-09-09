import json, numpy as np
from scipy.stats import wilcoxon, rankdata

S = json.load(open("stats.json")); cfg=S["cfg"]; grand=S["grand"]
DOMS=["BLOODCELL","BRAINTUMOR","CHEST-XRAY","ENDOSCOPIC","RETINAL","ULTRASOUND"]
ARCHS=["ViT","ConvNeXt","MobileNetV3"]
CFG=[f"{d}|{a}" for d in DOMS for a in ARCHS]

M15 = [("BAcc","Balanced accuracy",1),("F1","Macro F1",1),("AUC","Macro AUROC",1),
       ("ECE","ECE",0),("dECE","Debiased ECE",0),("SCE","Class-wise SCE",0),
       ("MCE","Maximum CE",0),("OCE","Over-confidence error",0),
       ("Brier","Brier score",0),("BSS","Brier skill score",1),("NLL","NLL",0),
       ("AURC","AURC",0),("EAURC","Excess AURC",0),
       ("Risk@90","Risk at 90\\% coverage",0),("Risk@80","Risk at 80\\% coverage",0)]
SHOW=["BASE","DIRICHLET","APEX"]
NICEM={"BASE":"None","DIRICHLET":"Dirichlet","APEX":"\\textbf{APEX}"}

def bymean(m, metric, arch):
    v=[cfg[m][f"{d}|{arch}"][metric] for d in DOMS]
    return float(np.nanmean(v))

# ---------------------------------------------------------------- TABLE I
L=[]
L.append("\\begin{table*}[t]")
L.append("\\centering")
L.append("\\caption{All fifteen metrics per backbone, averaged over the six datasets "
         "(each entry is itself a mean over five folds). \\emph{None} is the uncalibrated "
         "softmax; Dirichlet is the strongest baseline by mean rank on ECE, Brier and NLL. "
         "Arrows give the preferred direction; best of each block in bold.}")
L.append("\\label{tab:allmetrics}")
L.append("\\setlength{\\tabcolsep}{3.0pt}\\renewcommand{\\arraystretch}{0.78}")
L.append("\\scriptsize")
L.append("\\begin{tabular}{l c *{3}{ccc}}")
L.append("\\toprule")
L.append(" & & \\multicolumn{3}{c}{ViT-S/16 ($d{=}384$)} & \\multicolumn{3}{c}{ConvNeXt-T ($d{=}768$)}"
         " & \\multicolumn{3}{c}{MobileNetV3-L ($d{=}1280$)} \\\\")
L.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}")
L.append("Metric & Dir. & " + " & ".join(["None","Dirich.","\\textbf{APEX}"]*3) + " \\\\")
L.append("\\midrule")
for key,name,up in M15:
    cells=[]
    for a in ARCHS:
        vals=[bymean(m,key,a) for m in SHOW]
        best=int(np.argmax(vals)) if up else int(np.argmin(vals))
        for i,v in enumerate(vals):
            t=f"{v:.4f}"
            cells.append("\\textbf{"+t+"}" if i==best else t)
    arr = "$\\uparrow$" if up else "$\\downarrow$"
    L.append(name + " & " + arr + " & " + " & ".join(cells) + " \\\\")

L.append("\\bottomrule")
L.append("\\end{tabular}")
L.append("\\end{table*}")
open("tab1.tex","w").write("\n".join(L)+"\n")

# ---------------------------------------------------------------- TABLE II
MAIN=["BASE","MC_DROPOUT","TEMP_SCALE","VEC_SCALE","ISOTONIC","DIRICHLET","APEX"]
NICE={"BASE":"Uncalibrated softmax","MC_DROPOUT":"MC dropout \\cite{gal2016dropout}",
      "TEMP_SCALE":"Temperature scaling \\cite{guo2017calibration}","VEC_SCALE":"Vector scaling \\cite{guo2017calibration}",
      "ISOTONIC":"Isotonic regression \\cite{zadrozny2002transforming}","DIRICHLET":"Dirichlet calibration \\cite{kull2019beyond}",
      "APEX":"\\textbf{APEX (ours)}"}
KEYS=[("BAcc",1),("ECE",0),("SCE",0),("Brier",0),("NLL",0),("AURC",0),("Risk@90",0)]
def col(m,k): return np.array([cfg[m][c][k] for c in CFG])
L=[]
L.append("\\begin{table*}[t]")
L.append("\\centering")
L.append("\\caption{Mean $\\pm$ standard deviation over the 18 dataset$\\,\\times\\,$backbone "
         "configurations. The last two rows count the configurations APEX wins and give the worst "
         "two-sided Wilcoxon $p$ against any single baseline; $7.6\\times10^{-6}$ is the smallest "
         "value the test can return at $n{=}18$.}")
L.append("\\label{tab:headline}")
L.append("\\setlength{\\tabcolsep}{4.0pt}\\renewcommand{\\arraystretch}{0.90}")
L.append("\\footnotesize")
L.append("\\begin{tabular}{l ccccccc}")
L.append("\\toprule")
L.append("Method & BAcc $\\uparrow$ & ECE $\\downarrow$ & SCE $\\downarrow$ & Brier $\\downarrow$ "
         "& NLL $\\downarrow$ & AURC $\\downarrow$ & Risk@90 $\\downarrow$ \\\\")
L.append("\\midrule")
best={k:(max if up else min)(grand[m][k] for m in MAIN) for k,up in KEYS}
for m in MAIN:
    cells=[]
    for k,up in KEYS:
        v=col(m,k); t=f"{v.mean():.4f}\\,\\tiny{{$\\pm${v.std(ddof=1):.4f}}}"
        cells.append("\\textbf{"+t+"}" if abs(grand[m][k]-best[k])<1e-12 else t)
    L.append(f"{NICE[m]} & " + " & ".join(cells) + " \\\\")

    if m=="DIRICHLET": L.append("\\midrule")
L.append("\\midrule")
row=[]
for k,up in KEYS:
    a=col("APEX",k)
    wins=[]; ps=[]
    for b in MAIN[:-1]:
        bb=col(b,k)
        wins.append(int(np.sum(a>bb) if up else np.sum(a<bb)))
        ps.append(float(wilcoxon(a,bb).pvalue))
    row.append(f"{min(wins)}/18" if min(wins)==max(wins) else f"{min(wins)}--{max(wins)}/18")
L.append("APEX better in & " + " & ".join(row) + " \\\\")
row=[]
for k,up in KEYS:
    a=col("APEX",k)
    ps=[float(wilcoxon(a,col(b,k)).pvalue) for b in MAIN[:-1]]
    hi=max(ps)
    if   hi < 1e-4: row.append("$<10^{-4}$")
    elif hi < 1e-3: row.append("$<10^{-3}$")
    elif hi < 1e-2: row.append("$<10^{-2}$")
    else:           row.append(f"${hi:.2f}$")
L.append("worst $p$ vs.\\ any row & " + " & ".join(row) + " \\\\")
L.append("\\bottomrule")
L.append("\\end{tabular}")
L.append("\\end{table*}")
open("tab2.tex","w").write("\n".join(L)+"\n")

print("tab1.tex, tab2.tex written")
print()
for k,up in KEYS:
    a=col("APEX",k)
    print(f"  {k:<9}", " ".join(f"{b[:4]}:{int(np.sum(a>col(b,k)) if up else np.sum(a<col(b,k)))}/18"
                                f"(p={wilcoxon(a,col(b,k)).pvalue:.1e})" for b in MAIN[:-1]))
