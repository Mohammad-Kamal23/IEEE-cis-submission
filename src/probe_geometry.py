"""RQ2: does the frozen latent geometry predict error AFTER conditioning on the
model's own confidence?  No retraining -- reads the cached features only.

Protocol mirrors the pipeline exactly: train = first n_train rows, val = the rest.
Train is split 80/20 into fit / cal with SEED=42.  Neighbours are drawn ONLY from
the fit portion, so no test label is ever touched.  The correctness predictor is
fitted on cal and scored on val.
"""
import os, sys, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"Results","pipeline","features")
SEED=42
K=int(os.environ.get("PROBE_K","20"))
def feats(Zq, Zfit, yfit, pred, nn, K):
    d,i = nn.kneighbors(Zq, n_neighbors=K)
    lab = yfit[i]                                   # (n,K) neighbour labels
    C = lab.max()+1
    cnt = np.stack([(lab==c).sum(1) for c in range(C)],1).astype(float)/K
    H  = -(cnt*np.log(cnt+1e-12)).sum(1)            # neighbourhood label entropy
    agree = (lab == pred[:,None]).mean(1)           # do neighbours agree with the head?
    return np.column_stack([H, agree, d[:,0], d.mean(1)])

def run(key, arch, folds=(1,2,3,4,5)):
    out=[]
    for f in folds:
        p=os.path.join(BASE, f"{key}__{arch}__f{f}.npz")
        if not os.path.exists(p): continue
        z=np.load(p)
        Z=z["Z"].astype(np.float32); P=z["probs"].astype(np.float64); y=z["y"]
        ntr=int(z["n_train"][0])
        Ztr,Zva=Z[:ntr],Z[ntr:]; Ptr,Pva=P[:ntr],P[ntr:]; ytr,yva=y[:ntr],y[ntr:]
        idx_f, idx_c = train_test_split(np.arange(ntr), test_size=0.2,
                                        random_state=SEED, stratify=ytr)
        sc=StandardScaler().fit(Ztr[idx_f])
        Zf=sc.transform(Ztr[idx_f]); yf=ytr[idx_f]
        nn=NearestNeighbors(n_neighbors=K).fit(Zf)
        def conf_block(Pm):
            s=np.sort(Pm,1)
            return np.column_stack([s[:,-1], s[:,-1]-s[:,-2],
                                    -(Pm*np.log(Pm+1e-12)).sum(1)])
        # calibration split (fit the correctness model here)
        Zc=sc.transform(Ztr[idx_c]); Pc=Ptr[idx_c]; yc=ytr[idx_c]
        predc=Pc.argmax(1); okc=(predc==yc).astype(int)
        Xc_conf=conf_block(Pc); Xc_geo=feats(Zc,Zf,yf,predc,nn,K)
        # validation split (score here)
        Zv=sc.transform(Zva); predv=Pva.argmax(1); okv=(predv==yva).astype(int)
        Xv_conf=conf_block(Pva); Xv_geo=feats(Zv,Zf,yf,predv,nn,K)
        if okv.min()==okv.max() or okc.min()==okc.max(): continue
        r={}
        for name,(Xc,Xv) in {"conf":(Xc_conf,Xv_conf),
                             "conf+geo":(np.hstack([Xc_conf,Xc_geo]),
                                         np.hstack([Xv_conf,Xv_geo]))}.items():
            m=LogisticRegression(max_iter=2000,C=1.0).fit(Xc,okc)
            pv=m.predict_proba(Xv)[:,1]
            r[name]={"auroc":float(roc_auc_score(okv,pv)),
                     "nll":float(log_loss(okv,pv,labels=[0,1]))}
        out.append(r)
    if not out: return None
    a1=np.mean([o["conf"]["auroc"] for o in out]); a2=np.mean([o["conf+geo"]["auroc"] for o in out])
    n1=np.mean([o["conf"]["nll"]   for o in out]); n2=np.mean([o["conf+geo"]["nll"]   for o in out])
    return dict(n=len(out), auroc_conf=a1, auroc_geo=a2, d_auroc=a2-a1,
                nll_conf=n1, nll_geo=n2, d_nll=n2-n1)

if __name__=="__main__":
    key,arch = sys.argv[1], sys.argv[2]
    r=run(key,arch)
    print(json.dumps({"domain":key,"arch":arch,**(r or {})}))
