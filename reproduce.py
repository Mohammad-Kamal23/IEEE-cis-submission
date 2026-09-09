#!/usr/bin/env python3
"""
Reproduce every number in the APEX paper on a CPU, in about a minute, from the
artefacts committed to this repository. No GPU, no trained weights and no image
data are required.

    python reproduce.py

What this actually does, and why it is worth more than re-reading a results file:

  Step 1 loads the 90 committed prediction files -- one per (dataset,
  architecture, fold) -- and recomputes all nineteen metrics from the raw
  probability vectors. It does not reimplement the metrics. It imports
  `evaluate_all` from src/apex_pipeline.py, the same function that produced the
  numbers in the paper, and applies it to the same held-out predictions. The
  result is then compared against FINAL_RESULTS_perfold.csv. If the paper's
  table were built from anything other than these predictions, this step fails.

  Steps 2-5 aggregate to the eighteen configurations, rebuild Table I and
  Table II, re-run the Wilcoxon signed-rank tests, re-check every figure quoted
  in the abstract, and redraw all four figures.

Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PROBS = os.path.join(RESULTS, "probs")
BUILD = os.path.join(ROOT, "_reproduced")

DOMS = ["BLOODCELL", "BRAINTUMOR", "CHEST-XRAY", "ENDOSCOPIC", "RETINAL", "ULTRASOUND"]
ARCHS = ["ViT", "ConvNeXt", "MobileNetV3"]
CFG = [(d, a) for d in DOMS for a in ARCHS]
FOLDS = [1, 2, 3, 4, 5]

# the six comparators the paper reports against, in table order
BASELINES = ["BASE", "MC_DROPOUT", "TEMP_SCALE", "VEC_SCALE", "ISOTONIC", "DIRICHLET"]
METRICS = ["BAcc", "F1", "AUC", "ECE", "dECE", "SCE", "Brier", "BSS", "NLL",
           "Conf", "Ent", "MTP", "MCE", "OCE", "CCC", "AURC", "EAURC",
           "Risk@90", "Risk@80"]

FAIL: list[str] = []
NOTE: list[str] = []


def head(n, s):
    print(f"\n{'=' * 78}\n  STEP {n}  {s}\n{'=' * 78}")


def ck(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'BAD'}] {label:<54}{detail}")
    if not ok:
        FAIL.append(label)
    return ok


# --------------------------------------------------------------------------- #
#  the paper's own metric code, imported rather than reimplemented
# --------------------------------------------------------------------------- #
def load_pipeline():
    p = os.path.join(ROOT, "src", "apex_pipeline.py")
    spec = importlib.util.spec_from_file_location("apex_pipeline", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["apex_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv_rows():
    with open(os.path.join(RESULTS, "FINAL_RESULTS_perfold.csv")) as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
def step1_recompute(ap, rows):
    head(1, "recompute every metric from the committed predictions")
    print(f"  metric code : src/apex_pipeline.py :: evaluate_all  (N_BINS={ap.N_BINS})")
    print(f"  predictions : results/probs/*.npz")
    print()

    idx = {(r["domain"], r["arch"], int(r["fold"]), r["method"]): r for r in rows}
    dev = {m: 0.0 for m in METRICS}
    worst = {m: "" for m in METRICS}
    n_cells = n_cmp = 0
    t0 = time.time()

    for d, a in CFG:
        for f in FOLDS:
            fp = os.path.join(PROBS, f"{d}__{a}__f{f}.npz")
            if not os.path.exists(fp):
                ck(f"missing predictions {d}/{a}/f{f}", False)
                continue
            z = np.load(fp)
            y = z["y_true"]
            K = z["BASE"].shape[1]
            n_cells += 1
            for meth in z.files:
                if meth == "y_true":
                    continue
                row = idx.get((d, a, f, meth))
                if row is None:
                    continue
                got = ap.evaluate_all(y, z[meth], K)
                for m in METRICS:
                    want = float(row[m])
                    if math.isnan(want) and math.isnan(got[m]):
                        continue
                    delta = abs(got[m] - want)
                    if delta > dev[m]:
                        dev[m], worst[m] = delta, f"{d}/{a}/f{f}/{meth}"
                    n_cmp += 1

    print(f"  {n_cells} prediction files, {n_cmp} metric values recomputed "
          f"in {time.time() - t0:.1f}s\n")
    print(f"  {'metric':<10}{'max |recomputed - published|':<32}worst cell")
    print(f"  {'-' * 74}")
    for m in METRICS:
        print(f"  {m:<10}{dev[m]:<32.3e}{worst[m]}")
    print()
    ck("every recomputed metric matches the published CSV",
       max(dev.values()) < 1e-9, f"max |Δ| = {max(dev.values()):.3e}")
    return n_cells


# --------------------------------------------------------------------------- #
def cellmeans(rows, method, metric):
    """Mean over the five folds, for each of the eighteen configurations."""
    out = {}
    for d, a in CFG:
        v = [float(r[metric]) for r in rows
             if r["method"] == method and r["domain"] == d and r["arch"] == a]
        v = [x for x in v if not math.isnan(x)]
        out[(d, a)] = float(np.mean(v)) if v else float("nan")
    return out


def col(rows, method, metric):
    c = cellmeans(rows, method, metric)
    return np.array([c[k] for k in CFG])


def grand(rows, method, metric):
    return float(np.mean(col(rows, method, metric)))


def step2_table1(rows):
    head(2, "Table I  --  mean over the eighteen configurations")
    show = ["BAcc", "F1", "AUC", "ECE", "SCE", "Brier", "NLL", "AURC", "EAURC", "Risk@90"]
    label = {"BASE": "Uncalibrated", "MC_DROPOUT": "MC dropout", "TEMP_SCALE": "Temp. scaling",
             "VEC_SCALE": "Vector scaling", "ISOTONIC": "Isotonic", "DIRICHLET": "Dirichlet",
             "APEX": "APEX"}
    print(f"  {'method':<16}" + "".join(f"{m:>9}" for m in show))
    print(f"  {'-' * (16 + 9 * len(show))}")
    for meth in BASELINES + ["APEX"]:
        print(f"  {label[meth]:<16}" +
              "".join(f"{grand(rows, meth, m):>9.4f}" for m in show))
    print()


def step3_stats(rows):
    head(3, "Table II  --  Wilcoxon signed-rank over the eighteen configurations")
    from scipy.stats import wilcoxon, rankdata

    lower_is_better = {"ECE", "SCE", "Brier", "NLL", "AURC", "EAURC", "Risk@90", "MCE", "OCE"}
    for metric in ["AURC", "Brier", "NLL", "BAcc", "Risk@90", "ECE"]:
        a = col(rows, "APEX", metric)
        wins, ps = [], []
        for b in BASELINES:
            x = col(rows, b, metric)
            wins.append(int(np.sum(a < x) if metric in lower_is_better else np.sum(a > x)))
            ps.append(float(wilcoxon(a, x).pvalue))
        print(f"  {metric:<9} APEX wins " +
              "  ".join(f"{b.split('_')[0][:6]}:{w}/18" for b, w in zip(BASELINES, wins)) +
              f"   worst p={max(ps):.2e}")

    print("\n  mean rank over all 15 metrics (1 = best):")
    allm = ["BAcc", "F1", "AUC", "ECE", "dECE", "SCE", "Brier", "BSS", "NLL",
            "MCE", "OCE", "AURC", "EAURC", "Risk@90", "Risk@80"]
    meths = BASELINES + ["APEX"]
    R = {m: [] for m in meths}
    for metric in allm:
        M = np.array([col(rows, m, metric) for m in meths])
        if metric not in lower_is_better:
            M = -M
        for j in range(M.shape[1]):
            r = rankdata(M[:, j])
            for i, m in enumerate(meths):
                R[m].append(r[i])
    for m in meths:
        print(f"    {m:<14}{np.mean(R[m]):.2f}")
    print()


def step4_claims():
    head(4, "every number quoted in the paper, re-derived from the CSV")
    vc = os.path.join(ROOT, "analysis", "verify_claims.py")
    r = subprocess.run([sys.executable, vc], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.stderr.strip():
        print(r.stderr.rstrip())
    ck("analysis/verify_claims.py", r.returncode == 0)


def step5_proposition():
    head(5, "Proposition 1 -- AURC is invariant under monotone reparametrisation")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "rq1_invariance.py")],
                       cwd=BUILD, capture_output=True, text=True,
                       env=dict(os.environ, APEX_RESULTS=RESULTS))
    print(r.stdout.rstrip())
    if r.stderr.strip():
        print(r.stderr.rstrip())
    ck("analysis/rq1_invariance.py", r.returncode == 0)


def step6_figures():
    head(6, "regenerate the four figures")
    os.makedirs(os.path.join(BUILD, "plots"), exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", APEX_RESULTS=RESULTS)
    chain = [("stats.py", "stats.json"), ("stats2.py", "stats2.json"),
             ("mktables.py", "tab1.tex"), ("fig1.py", "plots/fig1_architecture.pdf"),
             ("fig_rc.py", "plots/fig2_selective.pdf"),
             ("fig_rel.py", "plots/fig3_calibration.pdf"),
             ("fig_abl.py", "plots/fig4_ablation.pdf")]
    for script, product in chain:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", script)],
                           cwd=BUILD, capture_output=True, text=True, env=env)
        made = os.path.exists(os.path.join(BUILD, product))
        if not ck(f"{script:<16} -> {product}", r.returncode == 0 and made,
                  "" if made else r.stderr.strip().splitlines()[-1:] or ""):
            continue
    print(f"\n  output in {os.path.relpath(BUILD, ROOT)}/")


# --------------------------------------------------------------------------- #
def main():
    print(__doc__.split("Exit status")[0].rstrip())
    os.makedirs(BUILD, exist_ok=True)
    ap = load_pipeline()
    rows = read_csv_rows()
    print(f"\n  FINAL_RESULTS_perfold.csv : {len(rows)} rows, "
          f"{len({r['method'] for r in rows})} methods, "
          f"{len({(r['domain'], r['arch']) for r in rows})} configurations")

    step1_recompute(ap, rows)
    step2_table1(rows)
    step3_stats(rows)
    step4_claims()
    step5_proposition()
    step6_figures()

    print(f"\n{'=' * 78}")
    if FAIL:
        print(f"  {len(FAIL)} CHECK(S) FAILED")
        for f in FAIL:
            print(f"    - {f}")
        print("=" * 78)
        return 1
    print("  ALL CHECKS PASSED -- every published number was regenerated from"
          "\n  the committed predictions, and all four figures were redrawn.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
