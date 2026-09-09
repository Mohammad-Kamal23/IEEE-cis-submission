# Reproducing the paper

Three levels, in increasing cost. Level 1 needs nothing but this repository and a
laptop. Levels 2 and 3 need artefacts too large for version control, listed at the
bottom.

---

## Level 1 — every number and figure, from the committed predictions

**Cost: about one minute, CPU only. No GPU, no model weights, no image data.**

```bash
pip install -r requirements.txt
python reproduce.py
```

This works because `results/probs/` contains, for each of the 90
(dataset × backbone × fold) cells, the **held-out predictions of every method** —
including APEX — together with the ground-truth labels. Every metric in the paper
is a function of those two arrays, so the whole results section can be rebuilt
without re-running anything expensive.

What it checks, in order:

| Step | What it does | Passing looks like |
|---|---|---|
| 1 | Recomputes all 19 metrics for all 12 probs-backed methods across 90 folds (20,520 values) using `evaluate_all` imported from `src/apex_pipeline.py`, and compares against `results/FINAL_RESULTS_perfold.csv` | `max \|Δ\| = 0.000e+00` |
| 2 | Rebuilds Table I | matches the paper |
| 3 | Rebuilds Table II — Wilcoxon signed-rank, win counts, mean ranks | 18/18 at p = 7.6e-06 |
| 4 | Runs `analysis/verify_claims.py` — 69 assertions re-deriving every number quoted in the paper, in four rounds | `ALL CLAIMS VERIFIED` |
| 5 | Regenerates `tab1.tex`, `tab2.tex` and all four figures | files appear in `_reproduced/` |

Exit status is 0 only if all of it passes.

The point of step 1 is that it is an *independent path*. It does not re-read the
results file and agree with itself; it goes back to the raw probability vectors
and the labels, applies the paper's own metric code, and only then compares. If
the published table had been produced by any other route, the deviation would not
be zero.

Separately:

```bash
python analysis/rq1_invariance.py
```

verifies Proposition 1 numerically — five strictly increasing maps applied to the
uncalibrated confidences of all 18 configurations leave AURC unchanged in every
digit while ECE moves by a factor of 2.9.

### A note on tied confidences

Isotonic regression is a step function and ties 71% of its confidences; on one
fold, 82% of its predictions carry a single identical confidence value. The
ordering of tied samples is arbitrary, so a naive `argsort` makes AURC depend on
the order rows happen to sit in memory — on that fold it moves by a factor of two.
`evaluate_all` therefore reports the **expectation over tie-breaking**: within a
run of equal confidence, every position takes that run's mean error. This is
order-independent, and it is identical to the naive value when confidences are
distinct — which they effectively are for every other method (tied fraction ≤ 0.03,
and 0.0006 for APEX). The choice affects only the isotonic row and changes no
claim in the paper: APEX still wins 18/18 on AURC against all six baselines at
p = 7.6 × 10⁻⁶ under either convention.

---

## Level 2 — refit APEX from the cached features

**Cost: minutes per configuration, CPU. Needs `features/` (785 MB).**

`results/probs/` lets you re-evaluate APEX but not re-fit it, because fitting the
kernel experts needs the latent representations. With the cached features you can
re-run the benchmark stage end to end without a GPU and without the images:

```bash
python src/apex_pipeline.py --stage bench --force bench
```

This refits the three experts on `D_fit`, refits the mixture weights on `D_cal`,
re-evaluates every baseline and ablation, and rewrites the per-fold CSV. It also
enables the mechanism probe:

```bash
python src/probe_geometry.py
```

which tests whether latent neighbourhood geometry predicts error after
conditioning on the model's own confidence (see `results/RQ_RESULTS.md`).

---

## Level 3 — train everything from the images

**Cost: about 22 hours on one CUDA GPU. Needs the six datasets and, if you want to
skip training, the trained heads.**

```powershell
# Windows
.\src\RUN_APEX.ps1 -Rebuild
```

```bash
# or directly
python src/apex_pipeline.py --stage all --force all
```

Stages run in order: `repair` (audit and freeze the index), `train` (90 linear
heads over frozen backbones, 10 epochs per fold), `extract` (cache latents,
logits, softmax and MC-dropout probabilities), `bench` (APEX, six baselines, the
ablations, the fair-comparison rows and the τ sweep), `analyse` (aggregation,
Wilcoxon tests, tables, figures).

Every unit of work checkpoints atomically, so a crash or a reboot loses at most
one fold and re-running the same command resumes. Download the datasets first —
see [DATASETS.md](DATASETS.md) — and run the audit in
[DATA_AUDIT.md](DATA_AUDIT.md) before training, or the numbers will not match.

**Important:** without `-Rebuild` / `--force`, the pipeline *resumes* — any stage
whose output already exists is skipped, and a run can finish in under a minute
while printing the previous run's numbers. If you have changed the image set, you
must force a rebuild, because the fold assignment changes and every cached
artefact keyed to the old folds is invalid.

---

## Artefacts not in version control

| Artefact | Size | Needed for | Where |
|---|---|---|---|
| `features/` — cached latents, logits, softmax, MC probabilities, 90 files | 785 MB | Level 2 | supplementary archive |
| `weights/` — 90 trained classification heads | 8.2 GB | skipping training in Level 3 | supplementary archive |
| the six image corpora | — | Level 3 from scratch | original providers, see [DATASETS.md](DATASETS.md) |

## Environment

Developed and verified on Python 3.10 with numpy 2.2, scipy 1.15,
scikit-learn 1.7 and matplotlib 3.10. Training used PyTorch with CUDA and `timm`
backbones. Level 1 has no pinned-version requirement beyond the floors in
`requirements.txt`; if a future scikit-learn changes `balanced_accuracy_score` or
`log_loss`, step 1 will report the deviation rather than hide it.
