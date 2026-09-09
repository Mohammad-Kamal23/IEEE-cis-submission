# APEX — Latent-Space Kernel Fusion for Selective Prediction with Frozen Medical Image Classifiers

Code, data manifests, held-out predictions and the full analysis chain for the
paper. Everything reported in the paper can be regenerated from this repository
on a laptop CPU in about a minute.

**Mohammad Kamal Abdulaziz**, King Abdullah II School of Information Technology,
The University of Jordan · supervised by **Dr. Rizik Al-Sayyed**, King Abdullah II
School of Information Technology, The University of Jordan.

---

## What the method does

A classifier that is allowed to abstain is judged less by how often it is right
than by how well its confidence *orders* its own mistakes. APEX is a post-hoc
layer that improves that ordering without touching the network. It leaves the
backbone frozen, reads the penultimate representation the classifier computes and
then discards, concatenates it with 62 image descriptors, projects the pair onto a
64-dimensional subspace by truncated SVD, fits three kernel experts there (RBF
SVM, cubic-polynomial SVM, k-NN), and mixes their posteriors with the original
softmax under a confidence-conditional gate. The mixture weights are fitted by
minimising negative log-likelihood — a strictly proper scoring rule.

## Headline result

Mean over 18 configurations (6 imaging modalities × 3 backbones, five-fold
cross-validation, 90 trained heads). Arrows give the preferred direction.

| Method | BAcc ↑ | ECE ↓ | SCE ↓ | Brier ↓ | NLL ↓ | AURC ↓ | Risk@90 ↓ |
|---|---|---|---|---|---|---|---|
| Uncalibrated   | 0.8523 | 0.0495 | 0.0468 | 0.2062 | 0.3710 | 0.0486 | 0.1071 |
| MC dropout     | 0.8531 | 0.0571 | 0.0497 | 0.2088 | 0.3773 | 0.0488 | 0.1078 |
| Temp. scaling  | 0.8523 | 0.0328 | 0.0393 | 0.2015 | 0.3597 | 0.0484 | 0.1066 |
| Vector scaling | 0.8583 | 0.0311 | 0.0313 | 0.1931 | 0.3488 | 0.0444 | 0.1000 |
| Isotonic       | 0.8569 | 0.0347 | 0.0328 | 0.1968 | 0.4466 | 0.0484 | 0.1018 |
| Dirichlet      | 0.8580 | **0.0299** | 0.0304 | 0.1930 | 0.3443 | 0.0447 | 0.1014 |
| **APEX**       | **0.8941** | 0.0339 | **0.0297** | **0.1432** | **0.2589** | **0.0271** | **0.0652** |

AURC, Brier and NLL improve in **all 18 of 18** configurations against **every one
of the six baselines**, at p = 7.6 × 10⁻⁶ — the smallest value a two-sided
Wilcoxon signed-rank test can return at n = 18. Balanced accuracy and risk at 90%
coverage do so in at least 17 of 18.

Expected calibration error does **not** improve past the parametric calibrators,
and the paper says so plainly. That asymmetry is the point: selective risk is
invariant to any strictly monotone rescaling of confidence and ECE is not
(Proposition 1), and in 15 of 18 configurations the method with the lowest
calibration error is not the method with the lowest selective risk.

## Reproduce it

```bash
git clone https://github.com/Mohammad-Kamal23/IEEE-cis-submission.git
cd IEEE-cis-submission
pip install -r requirements.txt
python reproduce.py
```

No GPU. No trained weights. No image data. About a minute.

`reproduce.py` runs seven steps:

1. **Recompute every metric from the committed predictions.** It loads the 90
   held-out prediction files in `results/probs/` and recomputes all nineteen
   metrics. It does *not* reimplement them — it imports `evaluate_all` from
   `src/apex_pipeline.py`, the same function that produced the paper, and applies
   it to the same predictions. The result is compared against
   `results/FINAL_RESULTS_perfold.csv`, the file the paper's tables are built
   from. **Expected output: max |Δ| = 0.000e+00.** If the tables had been built
   from anything other than these predictions, this step fails.
2. Rebuild Table I — the mean over the 18 configurations.
3. Rebuild Table II — Wilcoxon signed-rank tests and mean ranks.
4. Re-derive **every number quoted in the paper** from the CSV
   (`analysis/verify_claims.py`, 69 assertions across four rounds), including the
   figures quoted in the abstract, the ablation deltas, the τ sweep and the
   per-dataset reductions.
5. Verify **Proposition 1** numerically (`analysis/rq1_invariance.py`).
6. Redraw all four figures.
7. **Compare what you just generated against our committed reference outputs** —
   `reference_outputs/` holds the tables and statistics files our run produced, and
   step 7 diffs yours against them. This is the direct answer to "do I get the same
   thing you got?"

Exit status is 0 only if every check passes. A clean run ends with:

```
  [OK ] every recomputed metric matches the published CSV     max |Δ| = 0.000e+00
  ...
  [OK ] tab1.tex                 exact match
  [OK ] tab2.tex                 exact match

  ALL CHECKS PASSED -- every published number was regenerated from
  the committed predictions, and all four figures were redrawn.
```

87 checks in total.

`python analysis/rq1_invariance.py` separately verifies Proposition 1 by applying
five strictly increasing maps to the uncalibrated confidences of all 18
configurations: AURC does not move in any digit, while ECE ranges from
4.95 × 10⁻² to 1.44 × 10⁻¹.

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the three levels of reproduction —
from the committed predictions, from the cached features, and from the raw images.

## Layout

```
reproduce.py            one command; regenerates and checks everything
src/
  apex_pipeline.py      the full pipeline: repair, train, extract, bench, analyse
  clean_datasets.py     the leakage audit and quarantine tool
  verify_data.py        integrity checks over the image corpora
  probe_geometry.py     the latent-geometry mechanism probe
  RUN_APEX.ps1          Windows launcher
analysis/
  verify_claims.py      re-derives every number quoted in the paper
  rq1_invariance.py     numerical check of Proposition 1
  stats.py stats2.py    aggregation and significance testing
  mktables.py           emits tab1.tex / tab2.tex
  fig1.py fig_rc.py fig_rel.py fig_abl.py figstyle.py
results/
  FINAL_RESULTS_perfold.csv   2,070 rows: 23 methods × 18 configurations × 5 folds
  FINAL_RESULTS_perconfig.csv aggregated per configuration
  probs/                      90 files — held-out predictions of every method,
                              including APEX, plus ground-truth labels
  curve_data.npz              risk–coverage curves
  summary.json                pipeline summary
  duplicate_report.json       hash-level duplicate analysis
  repair_report.json          frozen per-dataset index
  RQ_RESULTS.md               mechanism experiments
reference_outputs/      what our run produced; step 7 diffs yours against it
weights_heads/
  heads.npz             all 90 trained models, 5 MB (see below)
  MANIFEST.json         architectures, trained tensors, backbone fingerprints
figures/                4 figures, PDF and PNG
paper/                  final PDF, LaTeX source, bibliography, IEEEtran files,
                        and APEX_overleaf.zip ready to drop into Overleaf
docs/                   datasets, data audit, reproduction guide
```

## The trained models, in 5 MB

The 90 checkpoints are 107 MB each — 9.4 GB — and almost all of it is redundant.
The backbones are frozen, and we checked rather than assumed it: across all 90
checkpoints every backbone tensor is **byte-identical**. What actually changed
during training is 954,823 numbers.

`weights_heads/heads.npz` stores exactly that, and `src/load_trained_model.py`
rebuilds any model from it:

```python
from src.load_trained_model import load_model
model = load_model("ConvNeXt", "BLOODCELL", fold=1)
```

It fetches the ImageNet backbone through `timm`/`torchvision`, loads the stored
tensors, and recomputes a SHA-256 fingerprint of the frozen part to confirm it
matches what was there during training — so a future change to the upstream
pretrained weights is reported rather than silently producing different numbers.

One honest detail. For ViT and ConvNeXt only the classifier weight and bias ever
change. MobileNetV3 uses batch normalisation, whose running statistics adapt
during training even with every weight frozen, so its 138 BatchNorm buffers are
stored too. Those updates come only from the inner training split and never from
an evaluation fold — we verified this in the training loop — so they carry no
leakage, but they are part of the trained model and are needed to reproduce it.
The paper states this in Section IV-C.

Requires `torch` and `timm` (`pip install -r requirements-full.txt`). **None of
this is needed to reproduce the results** — that is what `results/probs/` is for.

## Data integrity

Before any result in this paper was computed, all 33,588 candidate image files
were hashed and audited. **1,337 files (4.0%) were quarantined** and every number
in the paper is computed on the 32,251 that remain:

| Reason | Files |
|---|---|
| Binary segmentation masks a file glob was loading as ultrasound scans | 798 |
| Byte-identical duplicate copies within a class | 531 |
| Images appearing under two different labels (both members removed) | 8 |

The masks were the consequential ones — they were half the nominal ultrasound
corpus and made its classes separable by silhouette alone. Full breakdown in
[docs/DATA_AUDIT.md](docs/DATA_AUDIT.md); the audit is rerunnable via
`src/clean_datasets.py`.

## Data

The six corpora are public and are **not redistributed here**. Download links,
licences and the exact class structure are in [docs/DATASETS.md](docs/DATASETS.md).
`results/repair_report.json` freezes the exact image count and class list per
dataset so the splits can be rebuilt identically.

## Citation

```bibtex
@inproceedings{abdulaziz2026apex,
  title     = {{APEX}: Latent-Space Kernel Fusion for Selective Prediction
               with Frozen Medical Image Classifiers},
  author    = {Abdulaziz, Mohammad Kamal and Al-Sayyed, Rizik},
  year      = {2026},
  note      = {IEEE CIS Jordan AI Research Contest, Track A}
}
```

## Licence

Code is released under the MIT Licence (see `LICENSE`). The datasets are the
property of their respective providers and are governed by their own terms.
