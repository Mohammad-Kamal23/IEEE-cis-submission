# Data integrity audit

Every image file was hashed and audited before any result in the paper was
computed. **33,588 candidate files → 1,337 quarantined (4.0%) → 32,251 used.**

Reproduce the audit with:

```
python src/verify_data.py                 # report only
python src/clean_datasets.py --dry-run    # show what would move
python src/clean_datasets.py              # quarantine, with a manifest
python src/clean_datasets.py --restore    # undo
```

Nothing is deleted. Files are moved to `_quarantine/<DATASET>/<reason>/` and a
manifest records every move, so the operation is reversible and auditable.

## What was removed

| Dataset | Reason | Files |
|---|---|---|
| ULTRASOUND | segmentation mask loaded as a scan | 798 |
| RETINAL | byte-identical duplicate copy | 320 |
| BRAINTUMOR | byte-identical duplicate copy | 187 |
| CHEST-XRAY | byte-identical duplicate copy | 12 |
| ENDOSCOPIC | byte-identical duplicate copy | 10 |
| RETINAL | same image under two labels | 6 |
| ULTRASOUND | same image under two labels | 2 |
| BLOODCELL | byte-identical duplicate copy | 2 |
| | **total** | **1,337** |

Aggregated by cause: 798 segmentation masks, 531 duplicate copies (408 duplicate
groups), 8 files across 4 cross-class conflict groups.

## Why each category matters

**Segmentation masks.** The breast ultrasound corpus stores a binary mask beside
each scan. A file-extension glob had been loading all 798 of them as if they were
ultrasound images — 50.6% of that corpus. Masks are silhouettes, so the classes
become separable by shape alone and the reported accuracy is meaningless. This is
not visible in any metric; only inspecting the files surfaces it.

**Duplicate copies.** Under cross-validation, byte-identical copies of one image
land in different folds, so the model is evaluated on material it memorised
during training. This inflates every metric silently
(Barz & Denzler, *J. Imaging* 6(6):41, 2020). Detection is by SHA-256 over file
bytes; only exact duplicates within the same class were removed.

**Cross-class conflicts.** Four groups contained an identical image filed under
two different labels. At least one label in each group is wrong, and there is no
way to tell which, so **both members were removed** rather than guessing.

## Effect

The audit reduced the ultrasound corpus from 1,578 nominal files to 778 real
scans, which is why it is the smallest dataset in the paper and carries the
widest fold-to-fold spread. Every number reported in the paper was computed after
this removal; nothing from the pre-audit run survives in the results.

## Downstream invalidation

Changing the image set changes the fold assignment, so every artefact keyed to
the old folds — trained heads, cached features, benchmark cells — becomes invalid.
`invalidate_downstream()` in `src/apex_pipeline.py` moves them aside rather than
silently reusing them, and `RUN_APEX.ps1 -Rebuild` forces a full recomputation.
The results in this repository come from a complete 22-hour rebuild after the
audit, not from a partial refresh.
