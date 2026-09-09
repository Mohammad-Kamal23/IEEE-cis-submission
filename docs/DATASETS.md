# Datasets

Six public corpora, **32,251 images** after the integrity audit
(see [DATA_AUDIT.md](DATA_AUDIT.md)). The images themselves are **not
redistributed in this repository** — each remains the property of its provider
and is governed by that provider's terms. Download them from the sources below.

| Domain | Modality | Images | Classes | Source |
|---|---|---|---|---|
| `BLOODCELL` | Blood smear microscopy | 12,513 | 4 — EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL | Mooney, *Blood Cell Images*, Kaggle — `kaggle.com/datasets/paultimothymooney/blood-cells` |
| `BRAINTUMOR` | Brain MRI | 7,013 | 4 — glioma, meningioma, notumor, pituitary | Nickparvar, *Brain Tumor MRI Dataset*, Kaggle — `kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset` |
| `CHEST-XRAY` | Chest radiography | 3,994 | 2 — NORMAL, PNEUMONIA | Kermany et al., *Cell* 172(5):1122–1131, 2018 |
| `ENDOSCOPIC` | Endoscopy | 3,990 | 4 — Diverticulosis, Neoplasm, Peritonitis, Ureters | Heartzhacker, *Medical Imaging*, Kaggle — `kaggle.com/datasets/heartzhacker/medical-imaging` |
| `RETINAL` | Retinal OCT | 3,963 | 4 — CNV, DME, Drusen, Normal | Kermany et al., *Cell* 172(5):1122–1131, 2018 |
| `ULTRASOUND` | Breast ultrasound | 778 | 3 — benign, malignant, normal | Al-Dhabyani et al., *Data in Brief* 28:104863, 2020 |

All six are research datasets released for academic use. Check each provider's
current licence before redistributing or using commercially. No patient-level
identifiers are present in any of them, and none are included in this repository.

## Rebuilding the exact splits

`results/repair_report.json` freezes, per dataset, the image count, the class
list, the descriptor dimension and the build timestamp. `src/apex_pipeline.py`
rebuilds a canonical index keyed on relative path, then draws stratified
five-fold splits under `SEED = 42`. Point `BASE_DIR` in `src/apex_pipeline.py` at
the directory holding the six folders and run:

```
python src/apex_pipeline.py --stage repair
```

Within each training fold the data is split 80/20 into `D_fit` (for the kernel
experts) and `D_cal` (for the mixture weights). The evaluation fold is seen by no
component at any point.

## Expected directory layout

```
research/
  BloodCell/          <class>/*.jpeg
  BRAINTUMOR/         <class>/*.jpg
  CHEST-XRAY/         <class>/*.jpeg
  ENDOSCOPIC/         <class>/*.jpg
  RetinalOCT/         <class>/*.jpeg
  BreastUltrasound/   <class>/*.png
```

Folder names are resolved case-insensitively by `resolve_domain_dir()`; class
labels are taken from the immediate parent directory of each image.

## A note on the ultrasound corpus

The breast ultrasound dataset ships each scan alongside a binary segmentation
mask with a `_mask` suffix. Those masks are **not images of tissue** and must be
excluded. Loading them as scans halves the effective corpus and makes the classes
separable by silhouette alone. `src/apex_pipeline.py` skips any file whose name
contains `_mask`, `_mask_1` or `-mask`; `src/clean_datasets.py` quarantines any
already present. This is the single most consequential data defect we found.
