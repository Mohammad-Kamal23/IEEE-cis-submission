#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APEX end-to-end pipeline  --  repair, retrain, benchmark, ablate, analyse.

Designed to be started once and left alone. Every unit of work writes an atomic
checkpoint the moment it finishes, so if the machine dies, reboots, or you stop
the script with Ctrl-C, restarting the identical command resumes from the last
completed unit. Nothing is ever recomputed unless you ask for it.

    python apex_pipeline.py                  # run everything, resume-safe
    python apex_pipeline.py --stage repair   # run one stage only
    python apex_pipeline.py --status         # what is done, what remains
    python apex_pipeline.py --force bench    # redo a stage from scratch

Stages, in order:
    repair    quarantine the duplicated CHEST-XRAY tree, rebuild the descriptor
              index keyed on relative path (fixes Retinal OCT / BloodCell
              filename collisions), freeze a canonical image index per dataset
    train     train any missing classification heads (frozen backbone, 10 epochs,
              drop_rate>0 so MC dropout is a real baseline)
    extract   cache latent features, logits, softmax and MC probabilities per
              (dataset, architecture, fold) so every later stage is fast
    bench     evaluate APEX, all baselines, all ablations, the fair-comparison
              rows and the tau sweep; save every probability vector
    analyse   per-fold std, Wilcoxon signed-rank, AURC, LaTeX tables, figures

Author's note: APEX here means one method -- the gated, tau=0.85 configuration.
Everything else in the output is either a baseline or a named ablation of it.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import faulthandler
import json
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# This pipeline deliberately does not import pandas anywhere.
#
# On this machine pandas probes for pyarrow at import time and pyarrow's native
# extension raises a Windows access violation while loading (see crash.log).
# That kills the interpreter without raising a catchable Python exception, which
# is what ended the first six runs. Everything pandas was doing here -- grouping,
# averaging, pivoting, writing CSVs -- is done with numpy and the csv module
# instead, so no stage can be taken down by a broken Arrow build.
# ---------------------------------------------------------------------------

# ============================================================================ #
#  CONFIGURATION -- edit BASE_DIR only if you move the project
# ============================================================================ #

BASE_DIR = r"C:\Users\homeb\Desktop\research"

OUT_DIR      = os.path.join(BASE_DIR, "Results", "pipeline")
WEIGHTS_DIR  = os.path.join(BASE_DIR, "weights")
DATAINFO_DIR = os.path.join(BASE_DIR, "Data-Info")
QUARANTINE   = os.path.join(BASE_DIR, "_quarantine_duplicates")

INDEX_DIR = os.path.join(OUT_DIR, "index")
FEAT_DIR  = os.path.join(OUT_DIR, "features")
CELL_DIR  = os.path.join(OUT_DIR, "cells")
PROB_DIR  = os.path.join(OUT_DIR, "probs")
LOG_DIR   = os.path.join(OUT_DIR, "logs")
FIG_DIR   = os.path.join(BASE_DIR, "IEEE_Paper_v2", "plots")
TEX_DIR   = os.path.join(BASE_DIR, "IEEE_Paper_v2")

# dataset folder name on disk (the pipeline resolves either spelling)
DOMAINS = {
    "BLOODCELL":  "BloodCell",
    "BRAINTUMOR": "BRAINTUMOR",
    "CHEST-XRAY": "CHEST-XRAY",
    "ENDOSCOPIC": "ENDOSCOPIC",
    "RETINAL":    "RetinalOCT",
    "ULTRASOUND": "BreastUltrasound",
}
# descriptor CSV prefix when it differs from the domain key
CSV_PREFIX = {
    "RETINAL": "RetinalOCT",
    "ULTRASOUND": "BreastUltrasound",
    "BLOODCELL": "BloodCell",
}

ARCHS = ["ViT", "ConvNeXt", "MobileNetV3"]
N_FOLDS = 5
SEED = 42

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
SKIP_DIR_TOKENS = ("__MACOSX", ".ipynb_checkpoints", "_quarantine")
# BUSI ships a binary segmentation mask beside every ultrasound scan
# (`benign (1).png` / `benign (1)_mask.png`). A mask is not an image of the
# patient; it is the answer drawn in white on black, and its silhouette alone
# separates the classes. Loading one as a training image is label leakage.
SKIP_FILE_TOKENS = ("_mask", "_mask_1", "-mask")

# APEX -- the single published configuration
TAU        = 0.85     # confidence gate threshold
LAMBDA     = 0.20     # base-model trust multiplier below the threshold
TARGET_DIM = 64       # manifold dimension
GAMMA      = 2.0      # focal exponent
TAU_SWEEP  = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

DROP_RATE   = 0.10    # so MC dropout is a genuine baseline
MC_SAMPLES  = 30
EPOCHS      = 10
SEL_FRACTION = 0.10   # inner split for checkpoint selection; outer fold stays untouched
LR          = 1e-3
BATCH_TRAIN = 32
BATCH_EVAL  = 128
N_BINS      = 15

# Cap on samples used to fit the kernel experts. RBF-SVM fitting is O(n^2);
# without a cap a 10k-sample fold can take 20+ minutes. 0 disables the cap.
MAX_FIT_SAMPLES = 0

QUICK_MODE = False     # set by --quick

_LOG_FH = None
_CRASH_FH = None


# ============================================================================ #
#  INFRASTRUCTURE
# ============================================================================ #

def log(msg: str, level: str = "INFO") -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {level:<5} {msg}"
    print(line, flush=True)
    if _LOG_FH:
        try:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
        except Exception:
            pass


def open_log() -> None:
    """Open the run log and arm the native-crash handler.

    faulthandler catches access violations and other hard crashes that never
    reach a Python `except` clause, and writes a stack trace to crash.log.
    """
    global _LOG_FH, _CRASH_FH
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    _LOG_FH = open(path, "a", encoding="utf-8")
    try:
        _CRASH_FH = open(os.path.join(LOG_DIR, "crash.log"), "a", encoding="utf-8")
        _CRASH_FH.write(f"\n===== run started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        _CRASH_FH.flush()
        faulthandler.enable(file=_CRASH_FH, all_threads=True)
    except Exception:
        pass
    log(f"log file: {path}")


def keep_awake() -> None:
    """Stop Windows from sleeping while the pipeline runs. No admin needed."""
    if platform.system() != "Windows":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
    try:
        r = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        if r == 0:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log("sleep inhibited for the duration of this run (screen may still switch off)")
    except Exception as e:                                    # pragma: no cover
        log(f"could not inhibit sleep: {e}", "WARN")


def release_awake() -> None:
    if platform.system() != "Windows":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_savez(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def done(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def fmt_eta(sec: float) -> str:
    return str(timedelta(seconds=int(max(sec, 0))))


def resolve_domain_dir(key: str) -> str | None:
    for cand in (DOMAINS[key], key):
        p = os.path.join(BASE_DIR, cand)
        if os.path.isdir(p):
            return p
    for name in os.listdir(BASE_DIR):                        # tolerate GDrive suffixes
        if name.upper().startswith(DOMAINS[key].upper()) and os.path.isdir(os.path.join(BASE_DIR, name)):
            return os.path.join(BASE_DIR, name)
    return None


# ============================================================================ #
#  STAGE 1 -- REPAIR
# ============================================================================ #

def walk_images(root: str):
    """Deterministic, sorted walk. Sorting is what makes folds reproducible."""
    out = []
    for cur, dirs, files in os.walk(root):
        if any(tok in cur for tok in SKIP_DIR_TOKENS):
            dirs[:] = []
            continue
        dirs.sort()
        for f in sorted(files):
            low = f.lower()
            if not low.endswith(IMG_EXT):
                continue
            if any(tok in low for tok in SKIP_FILE_TOKENS):
                continue                      # segmentation mask, not a scan
            out.append(os.path.join(cur, f))
    return out


def label_from_path(full: str, root: str) -> str:
    seg = os.path.relpath(full, root).split(os.sep)
    if len(seg) < 2:
        return "unknown"
    lab = seg[-2]
    if lab.lower() in ("train", "test", "val", "valid", "images") and len(seg) >= 3:
        lab = seg[-3]
    return lab


def read_descriptor_csv(path: str):
    """Read a descriptor CSV with the standard library only.

    The repair stage used to do this with pandas. On some Windows / Python 3.13
    builds that combination aborts the interpreter outright, with no Python
    traceback to catch, so this reads the file with `csv` instead. Slower by a
    second or two, and it cannot take the process down with it.

    Returns (key_list, matrix, column_names) where key_list[i] = (dx, id_stem),
    or (None, None, None). The (dx, id) pair is what makes the mapping unique:
    Retinal OCT numbers its files from zero inside every class folder, so the id
    alone collides four ways.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if not header:
            return None, None, None
        try:
            id_col = header.index("image_id")
        except ValueError:
            id_col = 0
        dx_col = header.index("dx") if "dx" in header else None

        rows = [r for r in rdr if r and len(r) >= len(header)]
        if not rows:
            return None, None, None

        # a column is numeric if it parses as a float in most of a sample
        sample = rows[: min(200, len(rows))]
        numeric = []
        for j, name in enumerate(header):
            if j == id_col or j == dx_col:
                continue
            ok = 0
            for r in sample:
                v = r[j].strip()
                if v in ("", "nan", "NaN", "NA", "None"):
                    ok += 1
                    continue
                try:
                    float(v)
                    ok += 1
                except ValueError:
                    break
            if ok == len(sample):
                numeric.append(j)
        if not numeric:
            return None, None, None

        ids = []
        mat = np.zeros((len(rows), len(numeric)), dtype=np.float32)
        for i, r in enumerate(rows):
            stem = r[id_col].replace("\\", "/").rsplit(".", 1)[0]
            dx = r[dx_col].strip() if dx_col is not None else None
            ids.append((dx, stem))
            for k, j in enumerate(numeric):
                v = r[j].strip()
                if v and v not in ("nan", "NaN", "NA", "None"):
                    try:
                        mat[i, k] = float(v)
                    except ValueError:
                        mat[i, k] = 0.0
        return ids, mat, [header[j] for j in numeric]


def invalidate_downstream(key: str) -> None:
    """A rebuilt index means new folds, so every artefact keyed to the old fold
    assignment is now wrong. Heads, cached features and benchmark cells for this
    dataset are moved to Results/pipeline/_stale/ (not deleted) so the next stage
    regenerates them instead of silently reusing results from a different split."""
    stale = os.path.join(OUT_DIR, "_stale")
    moved = 0
    targets = [(WEIGHTS_DIR, lambda f: f"_{key}_" in f or f"_{key}_Fold" in f),
               (FEAT_DIR,    lambda f: f.startswith(f"{key}__")),
               (CELL_DIR,    lambda f: f.startswith(f"{key}__"))]
    for d, match in targets:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not match(fn):
                continue
            dst_dir = os.path.join(stale, os.path.basename(d.rstrip(os.sep)))
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fn)
            if os.path.exists(dst):
                dst += f".{int(time.time())}"
            try:
                shutil.move(os.path.join(d, fn), dst)
                moved += 1
            except OSError:
                pass
    if moved:
        log(f"  {key:<11}   {moved} downstream artefact(s) -> _stale/ (folds changed)", "WARN")


def stage_repair(force: bool = False) -> None:
    log("=" * 78)
    log("STAGE 1/5  REPAIR")
    log("=" * 78)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # -- 1a. quarantine the duplicated CHEST-XRAY tree ------------------------
    cxr = resolve_domain_dir("CHEST-XRAY") if "CHEST-XRAY" in DOMAINS else None
    if cxr:
        moved = []
        for sub in ("chest_xray", "__MACOSX"):
            src = os.path.join(cxr, sub)
            if os.path.isdir(src):
                os.makedirs(QUARANTINE, exist_ok=True)
                dst = os.path.join(QUARANTINE, f"CHEST-XRAY_{sub}")
                if os.path.exists(dst):
                    dst += f"_{int(time.time())}"
                shutil.move(src, dst)
                moved.append(sub)
        if moved:
            log(f"  CHEST-XRAY: moved {moved} -> _quarantine_duplicates (nothing deleted)")
        else:
            log("  CHEST-XRAY: no duplicate tree present (already repaired)")

    # -- 1b. build the canonical index for every dataset ----------------------
    summary = {}
    for key in DOMAINS:
        idx_path = os.path.join(INDEX_DIR, f"{key}.json")
        if done(idx_path) and not force:
            meta = load_json(idx_path)
            if meta:
                # The cache is only valid if the files it names are still on disk
                # and no new ones appeared. Spot-check the paths, then re-count.
                probe = meta["paths"][:: max(1, len(meta["paths"]) // 200)]
                missing = sum(1 for q in probe if not os.path.exists(q))
                root_now = resolve_domain_dir(key)
                n_now = len(walk_images(root_now)) if root_now else -1
                if missing or n_now != meta["n_images"]:
                    log(f"  {key:<11} index STALE ({meta['n_images']} indexed, "
                        f"{n_now} on disk, {missing}/{len(probe)} probed paths gone)"
                        f" -- rebuilding", "WARN")
                else:
                    log(f"  {key:<11} index cached: {meta['n_images']} images, "
                        f"{len(meta['classes'])} classes, descriptors {meta['n_matched']}")
                    summary[key] = meta
                    continue

        try:
            root = resolve_domain_dir(key)
            if root is None:
                log(f"  {key:<11} SKIPPED -- folder not found", "WARN")
                continue
            log(f"  {key:<11} scanning {root} ...")

            paths = walk_images(root)
            log(f"  {key:<11}   {len(paths)} image files found")
            if not paths:
                log(f"  {key:<11} SKIPPED -- no images", "WARN")
                continue

            labels = [label_from_path(p, root) for p in paths]
            classes = sorted(set(labels))
            log(f"  {key:<11}   {len(classes)} classes: {classes[:8]}"
                + (" ..." if len(classes) > 8 else ""))
            if len(classes) < 2:
                log(f"  {key:<11} SKIPPED -- fewer than 2 classes", "WARN")
                continue

            log(f"  {key:<11}   checking for duplicate files ...")
            seen, dup = set(), 0
            for p in paths:
                try:
                    k = (os.path.basename(p).lower(), os.path.getsize(p))
                except OSError:
                    continue
                if k in seen:
                    dup += 1
                seen.add(k)
            if dup:
                log(f"  {key:<11}   WARNING: {dup} probable duplicate files remain", "WARN")

            labels_idx = [classes.index(l) for l in labels]
            labels = labels_idx          # integer class ids from here on
            desc_row = [-1] * len(paths)
            n_matched, dim = 0, 0
            csv_path = os.path.join(DATAINFO_DIR, f"{CSV_PREFIX.get(key, key)}_Mathematical_Matrix.csv")
            if os.path.exists(csv_path):
                mb = os.path.getsize(csv_path) / 1e6
                log(f"  {key:<11}   reading descriptors ({mb:.1f} MB) ...")
                ids, mat, cols = read_descriptor_csv(csv_path)
                if ids is not None:
                    dim = mat.shape[1]
                    # (class, id) is the only key that is unique on every dataset
                    by_pair, by_rel, by_stem = {}, {}, {}
                    for i, (dx, stem) in enumerate(ids):
                        if dx is not None:
                            by_pair[(dx, stem)] = i
                        by_rel[stem] = i
                        by_stem.setdefault(stem.rsplit("/", 1)[-1], i)
                    n_pair = n_fallback = 0
                    for n_, p in enumerate(paths):
                        rel = os.path.relpath(p, root).replace(os.sep, "/").rsplit(".", 1)[0]
                        bare = rel.rsplit("/", 1)[-1]
                        j = by_pair.get((classes[labels[n_]], bare))
                        if j is not None:
                            n_pair += 1
                        else:
                            j = by_rel.get(rel)
                            if j is None:
                                j = by_stem.get(bare)
                            if j is not None:
                                n_fallback += 1
                        if j is not None:
                            desc_row[n_] = j
                            n_matched += 1
                    np.save(os.path.join(INDEX_DIR, f"{key}_desc.npy"), mat)
                    uniq = len({d for d in desc_row if d >= 0})
                    log(f"  {key:<11}   descriptor matrix {mat.shape} saved | "
                        f"keyed by (class,id) {n_pair}, by id {n_fallback} | "
                        f"{uniq} distinct rows used")
                    if uniq < n_matched:
                        log(f"  {key:<11}   WARNING: {n_matched - uniq} images share a "
                            f"descriptor row -- ids are ambiguous", "WARN")
                else:
                    log(f"  {key:<11}   descriptor CSV unusable, continuing without it", "WARN")
            else:
                log(f"  {key:<11}   no descriptor CSV at {csv_path}", "WARN")

            meta = {
                "domain": key, "root": root, "n_images": len(paths),
                "classes": classes, "n_classes": len(classes),
                "duplicates_remaining": dup,
                "n_matched": n_matched, "descriptor_dim": dim,
                "paths": paths, "labels": labels,
                "desc_row": desc_row, "descriptor_key": "(class, image_id)",
                "built": datetime.now().isoformat(timespec="seconds"),
            }
            atomic_write_json(idx_path, meta)
            invalidate_downstream(key)
            cov = 100.0 * n_matched / max(len(paths), 1)
            log(f"  {key:<11} OK  {len(paths)} images | {len(classes)} classes | "
                f"descriptors {n_matched}/{len(paths)} ({cov:.1f}%) | duplicates {dup}")
            summary[key] = meta
        except Exception:
            log(f"  {key:<11} FAILED:\n{traceback.format_exc()}", "ERROR")
            continue

    atomic_write_json(os.path.join(OUT_DIR, "repair_report.json"),
                      {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("paths", "labels", "desc_row")}
                       for k, v in summary.items()})
    log("STAGE 1 complete.")


def get_index(key: str):
    meta = load_json(os.path.join(INDEX_DIR, f"{key}.json"))
    if meta is None:
        return None
    dpath = os.path.join(INDEX_DIR, f"{key}_desc.npy")
    mat = np.load(dpath) if os.path.exists(dpath) else None
    y = np.asarray(meta["labels"], dtype=np.int64)
    rows = np.asarray(meta["desc_row"], dtype=np.int64)
    if mat is None:
        M = np.zeros((len(y), 0), dtype=np.float32)
    else:
        M = np.zeros((len(y), mat.shape[1]), dtype=np.float32)
        ok = rows >= 0
        M[ok] = mat[rows[ok]]
    return meta, y, M


def folds_for(key: str):
    from sklearn.model_selection import StratifiedKFold
    meta, y, _ = get_index(key)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y)), meta, y


# ============================================================================ #
#  STAGE 2 -- TRAIN HEADS
# ============================================================================ #

def build_model(arch: str, n_classes: int, pretrained: bool):
    import timm
    import torch.nn as nn
    from torchvision import models
    if arch == "ViT":
        return timm.create_model("vit_small_patch16_224", pretrained=pretrained,
                                 num_classes=n_classes, drop_rate=DROP_RATE)
    if arch == "ConvNeXt":
        return timm.create_model("convnext_tiny", pretrained=pretrained,
                                 num_classes=n_classes, drop_rate=DROP_RATE)
    m = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


def head_params(model, arch: str):
    if arch == "ViT":
        return model.head.parameters()
    if arch == "ConvNeXt":
        return model.head.fc.parameters()
    return model.classifier[3].parameters()


def weight_path(arch: str, key: str, fold: int) -> str:
    """
    Full mode  -> new weights trained with dropout, so MC dropout is genuine.
    Quick mode -> reuse the weights already on disk, except for CHEST-XRAY whose
                  folds changed when the duplicate tree was quarantined.
    """
    if QUICK_MODE and key != "CHEST-XRAY":
        for cand in (f"{arch}_{key}_Fold{fold}.pth",
                     f"{arch}_model_{key}_fold{fold}.pth"):
            p = os.path.join(WEIGHTS_DIR, cand)
            if os.path.exists(p):
                return p
    tag = f"_dr{int(DROP_RATE*100)}" if DROP_RATE > 0 else ""
    return os.path.join(WEIGHTS_DIR, f"{arch}_{key}_Fold{fold}{tag}.pth")


def stage_train(only_domains=None, force: bool = False) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms

    log("=" * 78)
    log("STAGE 2/5  TRAIN CLASSIFICATION HEADS  (backbone frozen)")
    log("=" * 78)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  device: {dev.type.upper()}"
        + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    tf_train = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    todo = []
    if QUICK_MODE and only_domains is None:
        log("  quick mode: only CHEST-XRAY is retrained (its folds changed after repair)")
    for key in (only_domains or DOMAINS):
        if get_index(key) is None:
            continue
        for arch in ARCHS:
            for fold in range(1, N_FOLDS + 1):
                if force or not done(weight_path(arch, key, fold)):
                    todo.append((key, arch, fold))
    log(f"  {len(todo)} head(s) to train "
        f"({len(DOMAINS) * len(ARCHS) * N_FOLDS - len(todo)} already on disk)")
    if not todo:
        log("STAGE 2 complete (nothing to do).")
        return

    t0, per = time.time(), []
    for n, (key, arch, fold) in enumerate(todo, 1):
        splits, meta, y = folds_for(key)
        tr, va = splits[fold - 1]
        paths, K = meta["paths"], meta["n_classes"]
        ts = time.time()
        log(f"  [{n}/{len(todo)}] {key} | {arch} | fold {fold} | {K} classes")

        # Checkpoint selection must NOT see the outer validation fold, or the
        # reported numbers are selected on the data they are reported on. We
        # carve an inner selection split out of the training fold instead, and
        # `va` is never touched until stage 3.
        from sklearn.model_selection import train_test_split as _tts
        try:
            tr_in, tr_sel = _tts(tr, test_size=SEL_FRACTION, random_state=SEED,
                                 stratify=y[tr])
        except ValueError:                       # a class too small to stratify
            tr_in, tr_sel = _tts(tr, test_size=SEL_FRACTION, random_state=SEED)

        ds_tr = ImageSet([paths[i] for i in tr_in], y[tr_in], tf_train)
        ds_se = ImageSet([paths[i] for i in tr_sel], y[tr_sel], tf_eval)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0)
        dl_se = DataLoader(ds_se, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)

        model = build_model(arch, K, pretrained=True).to(dev)
        for p in model.parameters():
            p.requires_grad = False
        for p in head_params(model, arch):
            p.requires_grad = True
        crit = nn.CrossEntropyLoss()
        opt = optim.AdamW(head_params(model, arch), lr=LR, weight_decay=1e-4)

        best, best_state = -1.0, None
        for ep in range(1, EPOCHS + 1):
            model.train()
            tot = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * xb.size(0)
            model.eval()
            corr = 0
            with torch.no_grad():
                for xb, yb in dl_se:
                    corr += (model(xb.to(dev)).argmax(1).cpu() == yb).sum().item()
            acc = corr / max(len(tr_sel), 1)
            log(f"      epoch {ep:>2}/{EPOCHS}  loss {tot/max(len(tr_in),1):.4f}  sel_acc {acc:.4f}"
                + ("  <- best" if acc > best else ""))
            if acc > best:
                best = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        wp = weight_path(arch, key, fold)
        tmp = wp + ".tmp"
        torch.save(best_state, tmp)
        os.replace(tmp, wp)
        atomic_write_json(wp + ".meta.json",
                          {"domain": key, "arch": arch, "fold": fold, "n_classes": K,
                           "best_inner_sel_acc": best, "drop_rate": DROP_RATE,
                           "n_train_inner": int(len(tr_in)), "n_select": int(len(tr_sel)),
                           "n_outer_val": int(len(va)),
                           "checkpoint_selected_on": "inner split of training fold",
                           "epochs": EPOCHS, "seed": SEED,
                           "finished": datetime.now().isoformat(timespec="seconds")})
        per.append(time.time() - ts)
        eta = float(np.mean(per)) * (len(todo) - n)
        log(f"      saved (best inner-selection acc {best:.4f}) | elapsed {fmt_eta(time.time()-t0)} | ETA {fmt_eta(eta)}")
        del model, best_state
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log("STAGE 2 complete.")


class ImageSet:
    """Torch Dataset defined lazily so the module imports without torch."""
    def __init__(self, paths, labels, tf):
        import torch
        self._t = torch
        self.paths, self.labels, self.tf = paths, np.asarray(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        x = self.tf(img) if self.tf else img
        return x, self._t.tensor(int(self.labels[i]), dtype=self._t.long)


# ============================================================================ #
#  STAGE 3 -- EXTRACT AND CACHE
# ============================================================================ #

def feat_path(key, arch, fold):
    return os.path.join(FEAT_DIR, f"{key}__{arch}__f{fold}.npz")


def stage_extract(force: bool = False) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms, models

    log("=" * 78)
    log("STAGE 3/5  EXTRACT AND CACHE FEATURES")
    log("=" * 78)
    os.makedirs(FEAT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    todo = [(k, a, f) for k in DOMAINS if get_index(k) is not None
            for a in ARCHS for f in range(1, N_FOLDS + 1)
            if (force or not done(feat_path(k, a, f))) and done(weight_path(a, k, f))]
    log(f"  {len(todo)} extraction job(s) pending")

    t0, per = time.time(), []
    for n, (key, arch, fold) in enumerate(todo, 1):
        ts = time.time()
        splits, meta, y = folds_for(key)
        tr, va = splits[fold - 1]
        paths, K = meta["paths"], meta["n_classes"]
        order = np.concatenate([tr, va])
        log(f"  [{n}/{len(todo)}] {key} | {arch} | fold {fold} | {len(order)} images")

        model = build_model(arch, K, pretrained=False)
        state = torch.load(weight_path(arch, key, fold), map_location="cpu")
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
        model = model.to(dev).eval()

        dl = DataLoader(ImageSet([paths[i] for i in order], y[order], tf),
                        batch_size=BATCH_EVAL, shuffle=False, num_workers=0)

        Z, L, P, MC = [], [], [], []
        is_mnet = isinstance(model, models.MobileNetV3)
        with torch.no_grad():
            for bi, (xb, _) in enumerate(dl):
                xb = xb.to(dev)
                model.eval()
                if is_mnet:
                    fmap = model.avgpool(model.features(xb)).flatten(1)
                    lat = model.classifier[1](model.classifier[0](fmap))
                    lg = model.classifier(fmap)
                else:
                    fe = model.forward_features(xb)
                    lat = model.forward_head(fe, pre_logits=True)
                    lg = model.forward_head(fe)
                Z.append(lat.float().cpu().numpy().astype(np.float16))
                L.append(lg.float().cpu().numpy().astype(np.float32))
                P.append(torch.softmax(lg, 1).float().cpu().numpy().astype(np.float32))

                # genuine MC dropout: put dropout layers (only) into train mode
                model.apply(lambda m: m.train() if isinstance(m, torch.nn.Dropout) else None)
                acc = None
                for _ in range(MC_SAMPLES):
                    if is_mnet:
                        lgm = model.classifier(model.avgpool(model.features(xb)).flatten(1))
                    else:
                        lgm = model.forward_head(model.forward_features(xb))
                    s = torch.softmax(lgm, 1).float()
                    acc = s if acc is None else acc + s
                MC.append((acc / MC_SAMPLES).cpu().numpy().astype(np.float32))
                model.eval()
                if bi % 20 == 0 and bi:
                    log(f"      batch {bi}/{len(dl)}")

        Z, L, P, MC = np.vstack(Z), np.vstack(L), np.vstack(P), np.vstack(MC)
        mc_live = float(np.abs(P - MC).max())
        _, _, Mall = get_index(key)
        atomic_savez(feat_path(key, arch, fold),
                     Z=Z, logits=L, probs=P, mc=MC, M=Mall[order].astype(np.float32),
                     y=y[order].astype(np.int64),
                     n_train=np.array([len(tr)]), n_classes=np.array([K]),
                     mc_live=np.array([mc_live]))
        per.append(time.time() - ts)
        log(f"      cached | MC deviation {mc_live:.5f} "
            f"({'active' if mc_live > 1e-6 else 'INACTIVE -- dropout not firing'}) | "
            f"ETA {fmt_eta(float(np.mean(per))*(len(todo)-n))}")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log("STAGE 3 complete.")


# ============================================================================ #
#  METRICS
# ============================================================================ #

def evaluate_all(y, p, K):
    """The fifteen-metric audit, plus AURC and the E-AURC excess measure."""
    from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score, log_loss)
    p = np.clip(np.nan_to_num(p, nan=1.0 / K), 1e-15, 1 - 1e-15)
    p = p / p.sum(1, keepdims=True)
    pred = p.argmax(1)
    conf = p.max(1)
    corr = (pred == y).astype(float)
    onehot = np.eye(K)[y]

    edges = np.linspace(0, 1, N_BINS + 1)
    ece = mce = dece = sce = 0.0
    for i in range(N_BINS):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        w = m.mean()
        if w > 0:
            gap = abs(conf[m].mean() - corr[m].mean())
            ece += gap * w
            mce = max(mce, gap)
            nk = int(m.sum())
            if nk > 1:
                a = corr[m].mean()
                bias = np.sqrt(max(a * (1 - a), 0) / (nk - 1))
                dece += max(0.0, gap - bias) * (nk / len(corr))
    for c in range(K):
        pc, ac = p[:, c], (y == c).astype(float)
        for i in range(N_BINS):
            m = (pc > edges[i]) & (pc <= edges[i + 1])
            w = m.mean()
            if w > 0:
                sce += abs(pc[m].mean() - ac[m].mean()) * w
    sce /= K

    brier = float(np.mean(np.sum((p - onehot) ** 2, 1)))
    bref = float(np.mean(np.sum((1.0 / K - onehot) ** 2, 1))) + 1e-15
    try:
        auc = float(roc_auc_score(y, p if K > 2 else p[:, 1],
                                  average="macro", multi_class="ovr" if K > 2 else "raise"))
    except Exception:
        auc = float("nan")

    # risk-coverage / AURC -- the selective-prediction curve.
    #
    # Confidence ties must be handled explicitly, not left to argsort. Isotonic
    # regression is a step function: on some folds 82% of its predictions carry
    # one identical confidence value, and the order of those samples is
    # arbitrary. Under a plain argsort the resulting AURC moves by a factor of
    # two depending on the order rows happen to sit in memory, which makes the
    # number unreproducible. We report instead the expectation over tie-breaking:
    # inside a run of equal confidence, every position takes that run's mean
    # error. That is what a uniformly random ordering yields on average, it is
    # independent of input order, and it is identical to the naive value when
    # confidences are distinct -- which they effectively are for every method
    # here except isotonic (tied fraction 0.71 against at most 0.03).
    o = np.argsort(-conf, kind="stable")
    err = 1.0 - corr[o]
    e = float(err.mean())
    # the oracle ranking is computed from the true errors, before any smoothing
    opt = float(np.mean(np.cumsum(np.sort(err)) / np.arange(1, len(err) + 1)))
    cs = conf[o]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], len(cs)]
    for _s, _t in zip(starts, ends):
        if _t - _s > 1:
            err[_s:_t] = err[_s:_t].mean()
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    aurc = float(risk.mean())
    cov90 = float(risk[max(int(0.90 * len(risk)) - 1, 0)])
    cov80 = float(risk[max(int(0.80 * len(risk)) - 1, 0)])

    return {
        "BAcc": float(balanced_accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, average="macro")),
        "AUC": auc,
        "ECE": float(ece), "dECE": float(dece), "SCE": float(sce),
        "Brier": brier, "BSS": float(1 - brier / bref),
        "NLL": float(log_loss(y, p, labels=list(range(K)))),
        "Conf": float(conf.mean()),
        "Ent": float(np.mean(-np.sum(p * np.log(p), 1))),
        "MTP": float(np.mean(p[np.arange(len(y)), y])),
        "MCE": float(mce),
        "OCE": float(conf[pred != y].mean()) if np.any(pred != y) else 0.0,
        "CCC": float(conf[pred == y].mean()) if np.any(pred == y) else 0.0,
        "AURC": aurc, "EAURC": float(aurc - opt),
        "Risk@90": cov90, "Risk@80": cov80, "Err": e,
    }


# ============================================================================ #
#  APEX
# ============================================================================ #

class Apex:
    """
    APEX -- the single published configuration.

        p_hat = th_b(p_base) * p_base + th_r * p_rbf + th_q * p_poly + th_k * p_knn
        th_b(p) = th_b*        if max(p) >= tau
                  lam * th_b*  otherwise, freed mass split across the experts

    Experts are fitted on D_fit; the weights th* are fitted on the disjoint
    D_cal by minimising the negative log-likelihood.

    The objective is deliberately a strictly proper scoring rule. An earlier
    version fitted th* with a class-balanced focal loss; because focal loss is
    not proper, its minimum does not sit at the honest probabilities, and it
    cost 0.016 ECE (p=0.0002, 16/18 configurations) for no measurable accuracy
    in return. `use_focal=True` reproduces that variant for the ablation.
    """

    def __init__(self, K, use_desc=True, experts=("rbf", "poly", "knn"),
                 use_focal=False, tau=TAU, lam=LAMBDA):
        self.K, self.use_desc, self.experts = K, use_desc, tuple(experts)
        self.use_focal, self.tau, self.lam = use_focal, tau, lam
        self.scaler = self.proj = None
        self.clf = {}
        self.theta = None

    # -- manifold -------------------------------------------------------------
    def _project(self, W, fit=False):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import TruncatedSVD
        if fit:
            self.scaler = StandardScaler().fit(W)
            Ws = self.scaler.transform(W)
            d = int(min(TARGET_DIM, max(2, min(Ws.shape) - 1)))
            self.proj = TruncatedSVD(n_components=d, random_state=SEED).fit(Ws)
            return self.proj.transform(Ws)
        return self.proj.transform(self.scaler.transform(W))

    def _stack(self, Z, M):
        return np.hstack([Z, M]) if (self.use_desc and M.size) else Z

    def _pad(self, clf, p):
        if p.shape[1] == self.K:
            return p
        full = np.zeros((p.shape[0], self.K))
        for i, c in enumerate(clf.classes_):
            full[:, int(c)] = p[:, i]
        return full

    # -- fit ------------------------------------------------------------------
    def fit_experts(self, Z, M, y):
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        U = self._project(self._stack(Z, M), fit=True)
        if MAX_FIT_SAMPLES and len(y) > MAX_FIT_SAMPLES:
            rs = np.random.RandomState(SEED)
            keep = np.concatenate([rs.choice(np.where(y == c)[0],
                                             max(1, int(MAX_FIT_SAMPLES * (y == c).mean())),
                                             replace=False) for c in np.unique(y)])
            U, y = U[keep], y[keep]
        if "rbf" in self.experts:
            self.clf["rbf"] = SVC(kernel="rbf", C=1.0, gamma="scale",
                                  probability=True, random_state=SEED).fit(U, y)
        if "poly" in self.experts:
            self.clf["poly"] = SVC(kernel="poly", degree=3, C=1.0,
                                   probability=True, random_state=SEED).fit(U, y)
        if "knn" in self.experts:
            k = int(min(5, max(1, len(y) - 1)))
            self.clf["knn"] = KNeighborsClassifier(n_neighbors=k, weights="distance",
                                                   n_jobs=-1).fit(U, y)
        return self

    def expert_probs(self, Z, M):
        U = self._project(self._stack(Z, M))
        out = {}
        for name in ("rbf", "poly", "knn"):
            if name in self.clf:
                out[name] = self._pad(self.clf[name], self.clf[name].predict_proba(U))
            else:
                out[name] = None
        return out

    def fit_weights(self, Z, M, p_base, y):
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss
        E = self.expert_probs(Z, M)
        active = [n for n in ("rbf", "poly", "knn") if E[n] is not None]
        oh = np.eye(self.K)[y]
        cnt = oh.sum(0)
        w = 1.0 / (cnt + 1e-5)
        w = w / w.sum() * self.K

        def obj(t):
            f = t[0] * p_base + sum(t[i + 1] * E[n] for i, n in enumerate(active))
            f = np.clip(f, 1e-15, 1 - 1e-15)
            f /= f.sum(1, keepdims=True)
            if self.use_focal:
                pt = np.sum(f * oh, 1)
                return float(np.mean(w[y] * (1 - pt) ** GAMMA * -np.log(pt)))
            return float(log_loss(y, f, labels=list(range(self.K))))

        n = len(active) + 1
        x0 = np.full(n, 1.0 / n)
        res = minimize(obj, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                       constraints=({"type": "eq", "fun": lambda t: 1.0 - t.sum()},),
                       options={"maxiter": 300, "ftol": 1e-9})
        t = np.clip(res.x, 0, 1)
        t = t / max(t.sum(), 1e-12)
        self.theta = {"base": float(t[0])}
        self.theta.update({n_: float(t[i + 1]) for i, n_ in enumerate(active)})
        return self

    # -- predict --------------------------------------------------------------
    def predict_proba(self, Z, M, p_base, gated=True, tau=None):
        tau = self.tau if tau is None else tau
        E = self.expert_probs(Z, M)
        active = [n for n in ("rbf", "poly", "knn") if E[n] is not None]
        tb0 = self.theta["base"]
        te = np.array([self.theta[n] for n in active])
        stack = np.stack([E[n] for n in active], 0)                # (A, N, K)

        if not gated:
            out = tb0 * p_base + np.tensordot(te, stack, axes=(0, 0))
        else:
            low = p_base.max(1) < tau
            tb = np.where(low, self.lam * tb0, tb0)                # (N,)
            freed = tb0 - tb                                       # (N,)
            te_i = te[:, None] + (freed / max(len(active), 1))[None, :]
            out = tb[:, None] * p_base + np.einsum("an,ank->nk", te_i, stack)

        out = np.clip(out, 1e-15, 1 - 1e-15)
        return out / out.sum(1, keepdims=True)


class Scalers:
    """Post-hoc scaling baselines, fitted on D_cal."""
    def __init__(self, K):
        self.K = K
        self.T = 1.0
        self.w = np.ones(K)
        self.b = np.zeros(K)
        self.iso = []
        self.dir = None

    @staticmethod
    def _sm(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return np.clip(e / e.sum(1, keepdims=True), 1e-15, 1 - 1e-15)

    def fit(self, logits, probs, y):
        from scipy.optimize import minimize
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss
        lab = list(range(self.K))
        self.T = float(minimize(lambda t: log_loss(y, self._sm(logits / t[0]), labels=lab),
                                x0=[1.0], bounds=[(0.01, 10.0)]).x[0])
        r = minimize(lambda v: log_loss(y, self._sm(logits * v[:self.K] + v[self.K:]), labels=lab),
                     x0=np.concatenate([np.ones(self.K), np.zeros(self.K)]), method="L-BFGS-B")
        self.w, self.b = r.x[:self.K], r.x[self.K:]
        self.iso = []
        for c in range(self.K):
            m = IsotonicRegression(out_of_bounds="clip")
            m.fit(probs[:, c], (y == c).astype(int))
            self.iso.append(m)
        self.dir = LogisticRegression(max_iter=1000).fit(np.log(np.clip(probs, 1e-15, 1)), y)
        return self

    def temperature(self, logits):
        return self._sm(logits / self.T)

    def vector(self, logits):
        return self._sm(logits * self.w + self.b)

    def isotonic(self, probs):
        out = np.column_stack([m.predict(probs[:, c]) for c, m in enumerate(self.iso)])
        out = np.clip(out, 1e-15, None)
        return out / out.sum(1, keepdims=True)

    def dirichlet(self, probs):
        p = self.dir.predict_proba(np.log(np.clip(probs, 1e-15, 1)))
        if p.shape[1] == self.K:
            return p
        full = np.zeros((p.shape[0], self.K))
        for i, c in enumerate(self.dir.classes_):
            full[:, int(c)] = p[:, i]
        return np.clip(full, 1e-15, 1 - 1e-15)


# ============================================================================ #
#  STAGE 4 -- BENCHMARK
# ============================================================================ #

def cell_path(key, arch, fold):
    return os.path.join(CELL_DIR, f"{key}__{arch}__f{fold}.json")


def stage_bench(force: bool = False) -> None:
    from sklearn.model_selection import train_test_split

    log("=" * 78)
    log("STAGE 4/5  BENCHMARK, ABLATIONS, FAIR BASELINES, TAU SWEEP")
    log("=" * 78)
    os.makedirs(CELL_DIR, exist_ok=True)
    os.makedirs(PROB_DIR, exist_ok=True)

    todo = [(k, a, f) for k in DOMAINS for a in ARCHS for f in range(1, N_FOLDS + 1)
            if done(feat_path(k, a, f)) and (force or not done(cell_path(k, a, f)))]
    log(f"  {len(todo)} cell(s) pending")

    t0, per = time.time(), []
    for n, (key, arch, fold) in enumerate(todo, 1):
        ts = time.time()
        log(f"  [{n}/{len(todo)}] {key} | {arch} | fold {fold}")
        try:
            d = np.load(feat_path(key, arch, fold))
            Z, L, P, MC = d["Z"].astype(np.float32), d["logits"], d["probs"], d["mc"]
            M, y = d["M"], d["y"]
            ntr, K = int(d["n_train"][0]), int(d["n_classes"][0])
            mc_live = float(d["mc_live"][0])

            Ztr, Zva = Z[:ntr], Z[ntr:]
            Ltr, Lva = L[:ntr], L[ntr:]
            Ptr, Pva = P[:ntr], P[ntr:]
            Mtr, Mva = M[:ntr], M[ntr:]
            ytr, yva = y[:ntr], y[ntr:]

            idx_f, idx_c = train_test_split(np.arange(ntr), test_size=0.2,
                                            random_state=SEED, stratify=ytr)
            Zf, Zc = Ztr[idx_f], Ztr[idx_c]
            Mf, Mc = Mtr[idx_f], Mtr[idx_c]
            Pc = Ptr[idx_c]
            Lc = Ltr[idx_c]
            yf, yc = ytr[idx_f], ytr[idx_c]

            res, probs_out = {}, {"y_true": yva}

            # ---- baselines ---------------------------------------------------
            res["BASE"] = evaluate_all(yva, Pva, K)
            probs_out["BASE"] = Pva
            res["MC_DROPOUT"] = evaluate_all(yva, MC[ntr:], K)
            probs_out["MC_DROPOUT"] = MC[ntr:]

            sc = Scalers(K).fit(Lc, Pc, yc)
            for nm, pr in (("TEMP_SCALE", sc.temperature(Lva)),
                           ("VEC_SCALE", sc.vector(Lva)),
                           ("ISOTONIC", sc.isotonic(Pva)),
                           ("DIRICHLET", sc.dirichlet(Pva))):
                res[nm] = evaluate_all(yva, pr, K)
                probs_out[nm] = pr

            # ---- APEX (the published configuration) --------------------------
            log("      fitting kernel experts on [Z;M] ...")
            ap = Apex(K, use_focal=False).fit_experts(Zf, Mf, yf).fit_weights(Zc, Mc, Pc, yc)
            p_apex = ap.predict_proba(Zva, Mva, Pva, gated=True)
            res["APEX"] = evaluate_all(yva, p_apex, K)
            probs_out["APEX"] = p_apex
            res["APEX"]["theta"] = ap.theta

            # ---- ablations that reuse the same fitted experts (free) ---------
            res["ABL_no_gate"] = evaluate_all(yva, ap.predict_proba(Zva, Mva, Pva, gated=False), K)
            for drop in ("rbf", "poly", "knn"):
                sub = tuple(e for e in ("rbf", "poly", "knn") if e != drop)
                a2 = Apex(K, experts=sub, use_focal=False)
                a2.scaler, a2.proj, a2.clf = ap.scaler, ap.proj, {k_: ap.clf[k_] for k_ in sub}
                a2.fit_weights(Zc, Mc, Pc, yc)
                res[f"ABL_no_{drop}"] = evaluate_all(
                    yva, a2.predict_proba(Zva, Mva, Pva, gated=True), K)

            # ablation: refit the same experts under the focal objective
            a_focal = Apex(K, use_focal=True)
            a_focal.scaler, a_focal.proj, a_focal.clf = ap.scaler, ap.proj, ap.clf
            a_focal.fit_weights(Zc, Mc, Pc, yc)
            res["ABL_focal"] = evaluate_all(
                yva, a_focal.predict_proba(Zva, Mva, Pva, gated=True), K)

            # ---- tau sweep (free) --------------------------------------------
            for tv in TAU_SWEEP:
                res[f"TAU_{tv:.2f}"] = evaluate_all(
                    yva, ap.predict_proba(Zva, Mva, Pva, gated=True, tau=tv), K)

            # ---- fair-comparison rows: the experts on their own ---------------
            E = ap.expert_probs(Zva, Mva)
            for nm, pr in (("FAIR_rbf_svm", E["rbf"]), ("FAIR_poly_svm", E["poly"]),
                           ("FAIR_knn", E["knn"])):
                if pr is not None:
                    res[nm] = evaluate_all(yva, pr, K)
                    probs_out[nm] = pr
            eq = np.mean(np.stack([E[e] for e in ("rbf", "poly", "knn")], 0), 0)
            res["FAIR_equal_blend"] = evaluate_all(yva, eq, K)
            probs_out["FAIR_equal_blend"] = eq

            # ---- ablation: latent only, no geometric descriptors --------------
            if M.shape[1] > 0:
                log("      fitting kernel experts on [Z] only (descriptor ablation) ...")
                apZ = Apex(K, use_desc=False, use_focal=False).fit_experts(
                    Zf, Mf, yf).fit_weights(Zc, Mc, Pc, yc)
                pz = apZ.predict_proba(Zva, Mva, Pva, gated=True)
                res["ABL_no_descriptors"] = evaluate_all(yva, pz, K)
                probs_out["ABL_no_descriptors"] = pz

            payload = {
                "domain": key, "arch": arch, "fold": fold, "n_classes": K,
                "n_val": int(len(yva)), "n_fit": int(len(yf)), "n_cal": int(len(yc)),
                "mc_dropout_active": bool(mc_live > 1e-6), "mc_deviation": mc_live,
                "tau": TAU, "lambda": LAMBDA, "seed": SEED,
                "metrics": res,
                "finished": datetime.now().isoformat(timespec="seconds"),
            }
            atomic_write_json(cell_path(key, arch, fold), payload)
            atomic_savez(os.path.join(PROB_DIR, f"{key}__{arch}__f{fold}.npz"), **probs_out)

            per.append(time.time() - ts)
            log(f"      BASE {res['BASE']['BAcc']:.4f} -> APEX {res['APEX']['BAcc']:.4f} "
                f"| ECE {res['BASE']['ECE']:.4f} -> {res['APEX']['ECE']:.4f} "
                f"| AURC {res['BASE']['AURC']:.4f} -> {res['APEX']['AURC']:.4f} "
                f"| ETA {fmt_eta(float(np.mean(per))*(len(todo)-n))}")
        except Exception:
            log(f"      FAILED: {traceback.format_exc(limit=3)}", "ERROR")
            continue
    log("STAGE 4 complete.")


# ============================================================================ #
#  STAGE 5 -- ANALYSE
# ============================================================================ #

METRIC_ORDER = ["BAcc", "F1", "AUC", "ECE", "dECE", "SCE", "Brier", "BSS",
                "NLL", "Conf", "Ent", "MTP", "MCE", "OCE", "CCC",
                "AURC", "EAURC", "Risk@90", "Risk@80"]
LOWER_BETTER = {"ECE", "dECE", "SCE", "Brier", "NLL", "MCE", "OCE",
                "AURC", "EAURC", "Risk@90", "Risk@80", "Err"}


def _mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _std(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def stage_analyse() -> None:
    from scipy.stats import wilcoxon

    log("=" * 78)
    log("STAGE 5/5  ANALYSE")
    log("=" * 78)

    rows = []
    if os.path.isdir(CELL_DIR):
        for fn in sorted(os.listdir(CELL_DIR)):
            if not fn.endswith(".json"):
                continue
            c = load_json(os.path.join(CELL_DIR, fn))
            if not c:
                continue
            for meth, m in c["metrics"].items():
                r = {"domain": c["domain"], "arch": c["arch"], "fold": c["fold"],
                     "method": meth, "mc_active": c.get("mc_dropout_active", False)}
                for k in METRIC_ORDER:
                    r[k] = m.get(k)
                rows.append(r)
    if not rows:
        log("  no completed cells -- run the earlier stages first", "WARN")
        return

    # ---- per-fold CSV -------------------------------------------------------
    hdr = ["domain", "arch", "fold", "method", "mc_active"] + METRIC_ORDER
    p_fold = os.path.join(OUT_DIR, "FINAL_RESULTS_perfold.csv")
    with open(p_fold, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in rows:
            w.writerow([r.get(k, "") for k in hdr])
    log(f"  {len(rows)} per-fold rows -> FINAL_RESULTS_perfold.csv")

    # ---- collapse folds -> one value per (domain, arch, method) -------------
    buckets = {}
    for r in rows:
        buckets.setdefault((r["domain"], r["arch"], r["method"]), []).append(r)
    cfg = {}
    for k, rs in buckets.items():
        cfg[k] = {m: _mean([x[m] for x in rs]) for m in METRIC_ORDER}
        cfg[k]["_fold_sd"] = {m: _std([x[m] for x in rs]) for m in METRIC_ORDER}

    p_cfg = os.path.join(OUT_DIR, "FINAL_RESULTS_perconfig.csv")
    with open(p_cfg, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "arch", "method"] + METRIC_ORDER
                   + [m + "_sd" for m in METRIC_ORDER])
        for (d, a, meth), v in sorted(cfg.items()):
            w.writerow([d, a, meth] + [v[m] for m in METRIC_ORDER]
                       + [v["_fold_sd"][m] for m in METRIC_ORDER])

    methods = sorted({k[2] for k in cfg})
    configs = sorted({(k[0], k[1]) for k in cfg})
    log(f"  {len({c[0] for c in configs})} datasets x {len({c[1] for c in configs})} "
        f"backbones = {len(configs)} configurations per method")

    def series(method, metric):
        """Paired vector over configurations, aligned across methods."""
        return np.array([cfg.get((d, a, method), {}).get(metric, np.nan)
                         for d, a in configs], dtype=float)

    summary = {"generated": datetime.now().isoformat(timespec="seconds"),
               "n_configurations": len(configs), "tau": TAU, "lambda": LAMBDA,
               "mean": {}, "sd_across_configs": {}, "sd_across_folds": {}}
    for meth in methods:
        summary["mean"][meth] = {m: _mean(series(meth, m).tolist()) for m in METRIC_ORDER}
        summary["sd_across_configs"][meth] = {m: _std(series(meth, m).tolist())
                                              for m in METRIC_ORDER}
        summary["sd_across_folds"][meth] = {
            m: _mean([cfg[k]["_fold_sd"][m] for k in cfg if k[2] == meth])
            for m in METRIC_ORDER}

    tests = {}
    for base in [m for m in methods if m != "APEX"]:
        tests[base] = {}
        for m in ["BAcc", "ECE", "Brier", "NLL", "AURC"]:
            try:
                a_, b_ = series("APEX", m), series(base, m)
                ok = np.isfinite(a_) & np.isfinite(b_)
                a_, b_ = a_[ok], b_[ok]
                if len(a_) < 3 or np.allclose(a_, b_):
                    tests[base][m] = {"p": 1.0, "wins": 0, "n": int(len(a_)),
                                      "note": "identical or too few pairs"}
                    continue
                p = float(wilcoxon(a_, b_).pvalue)
                wins = int((a_ < b_).sum() if m in LOWER_BETTER else (a_ > b_).sum())
                tests[base][m] = {"p": p, "wins": wins, "n": int(len(a_)),
                                  "median_delta": float(np.median(a_ - b_))}
            except Exception as e:
                tests[base][m] = {"error": str(e)}
    summary["wilcoxon_vs_APEX"] = tests
    atomic_write_json(os.path.join(OUT_DIR, "summary.json"), summary)

    # ---- console report -----------------------------------------------------
    core = ["BASE", "MC_DROPOUT", "TEMP_SCALE", "VEC_SCALE", "ISOTONIC", "DIRICHLET", "APEX"]
    ap = summary["mean"].get("APEX", {})
    log("")
    log("  HEADLINE  (mean over configurations)")
    log(f"  {'method':<16}{'BAcc':>9}{'ECE':>9}{'Brier':>9}{'NLL':>9}{'AURC':>9}")
    for m in core:
        if m in summary["mean"]:
            s_ = summary["mean"][m]
            log(f"  {m:<16}{s_['BAcc']:>9.4f}{s_['ECE']:>9.4f}{s_['Brier']:>9.4f}"
                f"{s_['NLL']:>9.4f}{s_['AURC']:>9.4f}")

    for title, prefix in (("ABLATIONS  (does each component earn its place?)", "ABL_"),
                          ("FAIR COMPARISON  (experts alone, no fusion, no gate)", "FAIR_"),
                          ("TAU SWEEP", "TAU_")):
        sel = sorted(k for k in summary["mean"] if k.startswith(prefix))
        if not sel:
            continue
        log("")
        log(f"  {title}")
        for m in sel:
            s_ = summary["mean"][m]
            log(f"  {m:<24} BAcc {s_['BAcc']:.4f} ({s_['BAcc']-ap.get('BAcc',0):+.4f})  "
                f"ECE {s_['ECE']:.4f} ({s_['ECE']-ap.get('ECE',0):+.4f})  "
                f"AURC {s_['AURC']:.4f} ({s_['AURC']-ap.get('AURC',0):+.4f})")

    log("")
    log("  APEX vs baselines  (Wilcoxon signed-rank over configurations)")
    for b in core[:-1]:
        if b in tests:
            t = tests[b]

            def cell(k):
                d = t.get(k, {})
                p = d.get("p", float("nan"))
                ps = "p<1e-4" if p < 1e-4 else f"p={p:.4f}"
                return f"{d.get('wins','?')}/{d.get('n','?')} {ps}"

            log(f"  vs {b:<14} BAcc {cell('BAcc'):<20} ECE {cell('ECE'):<20} "
                f"AURC {cell('AURC')}")

    mc = [r["mc_active"] for r in rows if r["method"] == "MC_DROPOUT"]
    if mc:
        log("")
        log(f"  MC dropout genuinely active in {int(sum(bool(x) for x in mc))}/{len(mc)} folds")

    _write_latex_table(cfg, summary)
    _write_figures(cfg, rows, summary)
    log("")
    log("STAGE 5 complete.")
    log("  send these back for the paper update:")
    log(f"    {p_fold}")
    log(f"    {os.path.join(OUT_DIR, 'summary.json')}")


def _hdr(name: str) -> str:
    """Column header with the preferred-direction arrow (none for diagnostics)."""
    if name in ("Conf", "Ent"):
        return name
    arrow = r"$\downarrow$" if name in LOWER_BETTER else r"$\uparrow$"
    return name + r"\," + arrow


def _write_latex_table(cfg, summary) -> None:
    show = ["BAcc", "F1", "AUC", "ECE", "dECE", "SCE", "Brier", "BSS", "NLL",
            "Conf", "Ent", "MTP", "MCE", "OCE", "CCC"]
    label = {"BASE": "Uncalibrated", "MC_DROPOUT": "MC dropout",
             "TEMP_SCALE": "Temp.\\ scaling", "VEC_SCALE": "Vector scaling",
             "ISOTONIC": "Isotonic", "DIRICHLET": "Dirichlet",
             "FAIR_rbf_svm": "\\;\\;RBF SVM alone", "FAIR_equal_blend": "\\;\\;equal blend",
             "ABL_no_descriptors": "\\;\\;APEX w/o descriptors",
             "ABL_no_gate": "\\;\\;APEX w/o gate", "APEX": "\\;\\;\\textbf{APEX}"}
    order = list(label)
    os.makedirs(TEX_DIR, exist_ok=True)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Complete fifteen-metric audit, averaged over six medical modalities "
         r"and five folds. All values $\times10^{2}$. Best per column within each backbone "
         r"block is \underline{underlined}.}", r"\label{tab:full}", r"\tiny",
         r"\setlength{\tabcolsep}{3.9pt}", r"\renewcommand{\arraystretch}{1.02}",
         r"\begin{tabular}{@{}l" + "c" * len(show) + r"@{}}", r"\toprule",
         "Method & " + " & ".join(_hdr(s) for s in show) + r" \\"]
    for arch in ARCHS:
        L.append(r"\midrule")
        L.append(r"\multicolumn{" + str(len(show) + 1) + r"}{@{}l}{\textit{" + arch + r"}}\\[1pt]")
        present = {k[2] for k in cfg if k[1] == arch}
        vals = {}
        for m in order:
            if m not in present:
                continue
            vals[m] = [_mean([cfg[k][s] for k in cfg if k[1] == arch and k[2] == m]) * 100
                       for s in show]
        if not vals:
            continue
        best = {}
        for j, s in enumerate(show):
            if s in ("Conf", "Ent"):
                best[j] = None
                continue
            col = {m: vals[m][j] for m in vals}
            best[j] = (min if s in LOWER_BETTER else max)(col, key=col.get)
        for m in vals:
            cells = [(r"\underline{" + f"{vals[m][j]:.1f}" + "}") if best[j] == m
                     else f"{vals[m][j]:.1f}" for j in range(len(show))]
            row = label[m] + " & " + " & ".join(cells) + r" \\"
            if m == "APEX":
                row = r"\rowcolor{apexrow} " + row
            L.append(row)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(TEX_DIR, "table_full.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"  wrote {os.path.join(TEX_DIR, 'table_full.tex')}")


def _write_figures(cfg, rows, summary) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"  matplotlib unavailable, skipping figures: {e}", "WARN")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})

    # risk-coverage from saved probabilities
    try:
        import glob
        series = [("BASE", "Uncalibrated", "#8C8C8C"),
                  ("TEMP_SCALE", "Temp. scaling", "#0072B2"),
                  ("APEX", "APEX", "#D55E00")]
        acc = {}
        for fp in glob.glob(os.path.join(PROB_DIR, "*.npz")):
            d = np.load(fp)
            if "y_true" not in d:
                continue
            for k, _, _ in series:
                if k in d:
                    acc.setdefault(k, {"y": [], "p": []})
                    acc[k]["y"].append(d["y_true"])
                    acc[k]["p"].append(d[k].max(1) * 0 + (d[k].argmax(1) == d["y_true"]))
                    acc[k].setdefault("c", []).append(d[k].max(1))
        fig, ax = plt.subplots(figsize=(3.45, 2.5))
        for k, lab, col in series:
            if k not in acc:
                continue
            conf = np.concatenate(acc[k]["c"])
            corr = np.concatenate(acc[k]["p"]).astype(float)
            o = np.argsort(-conf)
            e = 1.0 - corr[o]
            cov = np.arange(1, len(e) + 1) / len(e)
            risk = np.cumsum(e) / np.arange(1, len(e) + 1)
            ax.plot(cov, risk, lw=1.1, color=col, label=f"{lab} (AURC {risk.mean():.4f})")
        ax.set_xlabel("coverage")
        ax.set_ylabel("selective risk")
        ax.set_title("Risk--coverage across all configurations", fontsize=8)
        ax.legend(frameon=False, fontsize=6.5)
        ax.grid(color="#EFEFEF", lw=.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig7_risk_coverage.pdf"))
        plt.savefig(os.path.join(FIG_DIR, "fig7_risk_coverage.png"), dpi=300)
        plt.close()
        log(f"  wrote {os.path.join(FIG_DIR, 'fig7_risk_coverage.pdf')}")
    except Exception as e:
        log(f"  risk-coverage figure failed: {e}", "WARN")

    # ablation deltas
    try:
        ap = summary["mean"]["APEX"]
        abl = sorted(k for k in summary["mean"] if k.startswith(("ABL_", "FAIR_")))
        if abl:
            names = [a.replace("ABL_", "").replace("FAIR_", "").replace("_", " ") for a in abl]
            d_acc = [(summary["mean"][a]["BAcc"] - ap["BAcc"]) * 100 for a in abl]
            fig, ax = plt.subplots(figsize=(3.45, 0.32 * len(abl) + 1.0))
            ax.barh(names, d_acc, color=["#D55E00" if v < 0 else "#009E73" for v in d_acc])
            ax.axvline(0, color="#555", lw=.8)
            ax.set_xlabel("balanced accuracy vs. full APEX (pp)")
            ax.set_title("Ablations and fair-comparison baselines", fontsize=8)
            ax.grid(axis="x", color="#EFEFEF", lw=.5)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "fig8_ablation.pdf"))
            plt.savefig(os.path.join(FIG_DIR, "fig8_ablation.png"), dpi=300)
            plt.close()
            log(f"  wrote {os.path.join(FIG_DIR, 'fig8_ablation.pdf')}")
    except Exception as e:
        log(f"  ablation figure failed: {e}", "WARN")


# ============================================================================ #
#  STATUS AND ENTRY POINT
# ============================================================================ #

def show_status() -> None:
    tot = len(DOMAINS) * len(ARCHS) * N_FOLDS
    idx = sum(done(os.path.join(INDEX_DIR, f"{k}.json")) for k in DOMAINS)
    w = sum(done(weight_path(a, k, f)) for k in DOMAINS for a in ARCHS
            for f in range(1, N_FOLDS + 1))
    fe = sum(done(feat_path(k, a, f)) for k in DOMAINS for a in ARCHS
             for f in range(1, N_FOLDS + 1))
    ce = sum(done(cell_path(k, a, f)) for k in DOMAINS for a in ARCHS
             for f in range(1, N_FOLDS + 1))
    print("\n  APEX pipeline status")
    print(f"    repair  index built     {idx}/{len(DOMAINS)}")
    print(f"    train   heads on disk   {w}/{tot}")
    print(f"    extract features cached {fe}/{tot}")
    print(f"    bench   cells evaluated {ce}/{tot}")
    print(f"    output  {OUT_DIR}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="APEX pipeline (resume-safe)")
    ap.add_argument("--stage", choices=["repair", "train", "extract", "bench", "analyse", "all"],
                    default="all")
    ap.add_argument("--force", nargs="*", default=None,
                    choices=["repair", "train", "extract", "bench", "all"],
                    help="redo these stages from scratch; --force all redoes everything")
    ap.add_argument("--domains", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--max-fit", type=int, default=None,
                    help="cap samples used to fit kernel experts (0 = no cap)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="reuse existing weights; retrain only CHEST-XRAY (folds changed "
                         "when its duplicate tree was quarantined). ~5 h instead of ~30 h, "
                         "but MC dropout stays inactive and must be dropped from the paper.")
    a = ap.parse_args()

    if a.max_fit is not None:
        globals()["MAX_FIT_SAMPLES"] = a.max_fit
    if a.quick:
        globals()["QUICK_MODE"] = True
        globals()["DROP_RATE"] = 0.0

    if a.status:
        show_status()
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    open_log()
    keep_awake()
    log(f"BASE_DIR   {BASE_DIR}")
    log(f"python     {sys.version.split()[0]} on {platform.platform()}")
    try:
        import torch
        log(f"torch      {torch.__version__} | CUDA {torch.cuda.is_available()}")
    except Exception:
        log("torch      NOT AVAILABLE -- train/extract will fail", "WARN")
    log("protocol   checkpoint selected on an inner split; outer fold never seen in training")
    log(f"APEX       tau={TAU}  lambda={LAMBDA}  dim={TARGET_DIM}  gamma={GAMMA}")
    if QUICK_MODE:
        log("mode       QUICK -- reusing existing weights, retraining CHEST-XRAY only.")
        log("           MC dropout will be INACTIVE; drop that baseline from the paper.", "WARN")
    else:
        log("mode       FULL -- retraining all heads with dropout so MC dropout is genuine.")
    log(f"max_fit    {MAX_FIT_SAMPLES or 'no cap'}")

    t0 = time.time()
    try:
        forced = set(a.force or [])
        if "all" in forced:
            forced = {"repair", "train", "extract", "bench"}
        if forced:
            log(f"force      redoing from scratch: {', '.join(sorted(forced))}", "WARN")
        stages = ["repair", "train", "extract", "bench", "analyse"] if a.stage == "all" else [a.stage]
        for s in stages:
            f = (s in forced)
            if s == "repair":
                stage_repair(force=f)
            elif s == "train":
                stage_train(only_domains=a.domains, force=f)
            elif s == "extract":
                stage_extract(force=f)
            elif s == "bench":
                stage_bench(force=f)
            elif s == "analyse":
                stage_analyse()
        log("=" * 78)
        log(f"ALL DONE in {fmt_eta(time.time()-t0)}")
        log("=" * 78)
        return 0
    except KeyboardInterrupt:
        log("interrupted by user -- progress is saved, rerun the same command to resume", "WARN")
        return 130
    except BaseException:
        log(traceback.format_exc(), "ERROR")
        log("rerun the same command to resume from the last checkpoint", "WARN")
        return 1
    finally:
        release_awake()
        if _LOG_FH:
            _LOG_FH.close()


if __name__ == "__main__":
    sys.exit(main())
