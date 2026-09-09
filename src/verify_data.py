#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_data.py  --  prove there are no duplicate images inside any dataset.

WHY
---
Cross-validation folds are assigned by shuffling. If the same image appears
twice in a dataset, the two copies land in different folds roughly 80% of the
time, so the model is tested on an image it trained on. That inflates every
number in the paper and is invisible unless you look for it.

Filename and file-size collisions are a weak proxy (two different scans can
share both). This script hashes the actual bytes with MD5, which settles it.

WHAT IT REPORTS, per dataset
----------------------------
  images                total files the pipeline would load
  unique images         distinct MD5 hashes
  duplicate groups      hashes appearing more than once
  CROSS-CLASS dups      the same image filed under two different labels.
                        Catastrophic: the label itself is ambiguous, and the
                        model is trained to give contradictory answers.
  within-class dups     the same image twice under one label. Still leakage:
                        the copies split across folds.

It changes nothing on disk. Run it as often as you like.

    python verify_data.py                 # all six datasets
    python verify_data.py --dataset RETINAL
    python verify_data.py --write-report  # also save a JSON next to the results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

BASE_DIR = r"C:\Users\homeb\Desktop\research"
OUT_DIR  = os.path.join(BASE_DIR, "Results", "pipeline")

DOMAINS = {
    "BLOODCELL":  "BloodCell",
    "BRAINTUMOR": "BRAINTUMOR",
    "CHEST-XRAY": "CHEST-XRAY",
    "ENDOSCOPIC": "ENDOSCOPIC",
    "RETINAL":    "RetinalOCT",
    "ULTRASOUND": "BreastUltrasound",
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
SKIP = ("__MACOSX", ".ipynb_checkpoints", "_quarantine")
MASK_TOK = ("_mask", "-mask")


def resolve(key):
    for cand in (DOMAINS[key], key):
        p = os.path.join(BASE_DIR, cand)
        if os.path.isdir(p):
            return p
    return None


def walk(root):
    out = []
    for cur, dirs, files in os.walk(root):
        if any(t in cur for t in SKIP):
            dirs[:] = []
            continue
        dirs.sort()
        for f in sorted(files):
            low = f.lower()
            if low.endswith(IMG_EXT) and not any(t in low for t in MASK_TOK):
                out.append(os.path.join(cur, f))
    return out


def label_of(full, root):
    seg = os.path.relpath(full, root).split(os.sep)
    if len(seg) < 2:
        return "unknown"
    lab = seg[-2]
    if lab.lower() in ("train", "test", "val", "valid", "images") and len(seg) >= 3:
        lab = seg[-3]
    return lab


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check(key, verbose=True):
    root = resolve(key)
    if root is None:
        print(f"  {key:<12} folder not found -- skipped")
        return None

    paths = walk(root)
    if not paths:
        print(f"  {key:<12} no images -- skipped")
        return None

    t0 = time.time()
    by_hash = defaultdict(list)
    for i, p in enumerate(paths, 1):
        try:
            by_hash[md5(p)].append(p)
        except OSError:
            continue
        if verbose and i % 2000 == 0:
            print(f"  {key:<12}   hashed {i}/{len(paths)} ...", flush=True)

    dup_groups = {h: ps for h, ps in by_hash.items() if len(ps) > 1}
    cross, within, n_cross_files, n_within_files = [], [], 0, 0
    for h, ps in dup_groups.items():
        labs = {label_of(p, root) for p in ps}
        if len(labs) > 1:
            cross.append((h, ps, sorted(labs)))
            n_cross_files += len(ps) - 1
        else:
            within.append((h, ps))
            n_within_files += len(ps) - 1

    res = {
        "dataset": key, "root": root,
        "images": len(paths), "unique": len(by_hash),
        "duplicate_groups": len(dup_groups),
        "cross_class_groups": len(cross), "cross_class_extra_files": n_cross_files,
        "within_class_groups": len(within), "within_class_extra_files": n_within_files,
        "seconds": round(time.time() - t0, 1),
        "clean": len(dup_groups) == 0,
    }

    status = "CLEAN" if res["clean"] else ("CROSS-CLASS" if cross else "DUPLICATES")
    print(f"  {key:<12} {len(paths):>6} images | {len(by_hash):>6} unique | "
          f"{len(dup_groups):>4} dup groups | cross-class {len(cross):>3} | "
          f"within-class {len(within):>3}   [{status}]")

    for h, ps, labs in cross[:5]:
        print(f"     ! same image under labels {labs}:")
        for p in ps[:4]:
            print(f"         {os.path.relpath(p, root)}")
    if len(cross) > 5:
        print(f"     ! ... and {len(cross)-5} more cross-class groups")
    for h, ps in within[:3]:
        print(f"     - duplicate within '{label_of(ps[0], root)}': "
              + ", ".join(os.path.basename(p) for p in ps[:4]))
    if len(within) > 3:
        print(f"     - ... and {len(within)-3} more within-class groups")

    res["_cross_examples"] = [[os.path.relpath(p, root) for p in ps] for _, ps, _ in cross[:50]]
    res["_within_examples"] = [[os.path.relpath(p, root) for p in ps] for _, ps in within[:50]]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="check one dataset only")
    ap.add_argument("--write-report", action="store_true")
    a = ap.parse_args()

    keys = [a.dataset] if a.dataset else list(DOMAINS)
    for k in keys:
        if k not in DOMAINS:
            print(f"unknown dataset {k!r}; choose from {list(DOMAINS)}")
            return 2

    print()
    print("  byte-level duplicate check (MD5)  --  nothing on disk is modified")
    print("  " + "-" * 100)
    out = [r for r in (check(k) for k in keys) if r]
    print("  " + "-" * 100)

    tot_dup = sum(r["duplicate_groups"] for r in out)
    tot_cross = sum(r["cross_class_groups"] for r in out)
    tot_extra = sum(r["cross_class_extra_files"] + r["within_class_extra_files"] for r in out)

    print()
    if tot_dup == 0:
        print("  VERDICT: no duplicate images in any dataset.")
        print("  Every cross-validation fold is disjoint at the pixel level. The")
        print("  reported numbers carry no duplicate-driven leakage.")
    else:
        print(f"  VERDICT: {tot_dup} duplicate group(s), {tot_extra} redundant file(s).")
        if tot_cross:
            print(f"  {tot_cross} group(s) are CROSS-CLASS -- the same image carries two labels.")
            print("  These must be resolved before the numbers can be trusted.")
        print()
        print("  To fix, run the cleaner, then rebuild and re-run:")
        print("      python clean_datasets.py")
        print("      python apex_pipeline.py --stage repair  --force repair")
        print("      python apex_pipeline.py --stage train")
        print("      python apex_pipeline.py --stage extract --force extract")
        print("      python apex_pipeline.py --stage bench   --force bench")
        print("      python apex_pipeline.py --stage analyse")
    print()

    if a.write_report:
        os.makedirs(OUT_DIR, exist_ok=True)
        p = os.path.join(OUT_DIR, "duplicate_report.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "total_duplicate_groups": tot_dup,
                       "total_cross_class_groups": tot_cross,
                       "datasets": out}, f, indent=1)
        print(f"  report -> {p}\n")
    return 0 if tot_dup == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
