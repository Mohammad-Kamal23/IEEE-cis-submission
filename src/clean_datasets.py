#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clean_datasets.py  --  remove the three things that make the current numbers
                       untrustworthy, without deleting anything.

WHAT IT REMOVES, and why each one matters
-----------------------------------------
1. SEGMENTATION MASKS  (BreastUltrasound only, 798 files of 1,578)
   BUSI ships each scan next to its ground-truth mask:

       benign (1).png          <- the ultrasound scan
       benign (1)_mask.png     <- a white blob on black, drawn by a radiologist

   The pipeline globbed *.png and loaded both. A mask is not a picture of a
   patient; it is the answer. Benign masks are small and round, malignant masks
   are large and ragged, normal masks are empty. A classifier can read the label
   off the silhouette without learning anything about ultrasound. Every
   ULTRASOUND number in the current results was computed on a set that is half
   answer key, so none of them can be reported.

2. CROSS-CLASS DUPLICATES  (4 groups, 8 files)
   The identical file sits under two different labels:

       RetinalOCT   CNV-samples/1110.jpg   ==  Drusen-samples/144.jpg
       RetinalOCT   CNV-samples/982.jpg    ==  Drusen-samples/35.jpg
       RetinalOCT   CNV-samples/988.jpg    ==  Drusen-samples/38.jpg
       BUSI         benign (433).png       ==  malignant (145).png

   One of the two labels is wrong and there is no way to tell which, so the
   model is trained toward two contradictory targets and then graded on one of
   them at random. Both copies come out.

3. WITHIN-CLASS DUPLICATE COPIES  (404 groups)
   Folds are assigned by shuffling, so two copies of one image land in
   different folds about 80% of the time and the model is tested on something
   it memorised. This flatters everything, and it flatters nearest-neighbour
   methods most of all: a k-NN expert facing a test image whose exact twin sits
   in the training fold retrieves it at distance zero. One copy of each is kept
   -- the lexicographically first, so the choice is deterministic.

NOTHING IS DELETED
------------------
Every file is MOVED to  <research>\\_quarantine\\<DATASET>\\<reason>\\...  and
recorded in _quarantine\\manifest.json.  `--restore` puts all of it back
exactly where it was. The quarantine folder sits outside every dataset root
and its name is in the pipeline's skip list, so nothing there is ever loaded.

USAGE
-----
    python clean_datasets.py --dry-run     # report only, touch nothing
    python clean_datasets.py               # quarantine, write the manifest
    python clean_datasets.py --restore     # undo, using the manifest
    python clean_datasets.py --dataset RETINAL --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict

BASE_DIR = r"C:\Users\homeb\Desktop\research"
QDIR     = os.path.join(BASE_DIR, "_quarantine")
MANIFEST = os.path.join(QDIR, "manifest.json")

DOMAINS = {
    "BLOODCELL":  "BloodCell",
    "BRAINTUMOR": "BRAINTUMOR",
    "CHEST-XRAY": "CHEST-XRAY",
    "ENDOSCOPIC": "ENDOSCOPIC",
    "RETINAL":    "RetinalOCT",
    "ULTRASOUND": "BreastUltrasound",
}
IMG_EXT   = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
SKIP_DIRS = ("__MACOSX", ".ipynb_checkpoints", "_quarantine")
MASK_TOK  = ("_mask", "-mask")

R_MASK  = "segmentation_mask"
R_CROSS = "cross_class_conflict"
R_DUP   = "duplicate_copy"


# ---------------------------------------------------------------- filesystem

def resolve(key):
    for cand in (DOMAINS[key], key):
        p = os.path.join(BASE_DIR, cand)
        if os.path.isdir(p):
            return p
    return None


def walk(root):
    out = []
    for cur, dirs, files in os.walk(root):
        if any(t in cur for t in SKIP_DIRS):
            dirs[:] = []
            continue
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
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


def is_mask(path):
    low = os.path.basename(path).lower()
    return any(t in low for t in MASK_TOK)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


# ------------------------------------------------------------------- planning

def plan_dataset(key, verbose=True):
    """Return (root, [(src, reason, note), ...]) -- what should leave."""
    root = resolve(key)
    if root is None:
        print(f"  {key:<12} folder not found -- skipped")
        return None, []

    paths = walk(root)
    if not paths:
        print(f"  {key:<12} no images -- skipped")
        return root, []

    actions = []

    # 1. masks -------------------------------------------------------------
    masks = [p for p in paths if is_mask(p)]
    for p in masks:
        actions.append((p, R_MASK, ""))
    scans = [p for p in paths if not is_mask(p)]

    # 2 + 3. duplicates among what remains ---------------------------------
    by_hash = defaultdict(list)
    for p in scans:
        try:
            by_hash[md5(p)].append(p)
        except OSError:
            continue

    for h, ps in by_hash.items():
        if len(ps) < 2:
            continue
        labs = {label_of(p, root) for p in ps}
        if len(labs) > 1:
            note = " == ".join(sorted(labs))
            for p in ps:                       # every member leaves
                actions.append((p, R_CROSS, note))
        else:
            keep = sorted(ps)[0]
            for p in sorted(ps)[1:]:
                actions.append((p, R_DUP, "kept " + os.path.basename(keep)))

    n_cross = sum(1 for a in actions if a[1] == R_CROSS)
    n_dup   = sum(1 for a in actions if a[1] == R_DUP)
    kept    = len(paths) - len(actions)
    if verbose:
        print(f"  {key:<12} {len(paths):>6} -> {kept:>6}   {len(masks):>9}   "
              f"{n_cross:>11}   {n_dup:>10}")
    return root, actions


# ---------------------------------------------------------------------- main

def do_clean(keys, dry):
    print()
    print("  dataset cleanup -- files are MOVED to _quarantine, never deleted")
    print("  " + "-" * 88)
    print(f"  {'dataset':<12}{'before':>7} -> {'after':>6}   {'masks':>9}   "
          f"{'cross-class':>11}   {'dup copies':>10}")
    print("  " + "-" * 88)

    manifest = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f).get("moved", [])
    already = {m["src"] for m in manifest}

    total = 0
    for k in keys:
        root, actions = plan_dataset(k)
        if not actions:
            continue
        for src, reason, note in actions:
            if src in already:
                continue
            rel = os.path.relpath(src, root)
            dst = os.path.join(QDIR, k, reason, rel)
            if not dry:
                try:
                    move(src, dst)
                except OSError as e:
                    print(f"     ! could not move {rel}: {e}")
                    continue
            manifest.append({"dataset": k, "src": src, "dst": dst,
                             "reason": reason, "note": note})
            total += 1

    print("  " + "-" * 88)
    print()
    if dry:
        print(f"  DRY RUN -- {total} file(s) WOULD move. Nothing on disk changed.")
        print("  Run the same command without --dry-run to apply it.")
        print()
        return 0

    os.makedirs(QDIR, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "base": BASE_DIR, "moved": manifest}, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MANIFEST)

    print(f"  {total} file(s) moved this run; {len(manifest)} in quarantine in total.")
    print(f"  quarantine : {QDIR}")
    print(f"  manifest   : {MANIFEST}   (python clean_datasets.py --restore undoes it)")
    print()
    print("  Everything downstream was computed on the old file list, so it all")
    print("  has to be rebuilt. In order:")
    print()
    print("      python verify_data.py                                  # must say CLEAN")
    print("      python apex_pipeline.py --stage repair  --force repair")
    print("      python apex_pipeline.py --stage train   --force train")
    print("      python apex_pipeline.py --stage extract --force extract")
    print("      python apex_pipeline.py --stage bench   --force bench")
    print("      python apex_pipeline.py --stage analyse")
    print()
    print("  or simply:  powershell -ExecutionPolicy Bypass -File .\\RUN_APEX.ps1")
    print()
    return 0


def do_restore():
    if not os.path.exists(MANIFEST):
        print(f"\n  no manifest at {MANIFEST} -- nothing to restore\n")
        return 1
    with open(MANIFEST, encoding="utf-8") as f:
        moved = json.load(f).get("moved", [])
    back = fail = 0
    for m in moved:
        if not os.path.exists(m["dst"]):
            continue
        try:
            move(m["dst"], m["src"])
            back += 1
        except OSError as e:
            print(f"  ! {m['dst']}: {e}")
            fail += 1
    print(f"\n  restored {back} file(s); {fail} failed.")
    if not fail:
        os.replace(MANIFEST, MANIFEST + ".restored")
        print("  manifest renamed to manifest.json.restored\n")
    return 0 if not fail else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only")
    ap.add_argument("--restore", action="store_true", help="undo a previous run")
    ap.add_argument("--dataset", default=None, help="one dataset only")
    a = ap.parse_args()

    if a.restore:
        return do_restore()

    keys = [a.dataset] if a.dataset else list(DOMAINS)
    for k in keys:
        if k not in DOMAINS:
            print(f"unknown dataset {k!r}; choose from {list(DOMAINS)}")
            return 2
    return do_clean(keys, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
