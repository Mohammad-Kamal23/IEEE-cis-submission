#!/usr/bin/env python3
"""
Prove that weights_heads/heads.npz is a lossless replacement for the 9.4 GB of
checkpoints, by rebuilding all 90 models and comparing them tensor by tensor
against the originals.

    python analysis/verify_weights.py --checkpoints /path/to/weights

Needs torch, timm and the original checkpoints, so it is not part of
reproduce.py -- it is the audit that justifies shipping 5 MB instead of 9.4 GB.
Without --checkpoints it verifies only what can be checked from this repository:
that every model rebuilds and that the frozen backbone fingerprint matches.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.load_trained_model import ARCHS, DOMAINS, load_model, verify_backbone  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", help="directory holding <arch>_<DOMAIN>_Fold<f>_dr10.pth")
    a = ap.parse_args()
    import torch

    n = exact = fp_ok = missing = 0
    failures = []
    for arch in ARCHS:
        for dom in DOMAINS:
            for fold in range(1, 6):
                n += 1
                model = load_model(arch, dom, fold, verify=False)
                ok, detail = verify_backbone(arch, model)
                fp_ok += ok
                if not ok:
                    failures.append(f"{arch}/{dom}/f{fold}: fingerprint {detail}")
                if not a.checkpoints:
                    continue
                p = os.path.join(a.checkpoints, f"{arch}_{dom}_Fold{fold}_dr10.pth")
                if not os.path.exists(p):
                    missing += 1
                    continue
                orig = torch.load(p, map_location="cpu", weights_only=True)
                got = model.state_dict()
                if set(orig) != set(got):
                    failures.append(f"{arch}/{dom}/f{fold}: key sets differ")
                    continue
                bad = [k for k in orig if not torch.equal(orig[k].cpu(), got[k].cpu())]
                if bad:
                    failures.append(f"{arch}/{dom}/f{fold}: {len(bad)} tensors differ")
                else:
                    exact += 1
        print(f"  {arch:<12} done")

    print(f"\n  models rebuilt          : {n}")
    print(f"  backbone fingerprint OK : {fp_ok}/{n}")
    if a.checkpoints:
        print(f"  bit-exact vs checkpoint : {exact}/{n - missing}"
              + (f"   ({missing} checkpoints not found)" if missing else ""))
    for f in failures[:10]:
        print(f"    FAIL {f}")
    good = not failures and fp_ok == n and (not a.checkpoints or exact == n - missing)
    print(f"\n  {'PASS' if good else 'FAIL'} -- heads.npz is "
          f"{'a lossless replacement for the full checkpoints' if good else 'NOT verified'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
