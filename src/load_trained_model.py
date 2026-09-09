#!/usr/bin/env python3
"""
Rebuild any of the 90 trained models from `weights_heads/heads.npz`.

The backbones are frozen at their ImageNet weights, so the full 107 MB
checkpoints are almost entirely redundant: across all 90 of them the backbone
tensors are byte-identical. Only what actually changed during training is stored
here -- 5 MB in place of 9.4 GB.

    from src.load_trained_model import load_model
    model = load_model("ConvNeXt", "BLOODCELL", fold=1, n_classes=4)

What is stored, per architecture:

  ViT, ConvNeXt   the classifier weight and bias. Nothing else ever changes;
                  these networks normalise per layer.
  MobileNetV3     the classifier weight and bias, plus 138 BatchNorm running
                  statistics. Batch normalisation updates its running mean and
                  variance during training even when every weight is frozen.
                  Those updates come only from the inner training split -- never
                  from an evaluation fold -- so they carry no leakage, but they
                  are part of the trained model and are needed to reproduce it.

Requires torch, and timm for ViT/ConvNeXt. The ImageNet weights are downloaded
by timm/torchvision on first use.
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
HEADS = os.path.join(_ROOT, "weights_heads", "heads.npz")
MANIFEST = os.path.join(_ROOT, "weights_heads", "MANIFEST.json")

DOMAINS = ["BLOODCELL", "BRAINTUMOR", "CHEST-XRAY", "ENDOSCOPIC", "RETINAL", "ULTRASOUND"]
ARCHS = ["ViT", "ConvNeXt", "MobileNetV3"]
N_CLASSES = {"BLOODCELL": 4, "BRAINTUMOR": 4, "CHEST-XRAY": 2,
             "ENDOSCOPIC": 4, "RETINAL": 4, "ULTRASOUND": 3}
DROP_RATE = 0.10


def build_backbone(arch: str, n_classes: int, pretrained: bool = True):
    """Identical to build_model() in apex_pipeline.py."""
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


def load_model(arch: str, domain: str, fold: int, n_classes: int | None = None,
               pretrained: bool = True, verify: bool = True):
    """Return the trained model for one (architecture, dataset, fold) cell."""
    import torch
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}")
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}")
    if not 1 <= fold <= 5:
        raise ValueError("fold must be 1..5")
    if not os.path.exists(HEADS):
        raise FileNotFoundError(f"{HEADS} not found")

    n_classes = n_classes or N_CLASSES[domain]
    model = build_backbone(arch, n_classes, pretrained=pretrained)
    sd = model.state_dict()

    with np.load(HEADS) as z:
        prefix = f"{arch}|{domain}|{fold}|"
        stored = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
    if not stored:
        raise KeyError(f"no stored tensors for {arch}/{domain}/fold {fold}")

    for name, arr in stored.items():
        if name not in sd:
            raise KeyError(f"{name} is not in the freshly built {arch}")
        if tuple(sd[name].shape) != tuple(arr.shape):
            raise ValueError(f"{name}: expected {tuple(sd[name].shape)}, stored {tuple(arr.shape)}")
        sd[name] = torch.from_numpy(arr)

    model.load_state_dict(sd)
    model.eval()

    if pretrained and verify:
        ok, detail = verify_backbone(arch, model)
        if not ok:
            import warnings
            warnings.warn(
                f"frozen backbone fingerprint mismatch for {arch}: {detail}. "
                "The upstream ImageNet weights have probably changed since these "
                "models were trained; predictions will not match the paper. The "
                "full checkpoints in the supplementary archive are unaffected.",
                RuntimeWarning)
    return model


def verify_backbone(arch: str, model) -> tuple[bool, str]:
    """Recompute the frozen-backbone fingerprint and compare with the manifest.

    The stored heads are only sufficient if the frozen part of the network is
    exactly what it was during training. This checks that, instead of assuming it.
    """
    import hashlib
    man = describe()["architectures"][arch]
    expected = man.get("frozen_backbone_sha256")
    if not expected:
        return True, "no fingerprint recorded"
    with np.load(HEADS) as z:
        trained = {k.split("|", 3)[3] for k in z.files if k.startswith(f"{arch}|")}
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(k for k in sd if k not in trained):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    got = h.hexdigest()
    return got == expected, f"expected {expected[:16]}..., got {got[:16]}..."


def describe() -> dict:
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    import sys
    z = np.load(HEADS)
    cells = sorted({tuple(k.split("|")[:3]) for k in z.files})
    print(f"{len(cells)} trained models stored in "
          f"{os.path.getsize(HEADS) / 1e6:.1f} MB\n")
    m = describe()
    for a in ARCHS:
        info = m["architectures"][a]
        print(f"  {a:<12} {info['library']}/{info['model']}")
        print(f"               trained : {', '.join(info['trained_tensor_names'])}")
        if info["n_adapted_buffers"]:
            print(f"               adapted : {info['n_adapted_buffers']} BatchNorm buffers")
    print(f"\n  backbone check: the non-stored tensors are byte-identical across all")
    print(f"  {len(cells)} checkpoints, which is what makes this file sufficient.")
    if "--load" in sys.argv:
        mdl = load_model("ConvNeXt", "BLOODCELL", 1)
        print(f"\n  loaded ConvNeXt/BLOODCELL/fold 1 -> "
              f"{sum(p.numel() for p in mdl.parameters()):,} parameters")
