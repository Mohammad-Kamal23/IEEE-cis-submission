# APEX — mechanism experiments, run 2026-08-31
All computed from cached artefacts. No retraining. No new data.

## RQ1 — AURC is invariant under monotone reparametrisation of confidence; ECE is not

Five strictly increasing maps applied to the uncalibrated confidences of all 18
configurations:

| map | mean AURC | max deviation | mean ECE |
|---|---|---|---|
| identity | 0.04842737 | — | 0.0395 |
| c² | 0.04842737 | 0.00e+00 | 0.1429 |
| sqrt(c) | 0.04842737 | 0.00e+00 | 0.0494 |
| logistic(6(c−½)) | 0.04842737 | 0.00e+00 | 0.0478 |
| ½ + ½c | 0.04842737 | 0.00e+00 | 0.0580 |

AURC is **bit-identical** (deviation exactly 0) across every map. ECE moves by 3.6×.
Corollary: ECE cannot serve as a surrogate for selective ranking. The 15/18
disagreement already in the paper is the empirical magnitude of this.

## RQ2 — Latent geometry predicts error after conditioning on the model's confidence

Protocol: neighbours drawn only from D_fit; correctness predictor fitted on D_cal,
scored on the untouched validation fold. Confidence block = {max prob, margin,
softmax entropy}. Geometry block adds {neighbourhood label entropy, neighbour
agreement with the predicted class, distance to nearest neighbour, mean distance}.

| | improves in | mean change | Wilcoxon p |
|---|---|---|---|
| AUROC of the correctness predictor | 13/18 | +0.0274 | 3.4×10⁻³ |
| NLL of the correctness predictor | 16/18 | −0.0339 | 3.8×10⁻⁵ |

The frozen representation carries error-predictive information the linear head
discards.

## RQ4 — The amount of that information predicts how much APEX helps

Spearman across the 18 configurations:

- geometry AUROC gain vs APEX AURC reduction: **ρ = +0.872, p < 10⁻⁴**
- geometry NLL gain vs APEX AURC reduction: **ρ = +0.886, p < 10⁻⁴**

| domain | backbone | geometry ΔAUROC | APEX AURC cut |
|---|---|---|---|
| BLOODCELL | MobileNetV3 | +0.1239 | 81% |
| BLOODCELL | ViT | +0.1215 | 96% |
| BLOODCELL | ConvNeXt | +0.0707 | 68% |
| ULTRASOUND | MobileNetV3 | +0.0496 | 45% |
| BRAINTUMOR | MobileNetV3 | +0.0290 | 62% |
| RETINAL | ViT | +0.0256 | 50% |
| BRAINTUMOR | ConvNeXt | +0.0239 | 66% |
| BRAINTUMOR | ViT | +0.0230 | 69% |
| RETINAL | MobileNetV3 | +0.0169 | 33% |
| RETINAL | ConvNeXt | +0.0144 | 37% |
| CHEST-XRAY | MobileNetV3 | +0.0118 | 26% |
| CHEST-XRAY | ViT | +0.0101 | 26% |
| ENDOSCOPIC | ViT | +0.0025 | 21% |
| CHEST-XRAY | ConvNeXt | −0.0006 | 15% |
| ULTRASOUND | ConvNeXt | −0.0018 | 27% |
| ULTRASOUND | ViT | −0.0025 | 37% |
| ENDOSCOPIC | MobileNetV3 | −0.0059 | 18% |
| ENDOSCOPIC | ConvNeXt | −0.0192 | 15% |

## Caveats, stated plainly

- Ultrasound ViT and ConvNeXt have a negative geometry signal yet still gain from
  APEX. The probe is noisy there — 778 images, ~500 in D_fit, k=20.
- One value of k, one feature set, one distance. Robustness to k and to the
  geometry features is not yet checked.
- n = 18 for the correlation. Strong, but 18.
