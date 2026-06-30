# Verified Model I/O Contract — `best_float32.tflite`

Measured 2026-06-30 with `ml-training/verify_model.py` + `probe_model.py` (TF 2.21 TFLite interpreter). **This is ground truth — do not guess.**

## Tensor shapes

| Tensor | Shape | Dtype | Quant |
|--------|-------|-------|-------|
| Input `serving_default` | `[1, 416, 416, 3]` | float32 | none (0..1 expected) |
| Output `Identity` | `[1, 8, 3549]` | float32 | none |

- Input: RGB, pixel values normalized to **0..1** (divide by 255). NHWC layout.
- Output: **anchor-free**, channel-first `[1, C=8, N=3549]`.
  - `N = 3549 = 52² + 26² + 13²` (strides 8/16/32 at 416).
  - `C = 8 = 4 (box) + 4 (class)`. **No objectness channel.**
  - Channels `0..3` = `cx, cy, w, h`, normalized **0..1** (relative to the 416 input).
  - Channels `4..7` = per-class confidence, already sigmoid-activated (0..1).
- Classes (index → label): `0 Resistor, 1 Diode, 2 Transistor, 3 Condensator`.

→ The existing Kotlin decode **shape assumption was correct.** The 5 prior "decode fix" commits chased the wrong bug.

## Preprocessing — letterbox is REQUIRED

Naive `resize(416,416)` (aspect-distorting) nearly blinds the model. Letterbox (aspect-preserving + gray `114,114,114` pad) roughly 10–16× the confidence:

| Image | naive global-max | letterbox global-max |
|-------|------------------|----------------------|
| `1.jpeg` (real SMD tape) | 0.016 | **0.263** |
| `2.jpeg` (corrupted screenshot) | 0.285 | 0.481 |

Kotlin currently uses `Bitmap.createScaledBitmap` (naive) → **must switch to letterbox**, and un-letterbox the output boxes back to original-image coordinates for the overlay.

## Model quality caveat

Even with letterbox, peak confidence on the genuine SMD-tape photo is only **0.263** (and mislabels resistors as Transistor). The shipped `best_float32.tflite` is a weak YOLOv5n. Correct preprocessing makes it *usable for testing* but **retraining (Phase 3) is required for production accuracy** — this is not a threshold-tuning problem.

## Recommended runtime constants

- Confidence threshold: **0.25** for the retrained model; temporarily **0.15** to surface the weak current model during integration testing.
- NMS IoU: **0.5** (unchanged, works).
