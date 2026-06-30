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
- Classes (index → label): `0 Condensator, 1 Diode, 2 Resistor, 3 Transistor` — this is the smdComponents `data.yaml` order the model is trained on; the app's label list must match exactly.

→ The existing Kotlin decode **shape assumption was correct.** The 5 prior "decode fix" commits chased the wrong bug.

## Preprocessing — letterbox is REQUIRED

Naive `resize(416,416)` (aspect-distorting) nearly blinds the model. Letterbox (aspect-preserving + gray `114,114,114` pad) roughly 10–16× the confidence:

| Image | naive global-max | letterbox global-max |
|-------|------------------|----------------------|
| `1.jpeg` (real SMD tape) | 0.016 | **0.263** |
| `2.jpeg` (corrupted screenshot) | 0.285 | 0.481 |

Kotlin currently uses `Bitmap.createScaledBitmap` (naive) → **must switch to letterbox**, and un-letterbox the output boxes back to original-image coordinates for the overlay.

## Model quality

**Retrained 2026-06-30** with the YOLOv8n pipeline (`ml-training/train.py`), 10 epochs on smdComponents: **mAP50 0.993, mAP50-95 0.711**. On the genuine SMD-tape photo (`source/1.jpeg`) it now detects **16 Resistors at conf up to 0.59** with correct labels (was 0.263, mislabelled). The previous shipped YOLOv5n was weak (0.263 peak) — replaced.

Export path on Windows: `best.pt → ONNX (ultralytics) → onnx2tf → best_float32.tflite` (LiteRT export is Linux/macOS-only in ultralytics ≥8.4.83). Result verified `[1,8,3549]`, identical to the runtime contract above.

## Recommended runtime constants

- Confidence threshold: **0.25** (in use with the retrained model).
- NMS IoU: **0.5** (unchanged, works).
