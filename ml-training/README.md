# ML Training — SMD Component Detection (YOLOv8n → TFLite)

Train **YOLOv8n** on the smdComponents dataset and export a TFLite model that matches
the verified Android runtime contract. The exported model is a drop-in replacement for
`app/src/main/assets/best_float32.tflite`.

## Quick Start (Google Colab — Recommended)

1. Open `train_yolo.ipynb` in Google Colab.
2. Get a free Roboflow API key: https://app.roboflow.com/settings/api
3. Paste your API key into the `API_KEY` variable in the first code cell.
4. **Run All**.
5. Download the exported `*_float32.tflite` from the `runs/` folder.

## Local Training

```bash
pip install -r requirements.txt
python train.py --api_key YOUR_ROBOFLOW_KEY
```

(Optional) override epochs: `python train.py --api_key YOUR_KEY --epochs 100`.

A GPU (CUDA) is recommended for reasonable training time.

## Output

After training, Ultralytics writes the exported model under `runs/detect/train/weights/`
(e.g. `best_saved_model/best_float32.tflite`).

1. Copy the exported `*_float32.tflite` to `app/src/main/assets/best_float32.tflite`.
2. Rebuild the APK: `./gradlew assembleDebug`.

## Verified Runtime Contract

The Android app and this pipeline target the contract measured in
[`MODEL_IO.md`](./MODEL_IO.md) — **this is ground truth**:

- **Input**: `[1, 416, 416, 3]` float32, RGB, pixel values normalized to **0..1** (÷255), NHWC.
- **Output**: `[1, 8, 3549]` float32, **anchor-free**, channel-first.
  - `N = 3549 = 52² + 26² + 13²` (strides 8/16/32 at 416 input).
  - `C = 8 = 4 (box: cx, cy, w, h) + 4 (per-class confidence)`. **No objectness channel.**
- **Classes** (index → label): `0 Resistor`, `1 Diode`, `2 Transistor`, `3 Condensator`.
- **Preprocessing**: **letterbox** (aspect-preserving resize + gray `114,114,114` pad) is
  required; naive `resize(416,416)` nearly blinds the model. See `MODEL_IO.md`.

YOLOv8 `imgsz=416` export produces exactly this anchor-free `[1, 8, 3549]` layout.

## Dataset

**smdComponents** by Dainius (Vilnius Tech University)
- 4 classes: Resistor, Diode, Transistor, Condensator (Capacitor)
- ~3,000+ labeled images
- Source: https://universe.roboflow.com/dainius/smdcomponents
- Roboflow workspace `dainius`, project `smdcomponents`, version `2` (downloaded as `yolov8`).
