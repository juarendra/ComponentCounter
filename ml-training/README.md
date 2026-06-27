# ML Training — SMD Component Detection

Train EfficientDet-Lite0 to detect electronic components on tape/reel.

## Quick Start (Google Colab — Recommended)

1. Open `train_efficientdet.ipynb` in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juarendra/ComponentCounter/blob/main/ml-training/train_efficientdet.ipynb)

2. Get a free Roboflow API key:
   - Go to https://app.roboflow.com/settings/api
   - Copy your API key

3. Paste API key in the notebook → Run All

4. Download `efficientdet-lite0.tflite` from Colab

## Local Training

```bash
pip install -r requirements.txt
python train.py --api_key YOUR_ROBOFLOW_KEY
```

Requires Python 3.9+ with GPU (CUDA) for reasonable training time.

## Output

- `output/efficientdet-lite0.tflite` — TFLite model file (~6-8 MB)
- Copy to `app/src/main/assets/` in the Android project
- Rebuild APK

## Dataset

**smdComponents** by Dainius (Vilnius Tech University)
- 4 classes: Resistor, Diode, Transistor, Condensator (Capacitor)
- 3,000+ labeled images
- Source: https://universe.roboflow.com/dainius/smdcomponents
- mAP 99.5%, Precision 99.7%, Recall 99.7% (Roboflow benchmark)

## Model

- Architecture: EfficientDet-Lite0 (TensorFlow Model Maker)
- Input: 320×320 RGB
- Output: bounding boxes + class labels + confidence scores
- Quantization: FP16 (default) or INT8 (optional)
