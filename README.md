# Component Counter

Android app for counting SMD electronic components on tape/reel using camera + TensorFlow Lite object detection.

Built with Jetpack Compose, CameraX, and a YOLOv8 TFLite model (`best_float32.tflite`).

## Features

- Real-time component detection via camera
- 4 component types: Resistor, Diode, Transistor, Condensator
- Bounding boxes with confidence scores
- Snapshot capture & history
- CSV export
- Signed APK release via CI/CD

## Architecture

```
MVVM + StateFlow + Compose
├── CameraX (camera preview + frame analysis)
├── TFLite YOLOv8 (best_float32.tflite, 416 input, NMS in ObjectDetectorHelper)
├── CounterViewModel (state management)
└── Material3 UI (Camera + History screens)
```

## Training the Model

The `ml-training/` pipeline trains a YOLOv8 model and exports `best_float32.tflite`.
See [`ml-training/README.md`](ml-training/README.md) for training instructions and
[`ml-training/MODEL_IO.md`](ml-training/MODEL_IO.md) for the verified I/O contract
(416×416 input, `[1,8,3549]` output, letterbox preprocessing).

Quick start: Open the [`train_yolo.ipynb`](ml-training/train_yolo.ipynb) Colab notebook and run with a free Roboflow API key.

## Building

```bash
# Debug
./gradlew assembleDebug

# Release (requires signing config)
./gradlew assembleRelease
```

## CI/CD

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| Code Review | PR | detekt + lint + unit tests |
| QA | PR / push main | Unit tests |
| Release | push main | Build signed APK + GitHub Release |

## Commit Convention

```
feat: add new feature     → minor version bump
fix: bug fix              → patch version bump
BREAKING CHANGE: ...      → major version bump
```

## License

Public Domain — see `ml-training/README.md` for dataset attribution.
