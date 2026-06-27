#!/usr/bin/env python3
"""
One-click training: download dataset → train EfficientDet-Lite0 → export .tflite
Usage:
  1. pip install -r requirements.txt
  2. python train.py --api_key YOUR_ROBOFLOW_KEY
  3. Copy output/efficientdet-lite0.tflite to app/src/main/assets/
"""

import argparse
import os
import sys
import subprocess
import json

REQUIREMENTS = [
    "tensorflow==2.16.0",
    "tflite-model-maker",
    "roboflow",
    "pycocotools",
]

LABEL_MAP = {1: "Resistor", 2: "Diode", 3: "Transistor", 4: "Condensator"}
MODEL_SPEC = "efficientdet-lite0"
EPOCHS = 50
BATCH_SIZE = 8


def check_deps():
    for pkg in REQUIREMENTS:
        try:
            __import__(pkg.replace("-", "_").split("==")[0])
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg]
            )


def download_dataset(api_key: str) -> str:
    """Download smdComponents dataset from Roboflow in TFRecord format."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("dainius").project("smdcomponents")
    dataset = project.version(2).download("tfrecord")
    print(f"✅ Dataset downloaded to: {dataset.location}")
    return dataset.location


def train_model(dataset_path: str):
    """Train EfficientDet-Lite0 on the dataset."""
    from tflite_model_maker import configs, model_spec, object_detector

    spec = model_spec.get(MODEL_SPEC)
    print(f"✅ Loaded model spec: {MODEL_SPEC}")

    train_data, val_data, test_data = object_detector.DataLoader.from_tfrecord(
        train_dir=os.path.join(dataset_path, "train"),
        validation_dir=os.path.join(dataset_path, "valid"),
        test_dir=os.path.join(dataset_path, "test"),
        label_map=LABEL_MAP,
    )
    print(f"✅ Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    config = configs.ObjectDetectionConfig(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=0.005,
        lr_decay_steps=20,
        lr_decay_rate=0.96,
        cosine_lr=True,
    )

    model = object_detector.create(
        train_data=train_data,
        model_spec=spec,
        validation_data=val_data,
        config=config,
    )

    eval_result = model.evaluate(test_data)
    print(f"✅ Test mAP: {eval_result['mAP']:.3f}")
    return model


def export_tflite(model, output_dir: str):
    """Export trained model to TFLite."""
    os.makedirs(output_dir, exist_ok=True)

    model.export(export_dir=output_dir, export_format=object_detector.ExportFormat.TFLITE)

    tflite_files = [
        f for f in os.listdir(output_dir) if f.endswith(".tflite")
    ]
    if tflite_files:
        src = os.path.join(output_dir, tflite_files[0])
        dst = os.path.join(output_dir, "efficientdet-lite0.tflite")
        os.rename(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"✅ Model exported: {dst} ({size_mb:.2f} MB)")
        return dst
    raise FileNotFoundError("No .tflite found in output")


def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientDet-Lite0 for SMD component detection"
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="Roboflow API key (get from https://app.roboflow.com/settings/api)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for trained model (default: output/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Component Counter — Model Training Pipeline")
    print("=" * 60)

    check_deps()
    dataset_path = download_dataset(args.api_key)
    model = train_model(dataset_path)
    model_path = export_tflite(model, args.output)

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE")
    print(f"📦 Model: {model_path}")
    print()
    print("Next steps:")
    print(f"  1. Copy {model_path} to app/src/main/assets/")
    print("  2. Rebuild APK: ./gradlew assembleDebug")
    print("  3. Install on device")
    print("=" * 60)


if __name__ == "__main__":
    main()
