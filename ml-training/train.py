#!/usr/bin/env python3
"""Train YOLOv8n on smdComponents and export a TFLite model matching the
runtime contract (416 input, anchor-free [1,8,3549]). See ml-training/MODEL_IO.md.

Usage:
  pip install -r requirements.txt
  python train.py --api_key YOUR_ROBOFLOW_KEY
  # then copy the exported *_float32.tflite to app/src/main/assets/best_float32.tflite
"""
import argparse
from roboflow import Roboflow
from ultralytics import YOLO

CLASSES = ["Resistor", "Diode", "Transistor", "Condensator"]
IMGSZ = 416
EPOCHS = 100


def download(api_key: str) -> str:
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace("dainius").project("smdcomponents").version(2).download("yolov8")
    return f"{ds.location}/data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True, help="Roboflow API key")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    data_yaml = download(args.api_key)
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, imgsz=IMGSZ, epochs=args.epochs, batch=16)
    path = model.export(format="tflite", imgsz=IMGSZ)
    print(f"Exported: {path}")
    print("Copy the *_float32.tflite to app/src/main/assets/best_float32.tflite, then rebuild.")


if __name__ == "__main__":
    main()
