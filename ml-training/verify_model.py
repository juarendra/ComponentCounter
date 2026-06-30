#!/usr/bin/env python3
"""
Ground-truth verification for the runtime TFLite model.

Loads app/src/main/assets/best_float32.tflite, prints the REAL input/output
tensor shapes, runs the sample images through it, and decodes the output BOTH
ways (anchor-based YOLOv5 with objectness, and anchor-free YOLOv5u/v8
channel-first) so we can see which interpretation yields plausible boxes.

Usage:
  python ml-training/verify_model.py
"""
import os
import sys
import numpy as np
from PIL import Image

try:
    from tensorflow.lite.python.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter  # type: ignore

LABELS = ["Resistor", "Diode", "Transistor", "Condensator"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "app", "src", "main", "assets", "best_float32.tflite")
IMAGES = [os.path.join(ROOT, "source", "1.jpeg"),
          os.path.join(ROOT, "source", "2.jpeg")]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ba = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ba - inter)


def nms(boxes, iou_thr=0.5):
    boxes = sorted(boxes, key=lambda x: -x[4])
    kept = []
    for box in boxes:
        if all(iou(box, k) <= iou_thr for k in kept):
            kept.append(box)
    return kept


def decode_anchor_free(out, conf_thr):
    """out shape [C, N] channel-first: 0..3 = cx,cy,w,h ; 4.. = class conf."""
    c, n = out.shape
    num_cls = c - 4
    boxes = []
    for i in range(n):
        cls_scores = out[4:, i]
        k = int(np.argmax(cls_scores))
        s = float(cls_scores[k])
        if s < conf_thr:
            continue
        cx, cy, w, h = out[0, i], out[1, i], out[2, i], out[3, i]
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s, k])
    return boxes, num_cls


def decode_anchor_based(out, conf_thr):
    """out shape [N, 5+C]: 0..3 = cx,cy,w,h ; 4 = obj ; 5.. = class conf."""
    n, c = out.shape
    num_cls = c - 5
    boxes = []
    for i in range(n):
        obj = float(out[i, 4])
        cls_scores = out[i, 5:]
        k = int(np.argmax(cls_scores))
        s = obj * float(cls_scores[k])
        if s < conf_thr:
            continue
        cx, cy, w, h = out[i, 0], out[i, 1], out[i, 2], out[i, 3]
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s, k])
    return boxes, num_cls


def summarize(boxes, num_cls, conf_thr):
    kept = nms(boxes)
    coord_max = max((max(b[:4]) for b in boxes), default=0.0)
    print(f"      candidates>{conf_thr}: {len(boxes):5d}   after NMS: {len(kept):3d}   "
          f"max-coord: {coord_max:8.2f}   classes: {num_cls}")
    from collections import Counter
    cnt = Counter(LABELS[b[5]] if b[5] < len(LABELS) else f"cls{b[5]}" for b in kept)
    if kept:
        print(f"      kept by label: {dict(cnt)}")
        top = sorted(kept, key=lambda x: -x[4])[:3]
        for b in top:
            lbl = LABELS[b[5]] if b[5] < len(LABELS) else f"cls{b[5]}"
            print(f"        {lbl:12s} score={b[4]:.3f} box=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f})")


def main():
    if not os.path.exists(MODEL):
        sys.exit(f"Model not found: {MODEL}")
    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    print("=" * 70)
    print("INPUT")
    print(f"  shape={inp['shape'].tolist()} dtype={inp['dtype'].__name__} "
          f"quant={inp['quantization']}")
    in_h, in_w = int(inp["shape"][1]), int(inp["shape"][2])
    print("OUTPUTS")
    for o in outs:
        print(f"  name={o['name']!r} shape={o['shape'].tolist()} "
              f"dtype={o['dtype'].__name__} quant={o['quantization']}")
    print("=" * 70)

    conf_thr = 0.25
    for img_path in IMAGES:
        if not os.path.exists(img_path):
            print(f"  (skip missing {img_path})")
            continue
        print(f"\nIMAGE {os.path.basename(img_path)}  (model input {in_w}x{in_h})")
        img = Image.open(img_path).convert("RGB").resize((in_w, in_h))
        arr = (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]
        interp.set_tensor(inp["index"], arr)
        interp.run = getattr(interp, "invoke")
        interp.invoke()

        out = interp.get_tensor(outs[0]["index"])[0]  # drop batch
        print(f"  raw output (no batch) shape={out.shape}")

        # Decide layout by shape. Channel-first if dim0 small (e.g. 8 or 9).
        if out.shape[0] <= out.shape[1]:
            ch = out.shape[0]
            print(f"  -> channel-first [C={ch}, N={out.shape[1]}]")
            if ch == 4 + len(LABELS):
                print("    DECODE anchor-free (C = 4 + numClasses):")
                b, nc = decode_anchor_free(out, conf_thr); summarize(b, nc, conf_thr)
            elif ch == 5 + len(LABELS):
                print("    DECODE anchor-based, transposed to [N, 5+C]:")
                b, nc = decode_anchor_based(out.T, conf_thr); summarize(b, nc, conf_thr)
            else:
                print(f"    !! unexpected channel count {ch}; trying both")
                b, nc = decode_anchor_free(out, conf_thr); summarize(b, nc, conf_thr)
        else:
            cols = out.shape[1]
            print(f"  -> box-first [N={out.shape[0]}, C={cols}]")
            if cols == 5 + len(LABELS):
                print("    DECODE anchor-based (C = 5 + numClasses, has objectness):")
                b, nc = decode_anchor_based(out, conf_thr); summarize(b, nc, conf_thr)
            elif cols == 4 + len(LABELS):
                print("    DECODE anchor-free (C = 4 + numClasses, no objectness):")
                b, nc = decode_anchor_free(out.T, conf_thr); summarize(b, nc, conf_thr)
            else:
                print(f"    !! unexpected col count {cols}; trying anchor-based")
                b, nc = decode_anchor_based(out, conf_thr); summarize(b, nc, conf_thr)


if __name__ == "__main__":
    main()
