#!/usr/bin/env python3
"""Probe: does the model have ANY real signal on 1.jpeg? Test naive-resize vs
letterbox, sweep thresholds, and report the global max class score."""
import os
import numpy as np
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

LABELS = ["Resistor", "Diode", "Transistor", "Condensator"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "app", "src", "main", "assets", "best_float32.tflite")
SIZE = 416


def letterbox(img, size):
    w, h = img.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = img.resize((nw, nh))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def run(interp, img):
    inp = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    arr = (np.asarray(img, np.float32) / 255.0)[None, ...]
    interp.set_tensor(inp["index"], arr)
    interp.invoke()
    return interp.get_tensor(out_d["index"])[0]  # [8, 3549]


def report(tag, out):
    cls = out[4:, :]                       # [4, 3549]
    gmax = float(cls.max())
    per_class_max = cls.max(axis=1)
    counts = {t: int((cls.max(axis=0) > t).sum()) for t in (0.05, 0.1, 0.25, 0.4)}
    print(f"  [{tag}]  global-max-score={gmax:.3f}")
    print(f"      per-class max: " +
          ", ".join(f"{LABELS[i]}={per_class_max[i]:.3f}" for i in range(4)))
    print(f"      boxes over thr: {counts}")


def main():
    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    for name in ("1.jpeg", "2.jpeg"):
        p = os.path.join(ROOT, "source", name)
        if not os.path.exists(p):
            continue
        base = Image.open(p).convert("RGB")
        print(f"\nIMAGE {name}  orig-size={base.size}")
        report("naive-resize", run(interp, base.resize((SIZE, SIZE))))
        report("letterbox   ", run(interp, letterbox(base, SIZE)))


if __name__ == "__main__":
    main()
