# Component Counter — Path to a Shippable, Usable App

**Date:** 2026-06-30
**Branch base:** `fix/detector-anchor-free`
**Goal:** Turn the project into an Android app the user can "just use" — detection that works correctly, an easy build/release flow, and a reproducible training pipeline — orchestrated through the existing agent pipeline (Builder → Reviewer → QA → Release).

---

## Problem Statement

The app builds and runs on-device, but **detection is wrong** ("jalan tapi deteksi salah"). Root cause: the TFLite model output decoding in `ObjectDetectorHelper.kt` is **guesswork that was never verified against the actual model**. Git history shows 5 consecutive commits thrashing the decode logic (EfficientDet → YOLOv5 anchor-based → anchor-free) with no ground-truth verification. Code comments literally read `// guard in case a build emits ...`.

Secondary problems (documentation/pipeline drift):
- `ml-training/` documents an **EfficientDet-Lite0 + `tflite-model-maker`** pipeline that is **dead** (abandoned 2023, requires exact TF 2.16, will not install on modern Python). The app actually runs an Ultralytics YOLOv5n export (`best_float32.tflite`, 416×416). Two incompatible pipelines.
- `CLAUDE.md` claims NMS lives in `CounterViewModel` and that there are "NMS logic tests"; NMS actually lives in `ObjectDetectorHelper`. Drift.
- `app/src/main/assets/mobilenetv1.tflite` (4 MB) is unused dead weight.

## Success Criteria

1. Running `source/1.jpeg` / `source/2.jpeg` through the model produces **correct, sane bounding boxes** for the 4 classes (Resistor, Diode, Transistor, Condensator).
2. The Kotlin decoder matches the **verified** model output format — no guessing.
3. An **image-based** detection test exists so detection can be validated without the live camera (per user request).
4. The app installs on a physical Android phone and shows correct live detection + counts.
5. A **reproducible** training pipeline (Ultralytics YOLOv8 → TFLite) replaces the dead EfficientDet path.
6. Merge to `main` auto-builds a signed APK via existing CI; versioning is automatic from commit type.

## Constraints / Decisions

- **Local ML tooling:** full `tensorflow` installed locally (also enables local training later).
- **Git flow:** one branch + PR per phase, CI-reviewed before merge (matches CLAUDE.md pipeline).
- **Test device:** physical Android phone; validate detection on still images first, then live camera.
- **Build OS:** AGP 9.0.1 has Windows SDK-discovery bugs (CLAUDE.md gotcha #7) → rely on GitHub Actions (Ubuntu) for authoritative builds; local Windows builds best-effort.

---

## Architecture of the Work

### Phase 0 — Establish Model Ground Truth ⭐ (gates everything)

Build a Python verification harness (`ml-training/verify_model.py`):
- Load `app/src/main/assets/best_float32.tflite` via the TFLite interpreter.
- Print real **input** tensor shape/dtype and **output** tensor shape/dtype.
- Run `source/1.jpeg` and `source/2.jpeg` through it.
- Decode the output **both ways** (anchor-based `[1, N, 5+C]` with objectness, and anchor-free `[1, 4+C, N]` channel-first), apply NMS, and report which interpretation yields plausible boxes.
- Emit a written **decode spec**: exact shape, channel layout, whether objectness exists, coordinate normalization (0..1 vs pixel), and the input size the model truly expects.

**Output artifact:** a short `ml-training/MODEL_IO.md` documenting the verified contract. This is the single source of truth Phase 1 implements against.

### Phase 1 — Fix the Kotlin Decoder

- Rewrite `ObjectDetectorHelper.detect()` decode to match the verified contract exactly (correct input size, channel order, objectness handling if present, coordinate scaling).
- Add a JVM/Robolectric **image-based test**: load a bundled test image, run detection, assert a non-zero, expected-range detection count and label set. Lets detection be validated in CI without a camera.
- Keep NMS where it is (`ObjectDetectorHelper`); update `CLAUDE.md` to stop claiming NMS is in the ViewModel.

### Phase 2 — App Polish ("tinggal pakai")

- Verify the camera overlay maps boxes correctly to preview coordinates (aspect/rotation).
- Tune confidence + NMS thresholds based on Phase 0 evidence.
- Remove unused `mobilenetv1.tflite` (saves 4 MB).
- Fix `README.md` and `CLAUDE.md` drift so docs describe the real (YOLO) pipeline.

### Phase 3 — Modern Training Pipeline

- Replace `train.py` / `train_efficientdet.ipynb` / `requirements.txt` with an **Ultralytics YOLOv8 → TFLite** flow:
  - Colab notebook + local `train.py`: pull smdComponents from Roboflow, train YOLOv8n, export `best_float32.tflite` (or INT8), matching the runtime's expected I/O (verified in Phase 0).
  - Document the exact export command and where to drop the file.
- This makes "training datasheet" reproducible end-to-end: dataset → train → export → drop in assets → rebuild.

### Phase 4 — Agent Orchestration & Release

- Run each phase's change through the pipeline: Builder (code) → Code Reviewer (detekt + lint) → QA (unit tests) → Release.
- Confirm `qa.yml` / `review.yml` pass on each PR; confirm `release.yml` produces a signed `ComponentCounter-vX.Y.Z.apk` on merge to `main`.
- Use conventional-commit prefixes so CI bumps the version correctly (`fix:` patch, `feat:` minor).

---

## Error Handling & Risks

- **Model shape ambiguity:** mitigated by Phase 0 — we measure, not guess.
- **Windows build flakiness (AGP 9.0.1):** authoritative builds run in CI (Ubuntu); local builds are best-effort only.
- **`tflite-model-maker` is dead:** removed entirely in Phase 3; not revived.
- **Signing secrets:** release requires `KEYSTORE_BASE64` + signing env vars already configured in CI; if absent, release job fails loudly (out of scope to provision new keys unless user asks).
- **Camera coordinate mapping:** if overlay misaligns after decode fix, treat as a separate Phase 2 bug with its own investigation.

## Testing Strategy

- **Unit/Robolectric:** existing 8 ViewModel + 3 detector tests, plus new image-based detection test (Phase 1).
- **Ground-truth check:** Python harness diff against still images (Phase 0).
- **Manual:** install debug APK on the physical phone, confirm live detection + count + CSV export (Phase 2/4).
- **CI gates:** detekt + lint + unit tests green on every PR.

## Out of Scope (YAGNI)

- Persisting snapshots across process death (currently in-memory; not requested).
- New component classes beyond the existing 4.
- Provisioning new signing keys.
- Any DI framework or architecture rewrite.
