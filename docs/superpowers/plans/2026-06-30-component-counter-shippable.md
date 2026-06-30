# Component Counter — Shippable App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix detection (camera conversion + preprocessing), add image-based tests, replace the dead training pipeline with a reproducible Ultralytics YOLOv8 flow, and ship a signed APK via the existing CI — orchestrated PR-per-phase.

**Architecture:** On-device YOLOv5n/v8 TFLite anchor-free detector (verified `[1,8,3549]`, 416 input, letterbox preprocessing) behind an MVVM Compose UI. Camera frames go RGBA_8888 → `ImageProxy.toBitmap()` → letterbox → interpreter → decode → NMS → StateFlow → overlay.

**Tech Stack:** Kotlin 2.3 / Jetpack Compose / CameraX 1.4 / TFLite 2.16 / Ultralytics YOLOv8 (training) / GitHub Actions CI.

**Ground truth:** see `ml-training/MODEL_IO.md` (measured). Author/pusher: `juarendra`. PR per phase.

---

## File Structure

| File | Responsibility | Phase |
|------|----------------|-------|
| `ml-training/verify_model.py`, `probe_model.py`, `MODEL_IO.md` | Model ground-truth (DONE) | 0 |
| `app/.../ml/ObjectDetectorHelper.kt` | Letterbox preprocess + verified decode + un-letterbox mapping | 1 |
| `app/.../ui/camera/CameraScreen.kt` | RGBA_8888 analyzer + built-in `toBitmap()` (delete broken YUV path) | 1 |
| `app/src/test/.../ml/DetectionImageTest.kt` | Image-based detection test (no camera) | 1 |
| `app/src/test/resources/test_tape.jpg` | Bundled test image (copy of `source/1.jpeg`) | 1 |
| `CLAUDE.md`, `README.md` | Fix drift (NMS location, YOLO not EfficientDet) | 2 |
| `app/src/main/assets/mobilenetv1.tflite` | DELETE (unused 4 MB) | 2 |
| `ml-training/train.py`, `train_yolo.ipynb`, `requirements.txt`, `README.md`, `train_efficientdet.ipynb` | Replace EfficientDet → YOLOv8 | 3 |
| `.github/workflows/*` | Verify gates green; no change expected | 4 |

---

## PHASE 0 — Model Ground Truth ✅ DONE

Artifacts committed: `ml-training/verify_model.py`, `ml-training/probe_model.py`, `ml-training/MODEL_IO.md`. Findings: output `[1,8,3549]` anchor-free (decode shape was already correct); **letterbox required** (10–16× confidence vs naive resize); current model is weak (peak 0.263 on real photo) → retrain needed in Phase 3.

---

## PHASE 1 — Fix Detection (branch `fix/detection-pipeline`)

### Task 1.1 — Letterbox preprocessing + un-letterbox output in ObjectDetectorHelper

**Files:**
- Modify: `app/src/main/java/com/example/componentcounter/ml/ObjectDetectorHelper.kt`
- Test: `app/src/test/java/com/example/componentcounter/ml/LetterboxTest.kt`

- [ ] **Step 1: Write failing test for letterbox geometry**

Create `app/src/test/java/com/example/componentcounter/ml/LetterboxTest.kt`:

```kotlin
package com.example.componentcounter.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class LetterboxTest {
    @Test fun landscape_scales_by_width_and_pads_top_bottom() {
        // src 800x400 into 416 -> scale 0.52, nw=416, nh=208, padX=0, padY=104
        val lb = Letterbox.compute(srcW = 800, srcH = 400, dst = 416)
        assertEquals(0.52f, lb.scale, 1e-4f)
        assertEquals(0f, lb.padX, 1e-4f)
        assertEquals(104f, lb.padY, 1e-4f)
    }

    @Test fun maps_letterboxed_norm_box_back_to_source_pixels() {
        val lb = Letterbox.compute(srcW = 800, srcH = 400, dst = 416)
        // a box centered in the letterboxed image -> center of source
        val (x, y) = lb.toSource(0.5f, 0.5f)
        assertEquals(400f, x, 0.5f)
        assertEquals(200f, y, 0.5f)
    }
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `./gradlew test --tests "*LetterboxTest*"`
Expected: FAIL — `Letterbox` unresolved.

- [ ] **Step 3: Add the `Letterbox` helper to ObjectDetectorHelper.kt**

Add near the top of `ObjectDetectorHelper.kt` (after `BBox`):

```kotlin
/** Aspect-preserving resize geometry. `scale`/`padX`/`padY` map dst(416)↔src pixels. */
data class Letterbox(val scale: Float, val padX: Float, val padY: Float, val dst: Int) {
    /** Map a normalized (0..1) coord in the letterboxed dst image to source pixels. */
    fun toSource(nx: Float, ny: Float): Pair<Float, Float> {
        val px = nx * dst - padX
        val py = ny * dst - padY
        return Pair(px / scale, py / scale)
    }
    companion object {
        fun compute(srcW: Int, srcH: Int, dst: Int): Letterbox {
            val scale = minOf(dst.toFloat() / srcW, dst.toFloat() / srcH)
            val padX = (dst - srcW * scale) / 2f
            val padY = (dst - srcH * scale) / 2f
            return Letterbox(scale, padX, padY, dst)
        }
    }
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `./gradlew test --tests "*LetterboxTest*"`
Expected: PASS (2 tests).

- [ ] **Step 5: Rewrite `detect()` preprocessing + output mapping to use letterbox**

In `ObjectDetectorHelper.detect()`, replace the resize/input-fill block and the box-mapping block. Replace lines that build `resized`/`inputBuffer` with:

```kotlin
        val lb = Letterbox.compute(bitmap.width, bitmap.height, inputSize)
        val canvas = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(canvas).apply {
            drawColor(android.graphics.Color.rgb(114, 114, 114))
            val nw = bitmap.width * lb.scale
            val nh = bitmap.height * lb.scale
            val dstRect = android.graphics.RectF(lb.padX, lb.padY, lb.padX + nw, lb.padY + nh)
            drawBitmap(bitmap, null, dstRect, null)
        }
        val inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        canvas.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255.0f)
            inputBuffer.putFloat((p and 0xFF) / 255.0f)
        }
        inputBuffer.rewind()
```

Then replace the per-box coordinate mapping (the `cx/cy/w/h` → `x1..y2` block) with letterbox-aware mapping:

```kotlin
            // Box channels are normalized 0..1 in the 416 letterboxed space.
            val cx = rawOutput[0][idx]
            val cy = rawOutput[1][idx]
            val w = rawOutput[2][idx]
            val h = rawOutput[3][idx]
            val (sx1, sy1) = lb.toSource(cx - w / 2, cy - h / 2)
            val (sx2, sy2) = lb.toSource(cx + w / 2, cy + h / 2)
            val x1 = sx1; val y1 = sy1; val x2 = sx2; val y2 = sy2
```

Lower `threshold` to `0.15f` (temporary, weak current model — see MODEL_IO.md). Delete the now-stale "guard in case pixel coords" comment block.

- [ ] **Step 6: Run full unit suite, verify green**

Run: `./gradlew test`
Expected: PASS (existing 11 + 2 new).

- [ ] **Step 7: Commit**

```bash
git add app/src/main/java/com/example/componentcounter/ml/ObjectDetectorHelper.kt app/src/test/java/com/example/componentcounter/ml/LetterboxTest.kt
git commit -m "fix: letterbox preprocessing + un-letterbox box mapping in detector"
```

### Task 1.2 — Fix camera frame conversion (green corruption + garbage count)

**Files:**
- Modify: `app/src/main/java/com/example/componentcounter/ui/camera/CameraScreen.kt`

- [ ] **Step 1: Set RGBA_8888 output on the analyzer**

In `CameraPreviewWithDetection`, change the `ImageAnalysis.Builder()` to request RGBA_8888:

```kotlin
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { proxy ->
                    proxy.use {
                        val bmp = proxy.toBitmap()
                        detector.detect(rotateBitmap(bmp, proxy.imageInfo.rotationDegrees))
                    }
                } }
```

- [ ] **Step 2: Delete the broken manual `toBitmap()` extension + unused imports**

Remove the entire `private fun androidx.camera.core.ImageProxy.toBitmap(): Bitmap?` function (lines ~190–208) so the call resolves to CameraX's built-in `ImageProxy.toBitmap()` (correct for RGBA_8888). Remove now-unused imports: `android.graphics.BitmapFactory`, `android.graphics.ImageFormat`, `android.graphics.Rect`, `android.graphics.YuvImage`, `java.io.ByteArrayOutputStream`.

- [ ] **Step 3: Build to verify compile + resolution of built-in toBitmap**

Run: `./gradlew :app:compileDebugKotlin`
Expected: BUILD SUCCESSFUL (built-in `ImageProxy.toBitmap()` available in camera-core 1.4).

- [ ] **Step 4: Commit**

```bash
git add app/src/main/java/com/example/componentcounter/ui/camera/CameraScreen.kt
git commit -m "fix: use RGBA_8888 + built-in toBitmap, remove broken YUV path (green-screen bug)"
```

### Task 1.3 — Image-based detection test (validate detection without camera)

**Files:**
- Create: `app/src/test/java/com/example/componentcounter/ml/DetectionImageTest.kt`
- Create: `app/src/test/resources/test_tape.jpg` (copy of `source/1.jpeg`)

- [ ] **Step 1: Add the test resource**

```bash
mkdir -p app/src/test/resources
cp source/1.jpeg app/src/test/resources/test_tape.jpg
```

- [ ] **Step 2: Write the Robolectric image test**

Create `app/src/test/java/com/example/componentcounter/ml/DetectionImageTest.kt`:

```kotlin
package com.example.componentcounter.ml

import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class DetectionImageTest {
    @Test fun runs_on_still_image_without_crash_and_returns_boxes_in_bounds() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        val stream = javaClass.classLoader!!.getResourceAsStream("test_tape.jpg")
        val bmp = BitmapFactory.decodeStream(stream)
        var got: List<BBox>? = null
        val helper = ObjectDetectorHelper(ctx, object : ObjectDetectorHelper.DetectorListener {
            override fun onError(error: String) { throw AssertionError(error) }
            override fun onResults(results: List<BBox>, inferenceTime: Long, imageHeight: Int, imageWidth: Int) {
                got = results
            }
        })
        helper.detect(bmp)
        val results = got ?: error("no results callback")
        // All boxes must be in-bounds and well-formed (no NaN, x2>x1).
        for (b in results) {
            assertTrue(b.x2 > b.x1 && b.y2 > b.y1)
            assertTrue(b.x1 >= 0f && b.y1 >= 0f)
            assertTrue(b.x2 <= bmp.width.toFloat() && b.y2 <= bmp.height.toFloat())
            assertTrue(b.label in listOf("Resistor", "Diode", "Transistor", "Condensator"))
        }
    }
}
```

Note: asserts correctness/in-bounds rather than a fixed count, because the current weak model under-detects (a count assertion becomes valid after Phase 3 retrain — add then).

- [ ] **Step 3: Run the test**

Run: `./gradlew test --tests "*DetectionImageTest*"`
Expected: PASS (model loads from assets, decode runs, boxes in-bounds).

- [ ] **Step 4: Run detekt + lint**

Run: `./gradlew detekt lint`
Expected: no issues.

- [ ] **Step 5: Commit + open PR**

```bash
git add app/src/test/java/com/example/componentcounter/ml/DetectionImageTest.kt app/src/test/resources/test_tape.jpg
git commit -m "test: add image-based detection test (camera-free validation)"
git push -u origin fix/detection-pipeline
gh pr create --fill --title "fix: detection pipeline (letterbox + camera conversion + image test)"
```

- [ ] **Step 6: Verify CI green, then merge**

Wait for `review.yml` + `qa.yml` green. Merge to `main`.

---

## PHASE 2 — Polish & Drift Cleanup (branch `chore/cleanup-drift`)

### Task 2.1 — Remove unused model + fix docs drift

**Files:**
- Delete: `app/src/main/assets/mobilenetv1.tflite`
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Delete unused model**

```bash
git rm app/src/main/assets/mobilenetv1.tflite
```

- [ ] **Step 2: Fix CLAUDE.md drift**

In `CLAUDE.md`: change the data-flow line that says NMS dedup happens in `CounterViewModel` to state NMS runs in `ObjectDetectorHelper`. Update gotcha #2 (TFLite Model) to describe the real model: `best_float32.tflite`, YOLOv5/v8 anchor-free, input 416, output `[1,8,3549]`, letterbox preprocessing, threshold 0.15 (interim) — and remove the "efficientdet-lite0" / 320 / 0.5 / 50-max claims. Point training reference to the YOLO notebook (Phase 3).

- [ ] **Step 3: Fix README.md training reference**

In `README.md`: replace EfficientDet/Model-Maker training mentions with the YOLOv8 flow and the `ml-training/MODEL_IO.md` contract.

- [ ] **Step 4: Build to confirm assets removal is clean**

Run: `./gradlew :app:assembleDebug`
Expected: BUILD SUCCESSFUL (no reference to mobilenetv1).

- [ ] **Step 5: Commit + PR + merge**

```bash
git add -A
git commit -m "chore: remove unused mobilenetv1 model, fix CLAUDE.md/README drift"
git push -u origin chore/cleanup-drift
gh pr create --fill --title "chore: cleanup unused model + doc drift"
```
Verify CI green → merge.

---

## PHASE 3 — Reproducible Training Pipeline (branch `feat/yolo-training`)

### Task 3.1 — Replace EfficientDet/Model-Maker with Ultralytics YOLOv8

**Files:**
- Delete: `ml-training/train_efficientdet.ipynb`
- Rewrite: `ml-training/train.py`, `ml-training/requirements.txt`, `ml-training/README.md`
- Create: `ml-training/train_yolo.ipynb` (Colab)

- [ ] **Step 1: Rewrite `requirements.txt`**

```
ultralytics>=8.3.0
roboflow>=1.1.0
onnx>=1.16.0
tensorflow>=2.16.0
```

- [ ] **Step 2: Rewrite `train.py`**

```python
#!/usr/bin/env python3
"""Train YOLOv8n on smdComponents and export a TFLite model matching the
runtime contract (416 input, anchor-free [1,8,3549]). See ml-training/MODEL_IO.md.

Usage:
  pip install -r requirements.txt
  python train.py --api_key YOUR_ROBOFLOW_KEY
  # then copy runs/.../best_float32.tflite to app/src/main/assets/
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
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    data_yaml = download(args.api_key)
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, imgsz=IMGSZ, epochs=args.epochs, batch=16)
    # Export float32 TFLite (matches best_float32.tflite runtime contract).
    path = model.export(format="tflite", imgsz=IMGSZ)
    print(f"Exported: {path}")
    print("Copy the *_float32.tflite to app/src/main/assets/best_float32.tflite, then rebuild.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Rewrite `ml-training/README.md`**

Document: Roboflow key → `python train.py --api_key ...` → copy `best_float32.tflite` to assets → rebuild. State the verified contract (416, `[1,8,3549]`, letterbox) and link `MODEL_IO.md`. Remove all EfficientDet / tflite-model-maker text.

- [ ] **Step 4: Create `train_yolo.ipynb`**

A 1-cell Colab notebook mirroring `train.py` (pip install, set `ROBOFLOW_KEY`, run train+export, download the float32 tflite). Replace the Colab badge target with `train_yolo.ipynb`.

- [ ] **Step 5: Remove dead notebook**

```bash
git rm ml-training/train_efficientdet.ipynb
```

- [ ] **Step 6: Smoke-check the script parses (no training run here)**

Run: `python -c "import ast; ast.parse(open('ml-training/train.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit + PR + merge**

```bash
git add -A
git commit -m "feat: replace EfficientDet pipeline with Ultralytics YOLOv8 training + export"
git push -u origin feat/yolo-training
gh pr create --fill --title "feat: reproducible YOLOv8 training pipeline"
```
Verify CI green → merge.

> **User action (out of band):** run `train.py` with a Roboflow key (Colab or local GPU), drop the new `best_float32.tflite` into assets via a follow-up PR. After that model lands, raise the detector threshold back to 0.25 and add a count assertion to `DetectionImageTest`.

---

## PHASE 4 — Orchestration & Release

### Task 4.1 — Verify full pipeline + cut release

- [ ] **Step 1: Confirm all phase PRs merged to `main`, CI green**

Run: `gh run list --branch main --limit 5`
Expected: latest `qa.yml` + `release.yml` success.

- [ ] **Step 2: Confirm release APK artifact built**

Run: `gh release list`
Expected: a `ComponentCounter-vX.Y.Z` release with signed APK (version auto-bumped from `fix:`/`feat:` commits).

- [ ] **Step 3: Manual device validation (user)**

Install the debug APK on the physical phone. Confirm: no green corruption, count is plausible (not 723), boxes track components. Record result.

- [ ] **Step 4: Tag completion**

If device validation passes, the app is shippable. If detection accuracy still weak, that is expected until the Phase-3-retrained model lands — not a code bug.

---

## Self-Review Notes

- **Spec coverage:** Phase 0 (ground truth) ✅, Phase 1 (detector+camera+image test) ✅, Phase 2 (polish/drift/cleanup) ✅, Phase 3 (modern training) ✅, Phase 4 (orchestration/release) ✅.
- **Decode correctness:** confirmed shape was already right; plan does NOT re-touch decode shape, only preprocessing + coordinate mapping (the real bugs).
- **Type consistency:** `Letterbox.compute`/`toSource`, `BBox`, `DetectorListener.onResults` signatures match existing code.
- **Known residual:** current model is weak; usable accuracy depends on the Phase 3 retrain (flagged explicitly, not hidden).
