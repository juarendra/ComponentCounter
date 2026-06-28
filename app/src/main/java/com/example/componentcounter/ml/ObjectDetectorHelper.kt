package com.example.componentcounter.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

data class BBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val label: String, val score: Float
)

class ObjectDetectorHelper(
    val context: Context,
    val listener: DetectorListener?
) {
    private var interpreter: org.tensorflow.lite.Interpreter? = null
    private val inputSize = 416
    private val labels = listOf("Resistor", "Diode", "Transistor", "Condensator")
    private val threshold = 0.25f

    // Anchor-free YOLO (v5u/v8) output: [1, 4+numClasses, numBoxes]
    // numBoxes = 52*52 + 26*26 + 13*13 = 3549 for 416 input
    private val numClasses = labels.size
    private val numOutputs = 4 + numClasses
    private val numBoxes = (inputSize / 8) * (inputSize / 8) +
        (inputSize / 16) * (inputSize / 16) +
        (inputSize / 32) * (inputSize / 32)

    companion object {
        val LABEL_MAP = mapOf(
            0 to "Resistor", 1 to "Diode", 2 to "Transistor", 3 to "Condensator"
        )
    }

    init {
        try {
            val model = loadModelFile("best_float32.tflite")
            interpreter = org.tensorflow.lite.Interpreter(model)
        } catch (e: Exception) {
            listener?.onError("Model load failed: ${e.message}")
        }
    }

    private fun loadModelFile(path: String): MappedByteBuffer {
        val fd = context.assets.openFd(path)
        val stream = FileInputStream(fd.fileDescriptor)
        return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun detect(bitmap: Bitmap) {
        val interpreter = interpreter
        if (interpreter == null) {
            listener?.onResults(emptyList(), 0, bitmap.height, bitmap.width)
            return
        }

        val startTime = SystemClock.uptimeMillis()

        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255.0f)
            inputBuffer.putFloat((p and 0xFF) / 255.0f)
        }
        inputBuffer.rewind()

        // Output: [1, numOutputs, numBoxes] — anchor-free, channel-first.
        // Channels 0..3 = cx,cy,w,h ; channels 4.. = per-class confidence (already 0..1).
        val output = Array(1) { Array(numOutputs) { FloatArray(numBoxes) } }
        try {
            interpreter.run(inputBuffer, output)
        } catch (e: Exception) {
            listener?.onError("Inference failed: ${e.message}")
            listener?.onResults(emptyList(), 0, bitmap.height, bitmap.width)
            return
        }

        val rawOutput = output[0]
        val candidates = mutableListOf<BBox>()

        for (idx in 0 until numBoxes) {
            // Best class (scores already passed through sigmoid in the exported graph)
            var maxScore = 0f
            var maxClsIdx = -1
            for (c in 0 until numClasses) {
                val s = rawOutput[4 + c][idx]
                if (s > maxScore) { maxScore = s; maxClsIdx = c }
            }
            if (maxScore < threshold || maxClsIdx < 0) continue

            // Box channels. Ultralytics TFLite export emits normalized 0..1 xywh;
            // guard in case a build emits pixel (0..inputSize) coords instead.
            var cx = rawOutput[0][idx]
            var cy = rawOutput[1][idx]
            var w = rawOutput[2][idx]
            var h = rawOutput[3][idx]
            if (cx > 1.5f || cy > 1.5f || w > 1.5f || h > 1.5f) {
                cx /= inputSize; cy /= inputSize; w /= inputSize; h /= inputSize
            }

            val x1 = (cx - w / 2) * bitmap.width
            val y1 = (cy - h / 2) * bitmap.height
            val x2 = (cx + w / 2) * bitmap.width
            val y2 = (cy + h / 2) * bitmap.height

            if (x2 > x1 && y2 > y1) {
                candidates.add(BBox(
                    max(0f, x1), max(0f, y1),
                    min(bitmap.width.toFloat(), x2), min(bitmap.height.toFloat(), y2),
                    labels[maxClsIdx], maxScore
                ))
            }
        }

        // NMS
        val results = nms(candidates)
        val inferenceMs = SystemClock.uptimeMillis() - startTime
        listener?.onResults(results, inferenceMs, bitmap.height, bitmap.width)
    }

    private fun nms(boxes: List<BBox>): List<BBox> {
        val sorted = boxes.sortedByDescending { it.score }
        val kept = mutableListOf<BBox>()
        for (box in sorted) {
            var overlaps = false
            for (existing in kept) {
                if (iou(box, existing) > 0.5f) { overlaps = true; break }
            }
            if (!overlaps) kept.add(box)
        }
        return kept
    }

    private fun iou(a: BBox, b: BBox): Float {
        val ix1 = max(a.x1, b.x1); val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2); val iy2 = min(a.y2, b.y2)
        if (ix1 >= ix2 || iy1 >= iy2) return 0f
        val iArea = (ix2 - ix1) * (iy2 - iy1)
        val aArea = (a.x2 - a.x1) * (a.y2 - a.y1)
        val bArea = (b.x2 - b.x1) * (b.y2 - b.y1)
        return iArea / (aArea + bArea - iArea)
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(results: List<BBox>, inferenceTime: Long, imageHeight: Int, imageWidth: Int)
    }
}
