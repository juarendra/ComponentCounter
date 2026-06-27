package com.example.componentcounter.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class BBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val label: String, val score: Float
)

class ObjectDetectorHelper(
    var threshold: Float = 0.5f,
    var iouThreshold: Float = 0.5f,
    val context: Context,
    val listener: DetectorListener?
) {

    private var interpreter: Interpreter? = null
    private val labels = listOf("Resistor", "Diode", "Transistor", "Condensator")
    private val inputSize = 416
    private var isInitialized = false

    companion object {
        val LABEL_MAP = mapOf(
            0 to "Resistor", 1 to "Diode", 2 to "Transistor", 3 to "Condensator"
        )
    }

    init {
        setupDetector()
    }

    private fun setupDetector() {
        try {
            val model = loadModelFile("best_float32.tflite")
            interpreter = Interpreter(model)
            isInitialized = true
            Log.d("ObjectDetector", "YOLOv5n model loaded (input: ${inputSize}x${inputSize})")
        } catch (e: Exception) {
            Log.e("ObjectDetector", "Failed to load model: ${e.message}")
            isInitialized = false
            listener?.onError("Model load failed: ${e.message}")
        }
    }

    private fun loadModelFile(path: String): MappedByteBuffer {
        val fd = context.assets.openFd(path)
        val stream = FileInputStream(fd.fileDescriptor)
        return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun detect(bitmap: Bitmap) {
        if (!isInitialized) {
            listener?.onResults(emptyList(), 0, bitmap.height, bitmap.width)
            return
        }

        val startTime = SystemClock.uptimeMillis()

        // Preprocess: resize to 416x416, normalize
        val input = bitmapToByteBuffer(bitmap)

        // Output buffer: [1, 8, 3549] = float[1][8][3549]
        val output = Array(1) { Array(8) { FloatArray(3549) } }

        interpreter?.run(input, output)
        val inferenceMs = SystemClock.uptimeMillis() - startTime

        // Parse output
        val detections = parseOutput(output[0], bitmap.width, bitmap.height)
        listener?.onResults(detections, inferenceMs, bitmap.height, bitmap.width)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val buffer = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4)
        buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()

        val intValues = IntArray(inputSize * inputSize)
        resized.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in intValues) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            buffer.putFloat((pixel and 0xFF) / 255.0f)
        }
        buffer.rewind()
        return buffer
    }

    private fun parseOutput(output: Array<FloatArray>, imgW: Int, imgH: Int): List<BBox> {
        // output[8][3549]: 8 = [cx, cy, w, h, p0, p1, p2, p3], 3549 detections
        val data = Array(3549) { i -> FloatArray(8) { j -> output[j][i] } }
        val candidates = mutableListOf<BBox>()

        for (i in data.indices) {
            val row = data[i]
            val cx = row[0]
            val cy = row[1]
            val w = row[2]
            val h = row[3]

            var maxScore = 0f
            var maxIdx = -1
            for (c in 4..7) {
                val s = sigmoid(row[c])
                if (s > maxScore) { maxScore = s; maxIdx = c - 4 }
            }

            if (maxScore >= threshold && maxIdx in labels.indices) {
                // Denormalize to image coordinates
                val x1 = max(0f, (cx - w / 2) * imgW)
                val y1 = max(0f, (cy - h / 2) * imgH)
                val x2 = min(imgW.toFloat(), (cx + w / 2) * imgW)
                val y2 = min(imgH.toFloat(), (cy + h / 2) * imgH)
                candidates.add(BBox(x1, y1, x2, y2, labels[maxIdx], maxScore))
            }
        }

        return nms(candidates)
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

    private fun nms(boxes: List<BBox>): List<BBox> {
        val sorted = boxes.sortedByDescending { it.score }
        val kept = mutableListOf<BBox>()
        for (box in sorted) {
            var overlaps = false
            for (existing in kept) {
                if (iou(box, existing) > iouThreshold) { overlaps = true; break }
            }
            if (!overlaps) kept.add(box)
        }
        return kept
    }

    private fun iou(a: BBox, b: BBox): Float {
        val ix1 = max(a.x1, b.x1)
        val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2)
        val iy2 = min(a.y2, b.y2)
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
