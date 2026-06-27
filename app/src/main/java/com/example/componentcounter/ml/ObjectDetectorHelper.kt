package com.example.componentcounter.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
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
    private val threshold = 0.5f

    // YOLOv5 anchor configuration for 416x416
    private val strides = intArrayOf(8, 16, 32)
    private val anchors = arrayOf(
        arrayOf(intArrayOf(10, 13), intArrayOf(16, 30), intArrayOf(33, 23)),
        arrayOf(intArrayOf(30, 61), intArrayOf(62, 45), intArrayOf(59, 119)),
        arrayOf(intArrayOf(116, 90), intArrayOf(156, 198), intArrayOf(373, 326))
    )
    private val numClasses = labels.size
    private val numOutputs = 4 + numClasses

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

        // Output: [1, 8, 3549] — 3549 = (13*13 + 26*26 + 52*52) * 3 anchors
        val totalDetections = (13*13 + 26*26 + 52*52) * 3
        val output = Array(1) { Array(numOutputs) { FloatArray(totalDetections) } }
        interpreter.run(inputBuffer, output)

        val rawOutput = output[0]
        val candidates = mutableListOf<BBox>()

        // Iterate over 3 detection scales
        var offset = 0
        for (scaleIdx in strides.indices) {
            val stride = strides[scaleIdx]
            val gridSize = inputSize / stride
            val numAnchors = anchors[scaleIdx].size
            val numCells = gridSize * gridSize * numAnchors

            for (i in 0 until numCells) {
                val idx = offset + i
                if (idx >= totalDetections) break

                // Raw outputs
                val rawCx = rawOutput[0][idx]
                val rawCy = rawOutput[1][idx]
                val rawW  = rawOutput[2][idx]
                val rawH  = rawOutput[3][idx]

                // Find best class
                var maxScore = 0f
                var maxClsIdx = -1
                for (c in 0 until numClasses) {
                    val s = sigmoid(rawOutput[4 + c][idx])
                    if (s > maxScore) { maxScore = s; maxClsIdx = c }
                }

                if (maxScore < threshold) continue

                // Decode box coordinates (YOLOv5 format)
                val gridY = (i / numAnchors) / gridSize
                val gridX = (i / numAnchors) % gridSize
                val anchorIdx = i % numAnchors
                val anchorW = anchors[scaleIdx][anchorIdx][0].toFloat()
                val anchorH = anchors[scaleIdx][anchorIdx][1].toFloat()

                val cx = (sigmoid(rawCx) * 2 - 0.5f + gridX) * stride
                val cy = (sigmoid(rawCy) * 2 - 0.5f + gridY) * stride
                val w  = ((sigmoid(rawW) * 2).pow(2) * anchorW)
                val h  = ((sigmoid(rawH) * 2).pow(2) * anchorH)

                val x1 = (cx - w / 2) / inputSize * bitmap.width
                val y1 = (cy - h / 2) / inputSize * bitmap.height
                val x2 = (cx + w / 2) / inputSize * bitmap.width
                val y2 = (cy + h / 2) / inputSize * bitmap.height

                if (x2 > x1 && y2 > y1) {
                    candidates.add(BBox(
                        max(0f, x1), max(0f, y1),
                        min(bitmap.width.toFloat(), x2), min(bitmap.height.toFloat(), y2),
                        labels[maxClsIdx], maxScore
                    ))
                }
            }
            offset += numCells
        }

        // NMS
        val results = nms(candidates)
        val inferenceMs = SystemClock.uptimeMillis() - startTime
        listener?.onResults(results, inferenceMs, bitmap.height, bitmap.width)
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

    private fun Float.pow(n: Int): Float {
        var r = 1f; repeat(n) { r *= this }; return r
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
