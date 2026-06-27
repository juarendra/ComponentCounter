package com.example.componentcounter.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

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

    companion object {
        val LABEL_MAP = mapOf(
            0 to "Resistor", 1 to "Diode", 2 to "Transistor", 3 to "Condensator"
        )
        private var cellOutputBuffer: Array<out Any>? = null
    }

    init {
        try {
            val model = loadModelFile("best_nms.tflite")
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

        // Preprocess: resize to 416x416, normalize to [0,1]
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

        // Output format with NMS: [1, num_detections, 6] = [x1,y1,x2,y2,class,score]
        // Allocate large enough (max 300 detections)
        val output = Array(1) { Array(300) { FloatArray(6) } }

        interpreter.run(inputBuffer, output)
        val inferenceMs = SystemClock.uptimeMillis() - startTime

        // Parse valid detections
        val results = mutableListOf<BBox>()
        val raw = output[0]
        for (i in raw.indices) {
            val row = raw[i]
            val score = row[5]
            if (score <= 0f) continue  // end of detections

            val classIdx = row[4].toInt().coerceIn(0, labels.size - 1)
            val label = labels[classIdx]

            // Denormalize to bitmap dimensions
            val scaleX = bitmap.width.toFloat() / inputSize
            val scaleY = bitmap.height.toFloat() / inputSize
            val box = BBox(
                x1 = row[0] * scaleX, y1 = row[1] * scaleY,
                x2 = row[2] * scaleX, y2 = row[3] * scaleY,
                label = label, score = score
            )
            results.add(box)
        }

        listener?.onResults(results, inferenceMs, bitmap.height, bitmap.width)
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(results: List<BBox>, inferenceTime: Long, imageHeight: Int, imageWidth: Int)
    }
}
