package com.example.componentcounter.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class ObjectDetectorHelper(
    var threshold: Float = 0.4f,
    var numThreads: Int = 2,
    var maxResults: Int = 100, // components on tape can be many
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null
    private var initAttempts: Int = 0
    private val maxInitAttempts: Int = 3

    init {
        setupObjectDetector()
    }

    fun setupObjectDetector() {
        val optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = "mobilenetv1.tflite"

        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
            initAttempts = 0 // reset on success
        } catch (e: Exception) {
            initAttempts++
            Log.e("ObjectDetectorHelper", "TFLite failed to load model (attempt $initAttempts/$maxInitAttempts): ${e.message}")
            if (initAttempts >= maxInitAttempts) {
                objectDetectorListener?.onError("Object detector failed to initialize after $maxInitAttempts attempts")
            }
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            if (initAttempts >= maxInitAttempts) return // give up after max retries
            setupObjectDetector()
            if (objectDetector == null) return // still null after retry
        }

        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector?.detect(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        objectDetectorListener?.onResults(results, inferenceTime, tensorImage.height, tensorImage.width)
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(results: MutableList<Detection>?, inferenceTime: Long, imageHeight: Int, imageWidth: Int)
    }
}
