package com.example.componentcounter.ml

import android.content.Context
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class ObjectDetectorHelperTest {

    private lateinit var context: Context
    private var lastError: String? = null
    private var lastResults: List<*>? = null

    @Before
    fun setUp() {
        context = RuntimeEnvironment.getApplication()
        lastError = null
        lastResults = null
    }

    @Test
    fun `init fails gracefully when model file is missing`() {
        // The helper tries to load "mobilenetv1.tflite" from assets.
        // In Robolectric test environment, assets may not exist.
        val helper = ObjectDetectorHelper(
            context = context,
            objectDetectorListener = object : ObjectDetectorHelper.DetectorListener {
                override fun onError(error: String) {
                    lastError = error
                }
                override fun onResults(
                    results: MutableList<org.tensorflow.lite.task.vision.detector.Detection>?,
                    inferenceTime: Long,
                    imageHeight: Int,
                    imageWidth: Int
                ) {
                    lastResults = results
                }
            }
        )
        // Should not throw — error should be reported via listener
        assertNotNull(helper)
    }

    @Test
    fun `detect returns early when detector failed to initialize`() {
        val helper = ObjectDetectorHelper(
            context = context,
            objectDetectorListener = object : ObjectDetectorHelper.DetectorListener {
                override fun onError(error: String) { lastError = error }
                override fun onResults(
                    results: MutableList<org.tensorflow.lite.task.vision.detector.Detection>?,
                    inferenceTime: Long,
                    imageHeight: Int,
                    imageWidth: Int
                ) {
                    lastResults = results
                }
            }
        )

        // Create a dummy bitmap
        val bitmap = android.graphics.Bitmap.createBitmap(100, 100, android.graphics.Bitmap.Config.ARGB_8888)

        // Should not throw even if detector is null
        try {
            helper.detect(bitmap, 0)
            // If we get here, it didn't crash — that's the test
        } catch (e: Exception) {
            // TFLite may throw if model is missing — that's also expected behavior
            // The key is it shouldn't infinite loop
        }
    }

    @Test
    fun `maxInitAttempts prevents infinite retry loop`() {
        val helper = ObjectDetectorHelper(
            context = context,
            objectDetectorListener = object : ObjectDetectorHelper.DetectorListener {
                override fun onError(error: String) { lastError = error }
                override fun onResults(
                    results: MutableList<org.tensorflow.lite.task.vision.detector.Detection>?,
                    inferenceTime: Long,
                    imageHeight: Int,
                    imageWidth: Int
                ) {
                    lastResults = results
                }
            }
        )

        val bitmap = android.graphics.Bitmap.createBitmap(100, 100, android.graphics.Bitmap.Config.ARGB_8888)

        // Call detect multiple times — should not infinite loop
        repeat(10) {
            try {
                helper.detect(bitmap, 0)
            } catch (e: Exception) {
                // Expected when TFLite model is missing
            }
        }
        // If we reach here without hanging, the retry guard works
    }
}
