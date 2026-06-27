package com.example.componentcounter.viewmodel

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.tensorflow.lite.task.vision.detector.Detection
import android.graphics.RectF
import java.util.Arrays

class CounterViewModelTest {

    private lateinit var viewModel: CounterViewModel

    @Before
    fun setUp() {
        viewModel = CounterViewModel()
    }

    @Test
    fun `updateDetections with empty list sets totalCount to zero`() {
        viewModel.updateDetections(emptyList(), 100L, 480, 640)
        val state = viewModel.detectionState.value
        assertEquals(0, state.totalCount)
        assertEquals(100L, state.inferenceTime)
        assertEquals(480, state.imageHeight)
        assertEquals(640, state.imageWidth)
    }

    @Test
    fun `updateDetections with null results sets totalCount to zero`() {
        viewModel.updateDetections(null, 50L, 720, 1280)
        val state = viewModel.detectionState.value
        assertEquals(0, state.totalCount)
    }

    @Test
    fun `updateDetections deduplicates overlapping boxes via NMS`() {
        // Two detections with high IoU overlap — only highest-score should survive
        val det1 = createDetection(0f, 0f, 100f, 100f, "component", 0.9f)
        val det2 = createDetection(10f, 10f, 110f, 110f, "component", 0.6f)
        // IoU between these two ≈ 0.66 > 0.5 threshold → det2 should be suppressed

        viewModel.updateDetections(listOf(det1, det2), 200L, 480, 640)
        val state = viewModel.detectionState.value
        assertEquals(1, state.totalCount)
        assertEquals(0.9f, state.detections[0].categories[0].score)
    }

    @Test
    fun `updateDetections keeps non-overlapping boxes`() {
        // Two detections far apart — both should survive
        val det1 = createDetection(0f, 0f, 50f, 50f, "component", 0.8f)
        val det2 = createDetection(200f, 200f, 250f, 250f, "component", 0.7f)

        viewModel.updateDetections(listOf(det1, det2), 150L, 480, 640)
        val state = viewModel.detectionState.value
        assertEquals(2, state.totalCount)
    }

    @Test
    fun `takeSnapshot records current detection state`() {
        val det = createDetection(0f, 0f, 100f, 100f, "component", 0.9f)
        viewModel.updateDetections(listOf(det), 300L, 480, 640)
        viewModel.takeSnapshot()

        val snapshots = viewModel.snapshots.value
        assertEquals(1, snapshots.size)
        assertEquals(1, snapshots[0].count)
        assertEquals(300L, snapshots[0].inferenceTimeMs)
    }

    @Test
    fun `takeSnapshot captures multiple snapshots in order`() {
        viewModel.updateDetections(emptyList(), 100L, 480, 640)
        viewModel.takeSnapshot()

        val det = createDetection(0f, 0f, 100f, 100f, "component", 0.9f)
        viewModel.updateDetections(listOf(det), 200L, 480, 640)
        viewModel.takeSnapshot()

        val snapshots = viewModel.snapshots.value
        assertEquals(2, snapshots.size)
        assertEquals(0, snapshots[0].count)
        assertEquals(1, snapshots[1].count)
    }

    @Test
    fun `exportCsv produces valid CSV with headers`() {
        viewModel.updateDetections(emptyList(), 100L, 480, 640)
        viewModel.takeSnapshot()
        val csv = viewModel.exportCsv()

        val lines = csv.trim().split("\n")
        assertEquals("timestamp,count,inference_time_ms", lines[0])
        assertEquals(2, lines.size) // header + 1 data row
    }

    @Test
    fun `clearSnapshots removes all snapshots and csv data`() {
        viewModel.updateDetections(emptyList(), 100L, 480, 640)
        viewModel.takeSnapshot()
        viewModel.exportCsv()

        viewModel.clearSnapshots()
        assertEquals(0, viewModel.snapshots.value.size)
        assertNull(viewModel.csvData.value)
    }

    @Test
    fun `nms handles single detection`() {
        val det = createDetection(0f, 0f, 100f, 100f, "component", 0.5f)
        viewModel.updateDetections(listOf(det), 100L, 480, 640)
        assertEquals(1, viewModel.detectionState.value.totalCount)
    }

    @Test
    fun `nms handles identical boxes`() {
        // Two identical boxes — only highest score keeps
        val det1 = createDetection(0f, 0f, 100f, 100f, "a", 0.9f)
        val det2 = createDetection(0f, 0f, 100f, 100f, "b", 0.7f)

        viewModel.updateDetections(listOf(det1, det2), 100L, 480, 640)
        assertEquals(1, viewModel.detectionState.value.totalCount)
        assertEquals(0.9f, viewModel.detectionState.value.detections[0].categories[0].score)
    }

    // --- helpers ---

    private fun createDetection(
        left: Float, top: Float, right: Float, bottom: Float,
        label: String, score: Float
    ): Detection {
        val boundingBox = RectF(left, top, right, bottom)
        val categories = listOf(
            org.tensorflow.lite.task.vision.detector.Category.create(label, score)
        )
        // Detection constructor is internal; use reflection to create instance for testing
        return createDetectionReflective(boundingBox, categories)
    }

    @Suppress("UNCHECKED_CAST")
    private fun createDetectionReflective(
        boundingBox: RectF,
        categories: List<org.tensorflow.lite.task.vision.detector.Category>
    ): Detection {
        try {
            val clazz = Detection::class.java
            val constructor = clazz.getDeclaredConstructor(RectF::class.java, List::class.java)
            constructor.isAccessible = true
            return constructor.newInstance(boundingBox, categories)
        } catch (e: Exception) {
            throw RuntimeException("Failed to create Detection instance for testing. " +
                "TFLite Detection constructor may have changed.", e)
        }
    }
}
