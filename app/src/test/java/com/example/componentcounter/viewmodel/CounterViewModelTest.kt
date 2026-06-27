package com.example.componentcounter.viewmodel

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class CounterViewModelTest {

    private lateinit var viewModel: CounterViewModel

    @Before
    fun setUp() {
        viewModel = CounterViewModel()
    }

    @Test
    fun `updateDetections with empty list sets totalCount to zero`() {
        viewModel.updateDetections(emptyList(), 100L, 480, 640)
        assertEquals(0, viewModel.detectionState.value.totalCount)
    }

    @Test
    fun `updateDetections with list sets correct count`() {
        val boxes = listOf(
            com.example.componentcounter.ml.BBox(0f, 0f, 100f, 100f, "Resistor", 0.9f),
            com.example.componentcounter.ml.BBox(50f, 50f, 150f, 150f, "Diode", 0.8f),
            com.example.componentcounter.ml.BBox(200f, 200f, 300f, 300f, "Transistor", 0.7f)
        )
        viewModel.updateDetections(boxes, 150L, 480, 640)
        assertEquals(3, viewModel.detectionState.value.totalCount)
        assertEquals(150L, viewModel.detectionState.value.inferenceTime)
        assertEquals(480, viewModel.detectionState.value.imageHeight)
        assertEquals(640, viewModel.detectionState.value.imageWidth)
    }

    @Test
    fun `detectionState holds correct bbox data`() {
        val box = com.example.componentcounter.ml.BBox(10f, 20f, 110f, 120f, "Condensator", 0.95f)
        viewModel.updateDetections(listOf(box), 200L, 480, 640)
        val det = viewModel.detectionState.value.detections[0]
        assertEquals(10f, det.x1)
        assertEquals(20f, det.y1)
        assertEquals(110f, det.x2)
        assertEquals(120f, det.y2)
        assertEquals("Condensator", det.label)
        assertEquals(0.95f, det.score)
    }

    @Test
    fun `takeSnapshot records current detection state`() {
        val box = com.example.componentcounter.ml.BBox(0f, 0f, 100f, 100f, "Resistor", 0.9f)
        viewModel.updateDetections(listOf(box), 300L, 480, 640)
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

        val box = com.example.componentcounter.ml.BBox(0f, 0f, 100f, 100f, "Resistor", 0.9f)
        viewModel.updateDetections(listOf(box), 200L, 480, 640)
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
        assertEquals(2, lines.size)
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
    fun `multiple snapshots with different counts`() {
        viewModel.updateDetections(listOf(com.example.componentcounter.ml.BBox(0f, 0f, 10f, 10f, "R", 1f)), 100L, 480, 640)
        viewModel.takeSnapshot()
        assertEquals(1, viewModel.snapshots.value[0].count)

        viewModel.updateDetections(listOf(
            com.example.componentcounter.ml.BBox(0f, 0f, 10f, 10f, "R", 1f),
            com.example.componentcounter.ml.BBox(20f, 20f, 30f, 30f, "D", 1f)
        ), 200L, 480, 640)
        viewModel.takeSnapshot()
        assertEquals(2, viewModel.snapshots.value.size)
        assertEquals(1, viewModel.snapshots.value[0].count)
        assertEquals(2, viewModel.snapshots.value[1].count)
    }

    @Test
    fun `exportCsv contains correct data rows`() {
        viewModel.updateDetections(listOf(com.example.componentcounter.ml.BBox(0f, 0f, 10f, 10f, "R", 1f)), 999L, 480, 640)
        viewModel.takeSnapshot()
        val csv = viewModel.exportCsv()

        val lines = csv.trim().split("\n")
        assertEquals(2, lines.size)
        assertTrue(lines[1].endsWith(",999"))
        assertTrue(lines[1].contains(",1,"))
    }

    @Test
    fun `updateDetections with null handled as empty`() {
        viewModel.updateDetections(emptyList(), 0L, 480, 640)
        assertEquals(0, viewModel.detectionState.value.totalCount)
        assertEquals(0, viewModel.detectionState.value.detections.size)
    }
}
