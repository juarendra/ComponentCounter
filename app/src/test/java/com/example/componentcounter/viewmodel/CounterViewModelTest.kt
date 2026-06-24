package com.example.componentcounter.viewmodel

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Test
import org.tensorflow.lite.task.vision.detector.Detection

class CounterViewModelTest {

    @Test
    fun `updateDetections correctly updates state`() {
        val viewModel = CounterViewModel()
        
        // Initial state
        assertEquals(0, viewModel.detectionState.value.totalCount)
        
        // Mock detections
        viewModel.updateDetections(null, 10L, 100, 100)
        assertEquals(0, viewModel.detectionState.value.totalCount)
        assertEquals(10L, viewModel.detectionState.value.inferenceTime)
        
        // Simulate 2 items
        val mockList = listOf(
            Detection.create(RectF(0f,0f,10f,10f), emptyList()),
            Detection.create(RectF(10f,10f,20f,20f), emptyList())
        )
        viewModel.updateDetections(mockList, 15L, 200, 200)
        assertEquals(2, viewModel.detectionState.value.totalCount)
        assertEquals(15L, viewModel.detectionState.value.inferenceTime)
    }
}
