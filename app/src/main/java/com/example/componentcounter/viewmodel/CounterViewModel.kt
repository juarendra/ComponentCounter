package com.example.componentcounter.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.tensorflow.lite.task.vision.detector.Detection

data class DetectionState(
    val detections: List<Detection> = emptyList(),
    val inferenceTime: Long = 0,
    val totalCount: Int = 0,
    val imageHeight: Int = 1,
    val imageWidth: Int = 1
)

class CounterViewModel : ViewModel() {
    private val _detectionState = MutableStateFlow(DetectionState())
    val detectionState: StateFlow<DetectionState> = _detectionState.asStateFlow()
    
    fun updateDetections(results: List<Detection>?, time: Long, height: Int, width: Int) {
        val count = results?.size ?: 0
        _detectionState.value = DetectionState(
            detections = results ?: emptyList(),
            inferenceTime = time,
            totalCount = count,
            imageHeight = height,
            imageWidth = width
        )
    }
}
