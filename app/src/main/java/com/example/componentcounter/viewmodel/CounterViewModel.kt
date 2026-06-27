package com.example.componentcounter.viewmodel

import androidx.lifecycle.ViewModel
import com.example.componentcounter.ml.BBox
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class DetectionState(
    val detections: List<BBox> = emptyList(),
    val inferenceTime: Long = 0,
    val totalCount: Int = 0,
    val imageHeight: Int = 1,
    val imageWidth: Int = 1
)

data class Snapshot(
    val timestamp: Long = System.currentTimeMillis(),
    val count: Int = 0,
    val inferenceTimeMs: Long = 0
)

class CounterViewModel : ViewModel() {
    private val _detectionState = MutableStateFlow(DetectionState())
    val detectionState: StateFlow<DetectionState> = _detectionState.asStateFlow()

    private val _snapshots = MutableStateFlow<List<Snapshot>>(emptyList())
    val snapshots: StateFlow<List<Snapshot>> = _snapshots.asStateFlow()

    private val _csvData = MutableStateFlow<String?>(null)
    val csvData: StateFlow<String?> = _csvData.asStateFlow()

    fun updateDetections(results: List<BBox>, time: Long, height: Int, width: Int) {
        _detectionState.value = DetectionState(
            detections = results,
            inferenceTime = time,
            totalCount = results.size,
            imageHeight = height,
            imageWidth = width
        )
    }

    fun takeSnapshot() {
        val current = _detectionState.value
        val snapshot = Snapshot(
            count = current.totalCount,
            inferenceTimeMs = current.inferenceTime
        )
        _snapshots.value = _snapshots.value + snapshot
    }

    fun exportCsv(): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
        val sb = StringBuilder()
        sb.appendLine("timestamp,count,inference_time_ms")
        for (s in _snapshots.value) {
            sb.appendLine("${dateFormat.format(Date(s.timestamp))},${s.count},${s.inferenceTimeMs}")
        }
        val csv = sb.toString()
        _csvData.value = csv
        return csv
    }

    fun clearSnapshots() {
        _snapshots.value = emptyList()
        _csvData.value = null
    }
}
