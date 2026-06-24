package com.example.componentcounter.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.tensorflow.lite.task.vision.detector.Detection
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.min
import kotlin.math.max

data class DetectionState(
    val detections: List<Detection> = emptyList(),
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

    fun updateDetections(results: List<Detection>?, time: Long, height: Int, width: Int) {
        val rawDetections = results ?: emptyList()
        val deduped = nonMaximumSuppression(rawDetections, iouThreshold = 0.5f)
        _detectionState.value = DetectionState(
            detections = deduped,
            inferenceTime = time,
            totalCount = deduped.size,
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

    /**
     * Simple IoU-based deduplication. Two detections with IoU > threshold
     * are considered same object; keep the one with higher score.
     */
    private fun nonMaximumSuppression(
        detections: List<Detection>,
        iouThreshold: Float
    ): List<Detection> {
        if (detections.size <= 1) return detections

        val sorted = detections.sortedByDescending { it.categories.firstOrNull()?.score ?: 0f }
        val kept = mutableListOf<Detection>()

        for (detection in sorted) {
            var overlaps = false
            for (existing in kept) {
                if (iou(detection.boundingBox, existing.boundingBox) > iouThreshold) {
                    overlaps = true
                    break
                }
            }
            if (!overlaps) kept.add(detection)
        }
        return kept
    }

    private fun iou(a: android.graphics.RectF, b: android.graphics.RectF): Float {
        val intersectLeft = max(a.left, b.left)
        val intersectTop = max(a.top, b.top)
        val intersectRight = min(a.right, b.right)
        val intersectBottom = min(a.bottom, b.bottom)

        if (intersectLeft >= intersectRight || intersectTop >= intersectBottom) return 0f

        val intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop)
        val unionArea = (a.width() * a.height()) + (b.width() * b.height()) - intersectArea

        return if (unionArea <= 0f) 0f else intersectArea / unionArea
    }
}
