package com.example.componentcounter.ui.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.componentcounter.ml.ObjectDetectorHelper
import com.example.componentcounter.viewmodel.CounterViewModel
import java.util.concurrent.Executors

@Composable
fun CameraScreen(modifier: Modifier = Modifier, viewModel: CounterViewModel = viewModel()) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_BACK) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasCameraPermission = granted }
    )

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasCameraPermission) {
        Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Camera permission required", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurface)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("Grant Permission") }
            }
        }
        return
    }

    if (cameraError != null) {
        Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(imageVector = Icons.Filled.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(48.dp))
                Spacer(Modifier.height(12.dp))
                Text(cameraError ?: "Unknown error", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center, modifier = Modifier.padding(horizontal = 32.dp))
                Spacer(Modifier.height(12.dp))
                OutlinedButton(onClick = { cameraError = null }) { Text("Retry") }
            }
        }
        return
    }

    Box(modifier = modifier.fillMaxSize()) {
        CameraPreviewWithDetection(
            context = context,
            lifecycleOwner = lifecycleOwner,
            viewModel = viewModel,
            lensFacing = lensFacing,
            onError = { cameraError = it }
        )

        val detectionState by viewModel.detectionState.collectAsState()

        Canvas(modifier = Modifier.fillMaxSize()) {
            val scaleFactor = minOf(size.width / detectionState.imageWidth, size.height / detectionState.imageHeight)
            val offsetX = (size.width - detectionState.imageWidth * scaleFactor) / 2f
            val offsetY = (size.height - detectionState.imageHeight * scaleFactor) / 2f
            val textPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.GREEN
                textSize = 36f
                isAntiAlias = true
                style = android.graphics.Paint.Style.FILL
            }

            for (box in detectionState.detections) {
                val left = box.x1 * scaleFactor + offsetX
                val top = box.y1 * scaleFactor + offsetY
                val right = box.x2 * scaleFactor + offsetX
                val bottom = box.y2 * scaleFactor + offsetY

                drawRect(color = Color.Green, topLeft = Offset(left, top), size = Size(right - left, bottom - top), style = Stroke(width = 4.dp.toPx()))

                val label = "${box.label} ${(box.score * 100).toInt()}%"
                drawContext.canvas.nativeCanvas.drawText(label, left, top - 8f, textPaint)
            }
        }

        Box(modifier = Modifier.align(Alignment.TopCenter).padding(32.dp).background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(12.dp)).padding(horizontal = 20.dp, vertical = 12.dp)) {
            Text(text = "Count: ${detectionState.totalCount}", color = Color.White, fontSize = 24.sp, fontWeight = FontWeight.Bold)
        }

        Box(modifier = Modifier.align(Alignment.TopEnd).padding(32.dp).background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp)).padding(horizontal = 10.dp, vertical = 6.dp)) {
            Text(text = "${detectionState.inferenceTime}ms", color = Color.White.copy(alpha = 0.8f), fontSize = 12.sp)
        }

        FloatingActionButton(onClick = { lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK }, modifier = Modifier.align(Alignment.BottomEnd).padding(24.dp), shape = CircleShape, containerColor = Color.Black.copy(alpha = 0.5f)) {
            Text("⟳", fontSize = 20.sp, color = Color.White)
        }

        FloatingActionButton(onClick = { viewModel.takeSnapshot() }, modifier = Modifier.align(Alignment.BottomStart).padding(24.dp), shape = CircleShape, containerColor = Color.Black.copy(alpha = 0.5f)) {
            Text("📸", fontSize = 20.sp)
        }
    }
}

@Composable
fun CameraPreviewWithDetection(context: Context, lifecycleOwner: androidx.lifecycle.LifecycleOwner, viewModel: CounterViewModel, lensFacing: Int, onError: (String) -> Unit) {
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val detector = remember { ObjectDetectorHelper(context = context, listener = object : ObjectDetectorHelper.DetectorListener {
        override fun onError(error: String) { Log.e("CameraScreen", error); onError(error) }
        override fun onResults(results: List<com.example.componentcounter.ml.BBox>, inferenceTime: Long, imageHeight: Int, imageWidth: Int) {
            viewModel.updateDetections(results, inferenceTime, imageHeight, imageWidth)
        }
    }) }

    AndroidView(factory = { ctx ->
        val previewView = PreviewView(ctx)
        val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val hasLens = try { cameraProvider.hasCamera(CameraSelector.Builder().requireLensFacing(lensFacing).build()) } catch (e: Exception) { false }
            if (!hasLens) { onError("Selected camera is not available on this device"); return@addListener }

            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { proxy ->
                    proxy.use {
                        val bmp = proxy.toBitmap()
                        detector.detect(rotateBitmap(bmp, proxy.imageInfo.rotationDegrees))
                    }
                } }

            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            try { cameraProvider.unbindAll(); cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalyzer) }
            catch (exc: Exception) { Log.e("CameraScreen", "Use case binding failed", exc); onError("Failed to start camera: ${exc.message}") }
        }, ContextCompat.getMainExecutor(ctx))
        previewView
    }, modifier = Modifier.fillMaxSize())

    DisposableEffect(lensFacing) { onDispose { } }
    DisposableEffect(Unit) { onDispose { cameraExecutor.shutdown() } }
}

private fun rotateBitmap(bitmap: android.graphics.Bitmap, degrees: Int): android.graphics.Bitmap {
    if (degrees == 0) return bitmap
    val matrix = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
    return android.graphics.Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}
