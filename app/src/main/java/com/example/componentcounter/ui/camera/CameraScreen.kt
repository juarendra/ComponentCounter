package com.example.componentcounter.ui.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCharacteristics
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
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
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
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.ByteArrayOutputStream
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
        // Permission denied
        Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    "Camera permission required",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Spacer(Modifier.height(12.dp))
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text("Grant Permission")
                }
            }
        }
        return
    }

    if (cameraError != null) {
        // Camera initialization error
        Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = androidx.compose.material.icons.Icons.Outlined.Warning,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.error,
                    modifier = Modifier.size(48.dp)
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    cameraError ?: "Unknown error",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 32.dp)
                )
                Spacer(Modifier.height(12.dp))
                OutlinedButton(onClick = { cameraError = null }) {
                    Text("Retry")
                }
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

        // Detection overlay
        val detectionState by viewModel.detectionState.collectAsState()

        Canvas(modifier = Modifier.fillMaxSize()) {
            val scaleFactor = minOf(
                size.width / detectionState.imageWidth,
                size.height / detectionState.imageHeight
            )
            val offsetX = (size.width - detectionState.imageWidth * scaleFactor) / 2f
            val offsetY = (size.height - detectionState.imageHeight * scaleFactor) / 2f

            for (detection in detectionState.detections) {
                val boundingBox = detection.boundingBox
                val left = boundingBox.left * scaleFactor + offsetX
                val top = boundingBox.top * scaleFactor + offsetY
                val right = boundingBox.right * scaleFactor + offsetX
                val bottom = boundingBox.bottom * scaleFactor + offsetY

                drawRect(
                    color = Color.Green,
                    topLeft = Offset(left, top),
                    size = Size(right - left, bottom - top),
                    style = Stroke(width = 4.dp.toPx())
                )
            }
        }

        // Count badge
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(32.dp)
                .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                .padding(horizontal = 20.dp, vertical = 12.dp)
        ) {
            Text(
                text = "Count: ${detectionState.totalCount}",
                color = Color.White,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )
        }

        // Inference time badge
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(32.dp)
                .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                .padding(horizontal = 10.dp, vertical = 6.dp)
        ) {
            Text(
                text = "${detectionState.inferenceTime}ms",
                color = Color.White.copy(alpha = 0.8f),
                fontSize = 12.sp
            )
        }

        // Switch camera button
        FloatingActionButton(
            onClick = {
                lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                    CameraSelector.LENS_FACING_FRONT
                else
                    CameraSelector.LENS_FACING_BACK
            },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(24.dp),
            shape = CircleShape,
            containerColor = Color.Black.copy(alpha = 0.5f)
        ) {
            Text("⟳", fontSize = 20.sp, color = Color.White)
        }

        // Snapshot button
        FloatingActionButton(
            onClick = { viewModel.takeSnapshot() },
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(24.dp),
            shape = CircleShape,
            containerColor = Color.Black.copy(alpha = 0.5f)
        ) {
            Text("📸", fontSize = 20.sp)
        }
    }
}

@Composable
fun CameraPreviewWithDetection(
    context: Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    viewModel: CounterViewModel,
    lensFacing: Int,
    onError: (String) -> Unit
) {
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val objectDetectorHelper = remember(context) {
        ObjectDetectorHelper(
            context = context,
            objectDetectorListener = object : ObjectDetectorHelper.DetectorListener {
                override fun onError(error: String) {
                    Log.e("CameraScreen", error)
                    onError(error)
                }

                override fun onResults(
                    results: MutableList<Detection>?,
                    inferenceTime: Long,
                    imageHeight: Int,
                    imageWidth: Int
                ) {
                    viewModel.updateDetections(results, inferenceTime, imageHeight, imageWidth)
                }
            }
        )
    }

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                // Check if requested lens is available
                val hasLens = try {
                    cameraProvider.hasCamera(CameraSelector.Builder().requireLensFacing(lensFacing).build())
                } catch (e: Exception) { false }

                if (!hasLens) {
                    onError("Selected camera is not available on this device")
                    return@addListener
                }

                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy ->
                            imageProxy.use { proxy ->
                                val bitmap = proxy.toBitmap() ?: return@use
                                // Pass raw bitmap + rotation — TFLite handles rotation via Rot90Op
                                val rotationDegrees = proxy.imageInfo.rotationDegrees
                                objectDetectorHelper.detect(bitmap, rotationDegrees)
                            }
                        }
                    }

                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(lensFacing)
                    .build()

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalyzer
                    )
                } catch (exc: Exception) {
                    Log.e("CameraScreen", "Use case binding failed", exc)
                    onError("Failed to start camera: ${exc.message}")
                }
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )

    DisposableEffect(lensFacing) {
        onDispose { /* lens change handled by rebinding in factory */ }
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }
}

/**
 * Safely convert [androidx.camera.core.ImageProxy] to [Bitmap].
 * Returns null if conversion fails (e.g., corrupt JPEG data).
 */
private fun androidx.camera.core.ImageProxy.toBitmap(): Bitmap? {
    return when (format) {
        ImageFormat.YUV_420_888 -> {
            val yPlane = planes[0]
            val uPlane = planes[1]
            val vPlane = planes[2]

            val yBuffer = yPlane.buffer
            val uBuffer = uPlane.buffer
            val vBuffer = vPlane.buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)

            if (vPlane.pixelStride == 2) {
                var i = ySize
                var vi = 0
                while (vi < vSize) {
                    nv21[i++] = vBuffer.get(vi)
                    vi += 2
                }
                i = ySize + vSize / 2
                var ui = 0
                while (ui < uSize) {
                    nv21[i++] = uBuffer.get(ui)
                    ui += 2
                }
            } else {
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + vSize, uSize)
            }

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 95, out)
            val jpegData = out.toByteArray()
            BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)
        }
        else -> {
            // RGBA_8888: buffer is guaranteed contiguous
            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmap ->
                val buffer = planes[0].buffer
                buffer.rewind()
                bitmap.copyPixelsFromBuffer(buffer)
            }
        }
    }
}
