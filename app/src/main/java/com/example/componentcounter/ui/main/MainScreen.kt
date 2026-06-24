package com.example.componentcounter.ui.main

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation3.runtime.NavKey
import com.example.componentcounter.ui.camera.CameraScreen

@Composable
fun MainScreen(
  onItemClick: (NavKey) -> Unit,
  modifier: Modifier = Modifier
) {
  CameraScreen(modifier = modifier)
}
