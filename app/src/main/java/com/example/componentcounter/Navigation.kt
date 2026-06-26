package com.example.componentcounter

import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.ui.NavDisplay
import com.example.componentcounter.ui.camera.CameraScreen
import com.example.componentcounter.ui.history.HistoryScreen

enum class Tab(val label: String) {
    Camera("Camera"),
    History("History")
}

@Composable
fun MainNavigation() {
    var selectedTab by remember { mutableStateOf(Tab.Camera) }
    val backStack = rememberNavBackStack(Main)

    Scaffold(
        modifier = Modifier.safeDrawingPadding(),
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    icon = { Icon(Icons.Default.CameraAlt, contentDescription = null) },
                    label = { Text("Camera") },
                    selected = selectedTab == Tab.Camera,
                    onClick = { selectedTab = Tab.Camera }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Default.List, contentDescription = null) },
                    label = { Text("History") },
                    selected = selectedTab == Tab.History,
                    onClick = { selectedTab = Tab.History }
                )
            }
        }
    ) { innerPadding ->
        when (selectedTab) {
            Tab.Camera -> CameraScreen(modifier = Modifier.padding(innerPadding))
            Tab.History -> HistoryScreen(modifier = Modifier.padding(innerPadding))
        }
    }
}
