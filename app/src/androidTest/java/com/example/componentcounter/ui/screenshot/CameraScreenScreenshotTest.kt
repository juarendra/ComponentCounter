package com.example.componentcounter.ui.screenshot

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.componentcounter.MainActivity
import com.github.takahirom.roborazzi.captureRoboImage
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CameraScreenScreenshotTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun cameraPermissionScreen() {
        // Captures the permission prompt screen as a golden image
        composeTestRule.captureRoboImage("camera_permission_prompt.png")
    }
}
