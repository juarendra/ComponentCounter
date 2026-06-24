package com.example.componentcounter.ui.camera

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.componentcounter.MainActivity
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** UI tests for CameraScreen — permission flow and UI elements. */
@RunWith(AndroidJUnit4::class)
class CameraScreenTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun cameraPermissionPromptShown() {
        // On first launch without permission granted, the permission prompt
        // text should be visible
        composeTestRule
            .onNodeWithText("Camera permission required")
            .assertExists()
    }
}
