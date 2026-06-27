package com.example.componentcounter.ui.screenshot

import androidx.compose.material3.Text
import androidx.compose.ui.test.junit4.createComposeRule
import com.github.takahirom.roborazzi.captureRoboImage
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ComposeScreenshotTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun simpleComposeText() {
        composeTestRule.setContent {
            Text("Hello Roborazzi!")
        }
        composeTestRule.captureRoboImage()
    }
}
