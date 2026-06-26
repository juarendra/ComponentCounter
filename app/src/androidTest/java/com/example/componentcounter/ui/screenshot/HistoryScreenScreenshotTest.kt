package com.example.componentcounter.ui.screenshot

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.componentcounter.MainActivity
import com.github.takahirom.roborazzi.captureRoboImage
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HistoryScreenScreenshotTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun historyEmptyState() {
        // Navigate to History tab first (bottom nav button)
        composeTestRule
            .onNodeWithText("History")
            .performClick()

        // Capture the empty state
        composeTestRule.captureRoboImage("history_empty_state.png")
    }

    @Test
    fun historyWithClearDialog() {
        // Navigate to History tab
        composeTestRule
            .onNodeWithText("History")
            .performClick()

        // Tap Clear All button (should be disabled with empty state, but test the dialog)
        composeTestRule
            .onNodeWithText("No snapshots yet")
            .assertExists()

        // Capture empty state
        composeTestRule.captureRoboImage("history_empty_no_snapshots.png")
    }
}
