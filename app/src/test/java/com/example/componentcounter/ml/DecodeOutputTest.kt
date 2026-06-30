package com.example.componentcounter.ml

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class DecodeOutputTest {
    // numOutputs=8 (4 box + 4 class), numBoxes=3549 — matches the verified model contract.
    private fun emptyRaw() = Array(8) { FloatArray(3549) }

    // Constructed via Robolectric context (matches ObjectDetectorHelperTest). The model asset
    // won't load in a JVM test, so init swallows the failure and leaves interpreter=null —
    // fine, because decodeOutput is pure and never touches the interpreter.
    private fun newHelper() = ObjectDetectorHelper(
        context = InstrumentationRegistry.getInstrumentation().targetContext,
        listener = null
    )

    @Test fun no_candidates_below_threshold_returns_empty() {
        val helper = newHelper()
        val raw = emptyRaw() // all class scores 0 -> below 0.25 threshold
        val lb = Letterbox.compute(640, 480, 416)
        val out = helper.decodeOutput(raw, lb, 640, 480)
        assertTrue(out.isEmpty())
    }

    @Test fun single_high_score_box_decodes_and_maps_to_source() {
        val helper = newHelper()
        val raw = emptyRaw()
        val i = 100
        // centered box, 20% size. Class index 2 = Resistor (data.yaml order:
        // 0 Condensator, 1 Diode, 2 Resistor, 3 Transistor) -> channel 4+2.
        raw[0][i] = 0.5f; raw[1][i] = 0.5f; raw[2][i] = 0.2f; raw[3][i] = 0.2f
        raw[6][i] = 0.9f
        val lb = Letterbox.compute(640, 480, 416) // letterbox geometry for a 640x480 source
        val out = helper.decodeOutput(raw, lb, 640, 480)
        assertEquals(1, out.size)
        val b = out[0]
        assertEquals("Resistor", b.label)
        assertEquals(0.9f, b.score, 1e-4f)
        // box must be in-bounds and centered-ish on the 640x480 source
        assertTrue(b.x1 >= 0f && b.y1 >= 0f && b.x2 <= 640f && b.y2 <= 480f)
        assertTrue(b.x2 > b.x1 && b.y2 > b.y1)
        val cxSrc = (b.x1 + b.x2) / 2f
        val cySrc = (b.y1 + b.y2) / 2f
        assertEquals(320f, cxSrc, 2f)
        assertEquals(240f, cySrc, 2f)
    }

    @Test fun nms_suppresses_overlapping_duplicate() {
        val helper = newHelper()
        val raw = emptyRaw()
        // two near-identical boxes, same class, both high score -> NMS keeps 1
        for (i in intArrayOf(10, 11)) {
            raw[0][i] = 0.5f; raw[1][i] = 0.5f; raw[2][i] = 0.3f; raw[3][i] = 0.3f
            raw[4][i] = 0.8f
        }
        val lb = Letterbox.compute(416, 416, 416)
        val out = helper.decodeOutput(raw, lb, 416, 416)
        assertEquals(1, out.size)
    }
}
