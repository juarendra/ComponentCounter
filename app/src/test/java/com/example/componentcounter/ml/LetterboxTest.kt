package com.example.componentcounter.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class LetterboxTest {
    @Test fun landscape_scales_by_width_and_pads_top_bottom() {
        val lb = Letterbox.compute(srcW = 800, srcH = 400, dst = 416)
        assertEquals(0.52f, lb.scale, 1e-4f)
        assertEquals(0f, lb.padX, 1e-4f)
        assertEquals(104f, lb.padY, 1e-4f)
    }

    @Test fun maps_letterboxed_norm_box_back_to_source_pixels() {
        val lb = Letterbox.compute(srcW = 800, srcH = 400, dst = 416)
        val (x, y) = lb.toSource(0.5f, 0.5f)
        assertEquals(400f, x, 0.5f)
        assertEquals(200f, y, 0.5f)
    }
}
