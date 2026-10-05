package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Guards against accidental quadratic behaviour: a mid-sized library must ingest, build, export and validate in bounded time. */
class ScaleTest {
    @Test fun midSizedLibraryRunsEndToEndInBoundedTime() {
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("Big", "motorcycle service", "scale")).ok().id
        val t0 = System.nanoTime()
        val docs = (1..60).map { TK.input("big-$it.md", TK.doc(1000 + it, sections = 12, paragraphs = 4), "text/markdown") }
        val rep = s.ingest(p, docs, RightsStatus.OWNER_AUTHORED).ok()
        assertEquals(60, rep.ingested.size)
        val t1 = System.nanoTime()
        val prev = s.buildDataset(p).ok()
        val t2 = System.nanoTime()
        assertTrue(prev.stats.totalChunks > 300, "chunks=${prev.stats.totalChunks}")
        s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val out = ByteArrayOutputStream(); s.exportTrainingJobPackage(p, out).ok()
        val t3 = System.nanoTime()
        assertEquals(emptyList(), JobValidator.validate(out.toByteArray()))
        println("SCALE ingest=${(t1 - t0) / 1_000_000}ms build=${(t2 - t1) / 1_000_000}ms export=${(t3 - t2) / 1_000_000}ms chunks=${prev.stats.totalChunks} zip=${out.size()}B")
        assertTrue((t3 - t0) / 1_000_000_000 < 90, "pipeline took ${(t3 - t0) / 1_000_000_000}s")
    }
}
