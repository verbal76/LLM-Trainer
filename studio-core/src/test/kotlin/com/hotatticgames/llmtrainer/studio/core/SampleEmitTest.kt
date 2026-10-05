package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Emits real studio-core output for cross-language checks: build/sample/job.zip is consumed by
 * factory/tests/test_kotlin_job_interop.py (env LLMT_KOTLIN_JOB_ZIP) which runs the Python `import-job` on it.
 */
class SampleEmitTest {
    @Test fun emitSampleJobAndOtherPackages() {
        val dir = File(System.getProperty("sample.dir") ?: TK.tmp("sample").path)
        dir.mkdirs()
        val rig = TK.rig()
        val s = rig.open()
        val p = s.createProject(NewProject("Motorcycle Mechanic", "motorcycle service", "Diagnose and repair common faults")).ok().id
        s.ingest(p, TK.corpus(8), RightsStatus.OWNER_AUTHORED).ok()
        // the rest of a realistic library: a torque table (reference only), a licensed manual used for retrieval only, and a noisy scan-like text
        val csv = "part,torque_nm,notes\n" + (1..30).joinToString("\n") { "bolt-$it,${20 + it},check after first ride number $it" }
        s.ingest(p, listOf(TK.input("torque-table.csv", csv, "text/csv")), RightsStatus.OWNER_AUTHORED).ok()
        s.ingest(p, listOf(TK.input("dealer-manual.md", TK.doc(77), "text/markdown")), RightsStatus.REFERENCE_ONLY).ok()
        s.ingest(p, listOf(TK.input("noisy.md", "# Scan\n\n1234 5678 9012 3456 7890 ---- ==== #### 1234 5678\n\n## Real\n\n" + TK.doc(5).substringAfter("## "))), RightsStatus.OWNER_AUTHORED).ok()
        s.buildDataset(p).ok()
        s.approveDataset(p).ok()
        TK.verifyLicense(rig, s)
        s.selectBaseModel(p, TK.MODEL, null).ok()
        s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val job = ByteArrayOutputStream()
        val exp = s.exportTrainingJobPackage(p, job).ok()
        assertEquals(emptyList(), JobValidator.validate(job.toByteArray()))
        File(dir, "job.zip").writeBytes(job.toByteArray())
        File(dir, "job.sha256").writeText(exp.sha256)
        val ref = ByteArrayOutputStream()
        s.exportReferencePackage(p, ref).ok()
        File(dir, "reference.zip").writeBytes(ref.toByteArray())
        val held = ByteArrayOutputStream()
        s.exportHeldOutEvalSet(p, held).ok()
        File(dir, "heldout.zip").writeBytes(held.toByteArray())
        assertTrue(File(dir, "job.zip").length() > 1000)
    }
}
