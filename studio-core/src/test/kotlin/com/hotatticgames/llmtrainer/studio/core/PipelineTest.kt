package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** Builds a desktop-style results.zip for tests. */
object ResultsKit {
    fun results(
        projectId: String, jobId: String, status: String = "completed", stub: Boolean = false, claim: Boolean = false, n: Int = 24,
        reason: String = "n too small", corruptChecksum: Boolean = false, withReport: Boolean = true, extra: Map<String, ByteArray> = emptyMap(),
        kind: String = "adapter", rowsOverride: List<Map<String, Any?>>? = null,
    ): ByteArray {
        val w = PackageWriter()
        w.addJson("manifest.json", linkedMapOf<String, Any?>("format" to "llmtrainer-results", "version" to 1, "project_id" to projectId, "job_id" to jobId, "job_content_hash" to "sha256:" + "0".repeat(64),
            "created_at" to "2026-10-05T12:00:00Z", "status" to status, "base_model" to mapOf<String, Any?>("registry_id" to TK.MODEL, "family" to "Qwen3", "exact_version" to "x", "variant" to null, "license_entry_hash" to "sha256:" + "1".repeat(64)),
            "specialist" to mapOf<String, Any?>("kind" to kind, "artifact_refs" to (if (kind == "none") emptyList() else listOf("adapter/adapter_model.safetensors")),
                "sha256s" to (if (kind == "none") emptyMap() else mapOf<String, Any?>("adapter/adapter_model.safetensors" to "sha256:" + "a".repeat(64))), "included_artifacts" to emptyList<String>(), "quantization" to null,
                "method" to "qlora", "trainer" to "hf_lora/1", "local_experiment" to false, "is_pipeline_validation_stub" to stub),
            "stages" to listOf(mapOf<String, Any?>("stage" to "train", "state" to "executed", "detail" to "")), "packaging" to mapOf<String, Any?>("exportable" to false, "blockers" to emptyList<String>()),
            "generated_by" to mapOf<String, Any?>("tool" to "llmtrainer", "tool_version" to "0.1.0"), "notes" to emptyList<String>()))
        if (withReport) {
            val rows = rowsOverride ?: listOf(
                linkedMapOf<String, Any?>("metric" to "terminology_coverage", "category" to "domain", "higher_is_better" to true, "base" to 0.31, "specialist" to 0.44, "delta" to 0.13, "ci_low" to 0.02, "ci_high" to 0.24, "n" to n, "notes" to "", "quantized" to null),
                linkedMapOf<String, Any?>("metric" to "unsupported_claims", "category" to "safety", "higher_is_better" to false, "base" to 0.20, "specialist" to 0.12, "delta" to -0.08, "ci_low" to -0.2, "ci_high" to 0.01, "n" to n, "notes" to "", "quantized" to null))
            w.addJson("evaluation_report.json", linkedMapOf<String, Any?>("format" to "llmtrainer-evaluation-report", "version" to 1, "project_id" to projectId, "job_id" to jobId, "eval_id" to "eval-abc123",
                "evaluated_on" to "2026-10-05", "held_out_only" to true, "split" to "test",
                "generated_by" to mapOf<String, Any?>("stub" to stub, "backend" to "stub", "evaluator" to mapOf<String, Any?>("name" to "evalsuite_heuristic", "version" to "1", "is_stub" to stub), "tool" to "llmtrainer", "tool_version" to "0.1.0",
                    "subjects" to mapOf<String, Any?>("base" to "Qwen3-1.7B base", "specialist" to "motorcycle adapter")),
                "rows" to rows, "regressions" to emptyList<String>(), "caveats" to listOf("Heuristic lexical metrics only."),
                "sample_sizes" to mapOf<String, Any?>("heldout_items" to n, "per_metric_n" to mapOf<String, Any?>("terminology_coverage" to n)),
                "improvement_claim_allowed" to claim, "improvement_claim_reason" to reason, "performance" to mapOf<String, Any?>("base" to emptyMap<String, Any?>(), "specialist" to emptyMap<String, Any?>())))
        }
        for ((k, v) in extra) w.add(k, v)
        val out = ByteArrayOutputStream()
        w.write(out)
        var bytes = out.toByteArray()
        if (corruptChecksum) {
            val files = Zips.readAll(bytes.inputStream())
            val bo = ByteArrayOutputStream()
            val zos = java.util.zip.ZipOutputStream(bo)
            for ((k, v) in files) {
                zos.putNextEntry(java.util.zip.ZipEntry(k))
                zos.write(if (k == "evaluation_report.json") v + " ".toByteArray() else v)   // content changed, old checksums.json kept
                zos.closeEntry()
            }
            zos.finish()
            bytes = bo.toByteArray()
        }
        return bytes
    }
}

class PipelineTest {
    private fun jobIdOf(zip: ByteArray) = JSONObject(Zips.readAll(zip.inputStream()).getValue("manifest.json").toString(Charsets.UTF_8)).getString("job_id")

    @Test fun endToEndProducesValidJobAndSpecialistPackages() {
        val rig = TK.rig()
        val s = rig.open()
        val p = TK.readyProject(s, 6)
        // Rights set, dataset built: needs review -> approve -> pick base model with VERIFIED license -> method -> export
        val prev = s.datasetPreview(p).ok()!!
        assertEquals(DatasetStatus.NEEDS_REVIEW, prev.status)
        assertTrue(prev.stats.byRole.getValue(ChunkRole.HELD_OUT_EVAL) > 0)
        s.approveDataset(p).ok()
        val lic = TK.verifyLicense(rig, s)
        assertEquals(LicenseState.VERIFIED, lic.state)
        s.selectBaseModel(p, TK.MODEL, null).ok()
        val methods = s.methodOptions(p).ok()
        assertTrue(methods.first { it.id == MethodIds.ADAPTER_DESKTOP }.available)
        assertFalse(methods.first { it.id == MethodIds.ADAPTER_ON_DEVICE }.available)
        s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val out = ByteArrayOutputStream()
        val exp = s.exportTrainingJobPackage(p, out).ok()
        assertEquals("training-job", exp.kind)
        assertEquals(Hashing.sha256(out.toByteArray()), exp.sha256)
        assertEquals(out.size().toLong(), exp.sizeBytes)
        assertTrue(exp.files.containsAll(listOf("manifest.json", "chunks.jsonl", "dataset/test.jsonl", "eval/heldout.jsonl", "checksums.json", "license/license_text.txt")))
        val problems = JobValidator.validate(out.toByteArray())
        assertEquals(emptyList(), problems)
        val files = Zips.readAll(out.toByteArray().inputStream())
        val m = JSONObject(files.getValue("manifest.json").toString(Charsets.UTF_8))
        assertEquals(TK.MODEL, m.getJSONObject("base_model").getString("registry_id"))
        assertEquals("owner", m.getJSONObject("license_evidence").getString("attested_by"))
        // honest labelling of example provenance
        assertTrue(m.getJSONObject("extensions").getJSONObject("studio").getString("dataset_notes").contains("not human-written"))
        val dm = JSONObject(files.getValue("dataset_manifest.json").toString(Charsets.UTF_8))
        assertEquals("template", dm.getJSONObject("builder").getString("type"))

        // evaluation: held-out stage turns IN_PROGRESS -> import results
        val proj = s.getProject(p).ok()
        assertEquals(StageStatus.DONE, proj.stages.first { it.id == StageId.TRAINING_PACKAGE }.status)
        val results = ResultsKit.results(p.value, jobIdOf(out.toByteArray()), claim = false, n = 24)
        val ev = s.importEvaluation(p, results.inputStream()).ok()
        assertFalse(ev.improvementClaimAllowed)
        assertEquals(2, ev.metrics.size)
        assertEquals(0.44, ev.metrics[0].specialist)
        assertEquals(StageStatus.NEEDS_ATTENTION, s.getProject(p).ok().stages.first { it.id == StageId.EVALUATION }.status)

        // specialist export + import elsewhere
        val spec = ByteArrayOutputStream()
        val sp = s.exportSpecialistPackage(p, spec).ok()
        assertTrue(sp.warnings.any { it.contains("does NOT support an improvement claim") })
        val view = s.importSpecialistPackage(spec.toByteArray().inputStream()).ok()
        assertTrue(view.valid, view.validation.toString())
        assertEquals(TK.MODEL, view.baseModelId)
        assertFalse(view.improvementClaimAllowed)
        assertEquals(LicenseState.VERIFIED, view.licenseState)
        assertEquals(StageStatus.DONE, s.getProject(p).ok().stages.first { it.id == StageId.EXPORT }.status)
    }
}
