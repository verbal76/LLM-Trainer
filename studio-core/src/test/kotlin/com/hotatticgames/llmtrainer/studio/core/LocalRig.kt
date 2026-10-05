package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.File

/** A StudioCore wired to scripted engine backends, a real on-disk workspace, an installed (fake bytes) base model and a verified license. */
class LocalRig(
    nDocs: Int = 8, sections: Int = 5, withEngine: Boolean = true, installBase: Boolean = true,
    var extraSnapshot: String = ",\"isCharging\":true",
) {
    val manual = ManualRunner()
    val rig = TK.rig(runner = manual)
    var snap: () -> String = { TK.snapshot(extra = extraSnapshot) }
    val inference = ScriptedInference()
    val trainer = ScriptedTrainer { path -> Hashing.sha256File(File(path)) }
    val withEng = withEngine
    var s: StudioCore = open()
    val p: ProjectId
    val variantId: String

    fun open(): StudioCore {
        rig.snap = { snap() }
        return StudioCore(rig.dir, { snap() }, null, rig.http, rig.clock, rig.storage, manual, rig.ids, "test", SafetyPolicy(), EmbeddedRegistry.files,
            inference = if (withEng) inference else null, trainer = if (withEng) trainer else null)
    }

    init {
        p = TK.readyProject(s, nDocs)
        TK.verifyLicense(rig, s)
        variantId = s.model(TK.MODEL).ok().variants.first().id
        if (installBase) installBase()
        s.selectBaseModel(p, TK.MODEL, variantId).ok()
        s.approveDataset(p).ok()
        inference.knowledge = allSentences(nDocs)
        inference.sections = allSections(nDocs)
    }

    fun baseFile(): File {
        val dir = ModelFiles(File(rig.dir, "workspace/models")).dirOf(variantId)
        return File(dir, "base.gguf")
    }

    fun installBase(bytes: String = "GGUF-FAKE-BASE-MODEL-BYTES") {
        val mf = ModelFiles(File(rig.dir, "workspace/models"))
        val f = File(mf.dirOf(variantId), "base.gguf"); f.parentFile.mkdirs(); f.writeText(bytes)
        Fs.writeJson(mf.acquiredRecord(variantId), linkedMapOf("schema" to 1, "file" to "base.gguf", "sha256" to Hashing.sha256File(f), "size" to f.length()))
    }

    /** All sentences of the corpus (what the scripted specialist "knows"). */
    fun allSentences(n: Int): List<String> = ExampleGen.run { (1..n).flatMap { d -> TK.doc(d).lines().filter { it.isNotBlank() && !it.startsWith("#") }.flatMap { sentences(it) } } }

    /** section title -> the section's text */
    fun allSections(n: Int): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        for (d in 1..n) {
            var title: String? = null
            val sb = StringBuilder()
            fun flush() { title?.let { out[it] = sb.toString().trim() }; sb.setLength(0) }
            for (line in TK.doc(d).lines()) {
                if (line.startsWith("## ")) { flush(); title = line.removePrefix("## ") } else if (!line.startsWith("#") && line.isNotBlank()) sb.append(line).append(' ')
            }
            flush()
        }
        return out
    }

    fun run() = manual.runAll()

    fun train(settings: TrainingSettings = TrainingSettings(), confirmed: Boolean = true): TrainingRun = s.startLocalTraining(p, settings, confirmed).ok()

    fun trainToCompletion(settings: TrainingSettings = TrainingSettings()): SpecialistInfo {
        val r = train(settings)
        run()
        val done = s.trainingRun(r.id).ok()
        check(done.state == TrainingRunState.SUCCEEDED) { "training did not succeed: ${done.state} ${done.message} ${done.error}" }
        return s.specialists(p).first { it.id == done.specialistId }
    }
}
