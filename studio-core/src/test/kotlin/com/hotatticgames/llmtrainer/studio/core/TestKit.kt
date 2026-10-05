package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import java.util.Random
import java.util.concurrent.atomic.AtomicLong
import kotlin.test.fail

class FixedClock(var now: Long = 1_760_000_000_000L) : Clock {
    override fun nowMs(): Long { now += 1000; return now }
}

class SeqIds : IdSource {
    private val n = AtomicLong()
    override fun hex(nChars: Int): String = "%0${nChars}x".format(n.incrementAndGet()).takeLast(nChars)
}

class FakeStorage(var free: Long = 500L * 1024 * 1024 * 1024) : StorageProbe {
    override fun freeBytes(dir: File): Long = free
}

/** Queue-based runner: tests decide when background work runs. */
class ManualRunner : TaskRunner {
    val queue = ArrayDeque<Runnable>()
    override fun submit(task: Runnable) { queue.addLast(task) }
    fun runAll() { while (queue.isNotEmpty()) queue.removeFirst().run() }
    fun runOne() { queue.removeFirstOrNull()?.run() }
}

/**
 * In-memory HTTPS server. Supports Range, can cut the connection after N bytes (to test resume), counts requests and
 * records Range headers actually sent.
 */
class FakeHttp : Http {
    val resources = LinkedHashMap<String, ByteArray>()
    val requests = ArrayList<String>()
    val rangesSeen = ArrayList<Long>()
    var failAfterBytes: Long? = null      // next download delivers this many bytes then throws NETWORK
    var ignoreRange = false
    var failStatus: Int? = null

    override fun getBytes(url: String, maxBytes: Long, timeoutMs: Int): HttpBytes {
        requests.add(url)
        if (!url.startsWith("https://")) throw HttpException(HttpException.Kind.NOT_HTTPS, "Only HTTPS is allowed: $url")
        failStatus?.let { throw HttpException(HttpException.Kind.STATUS, "Server answered HTTP $it", it) }
        val b = resources[url] ?: throw HttpException(HttpException.Kind.STATUS, "Server answered HTTP 404", 404)
        if (b.size > maxBytes) throw HttpException(HttpException.Kind.TOO_LARGE, "too large")
        return HttpBytes(url, 200, b, "text/plain")
    }

    override fun downloadToFile(url: String, part: File, maxBytes: Long, cancelled: () -> Boolean, onProgress: (Long, Long) -> Unit, timeoutMs: Int): DownloadOutcome {
        requests.add(url)
        if (!url.startsWith("https://")) throw HttpException(HttpException.Kind.NOT_HTTPS, "Only HTTPS is allowed: $url")
        val b = resources[url] ?: throw HttpException(HttpException.Kind.STATUS, "Server answered HTTP 404", 404)
        part.parentFile?.mkdirs()
        var have = if (part.isFile) part.length() else 0L
        rangesSeen.add(have)
        if (ignoreRange && have > 0) { part.delete(); have = 0 }
        val limit = failAfterBytes
        failAfterBytes = null
        var pos = have.toInt()
        java.io.RandomAccessFile(part, "rw").use { raf ->
            raf.setLength(pos.toLong()); raf.seek(pos.toLong())
            while (pos < b.size) {
                if (cancelled()) throw HttpException(HttpException.Kind.CANCELLED, "Cancelled")
                if (limit != null && pos.toLong() - have >= limit) throw HttpException(HttpException.Kind.NETWORK, "Connection lost: injected")
                val n = minOf(1024, b.size - pos)
                raf.write(b, pos, n); pos += n
                onProgress(pos.toLong(), b.size.toLong())
            }
        }
        return DownloadOutcome(b.size.toLong(), pos.toLong())
    }
}

object TK {
    fun tmp(prefix: String = "studio"): File = Files.createTempDirectory(prefix).toFile().also { it.deleteOnExit() }

    /** An 8 GB-class phone: 8 GiB RAM, 4.5 GiB available, 100 GiB free of 128. */
    fun snapshot(totalGb: Double = 8.0, availGb: Double = 4.5, freeStorageGb: Double = 100.0, totalStorageGb: Double = 128.0, extra: String = ""): String {
        val g = 1024.0 * 1024 * 1024
        return """{"manufacturer":"Test","model":"Phone","sdkInt":34,"abis64":"arm64-v8a","totalRamBytes":${(totalGb * g).toLong()},"availRamBytes":${(availGb * g).toLong()},
            "lowMemoryThresholdBytes":${(0.3 * g).toLong()},"lowMemory":false,"isLowRamDevice":false,"memoryClassMb":256,"freeStorageBytes":${(freeStorageGb * g).toLong()},
            "totalStorageBytes":${(totalStorageGb * g).toLong()},"thermalStatus":0,"powerSaveMode":false,"batteryPct":80$extra}"""
    }

    class Rig(val dir: File, val http: FakeHttp, val clock: FixedClock, val storage: FakeStorage, val runner: TaskRunner, val ids: SeqIds, var snap: () -> String) {
        fun open(registry: List<Pair<String, String>> = EmbeddedRegistry.files, host: HostHooks? = null): StudioCore =
            StudioCore(dir, { snap() }, host, http, clock, storage, runner, ids, "test", com.hotatticgames.llmtrainer.qualify.SafetyPolicy(), registry)
    }

    fun rig(runner: TaskRunner = InlineTaskRunner, dir: File = tmp(), snap: String = snapshot()): Rig =
        Rig(dir, FakeHttp(), FixedClock(), FakeStorage(), runner, SeqIds()) { snap }

    fun <T> StudioResult<T>.ok(): T = (this as? StudioResult.Ok)?.value ?: fail("expected Ok but got ${(this as StudioResult.Err).error}")
    fun <T> StudioResult<T>.err(): StudioError = (this as? StudioResult.Err)?.error ?: fail("expected Err but got Ok(${(this as StudioResult.Ok).value})")

    fun input(name: String, text: String, mime: String = "text/plain") = bytesInput(name, text.toByteArray(Charsets.UTF_8), mime)
    fun bytesInput(name: String, b: ByteArray, mime: String = "text/plain") = SourceInput(name, mime, b.size.toLong()) { ByteArrayInputStream(b) }

    fun bytesOf(f: (ByteArrayOutputStream) -> Unit): ByteArray = ByteArrayOutputStream().also(f).toByteArray()

    // ---- a deterministic multi-document corpus -----------------------------------------------------------------------

    private val NOUNS = listOf("axle", "sprocket", "caliper", "piston", "gasket", "carburetor", "throttle", "crankcase", "radiator", "bearing", "chain", "swingarm", "fork", "rotor", "injector",
        "alternator", "regulator", "stator", "starter", "clutch", "camshaft", "tappet", "exhaust", "muffler", "battery", "fuse", "relay", "sensor", "coolant", "reservoir", "gearbox", "spindle",
        "flywheel", "manifold", "silencer", "headlamp", "indicator", "sidestand", "centrestand", "handlebar")
    private val VERBS = listOf("inspect", "tighten", "replace", "lubricate", "measure", "adjust", "clean", "remove", "install", "torque", "bleed", "flush", "align", "seat", "test", "service")
    private val ADJ = listOf("worn", "loose", "corroded", "clean", "dry", "seized", "bent", "cracked", "noisy", "leaking", "warm", "cold", "dirty", "pitted", "scored", "sticky")
    private val PLACES = listOf("beside the frame rail", "behind the left cover", "under the fuel tank", "near the rear shock", "inside the clutch housing", "above the swing arm pivot",
        "next to the oil filter", "below the airbox", "at the front sprocket", "around the steering head", "on the right footpeg bracket", "along the main harness")

    /** Returns a markdown document with [sections] sections of unique, factual-sounding sentences (each > 8 words). */
    fun doc(seed: Int, sections: Int = 5, paragraphs: Int = 3, title: String = "Manual $seed"): String {
        val r = Random(seed.toLong() * 7919 + 13)
        fun pick(l: List<String>) = l[r.nextInt(l.size)]
        val sb = StringBuilder("# $title\n\n")
        for (s in 1..sections) {
            sb.append("## ${pick(NOUNS).replaceFirstChar { it.uppercase() }} ${pick(ADJ)} service procedure $seed.$s\n\n")
            for (p in 1..paragraphs) {
                val sents = (1..3).map {
                    val n1 = pick(NOUNS); val n2 = pick(NOUNS)
                    when (r.nextInt(4)) {
                        0 -> "Always ${pick(VERBS)} the ${pick(ADJ)} $n1 ${pick(PLACES)} before you ${pick(VERBS)} the $n2 assembly, reference ${seed * 100 + s * 10 + p}."
                        1 -> "Tighten the $n1 bolt to ${20 + r.nextInt(80)} N-m and ${pick(VERBS)} the ${pick(ADJ)} $n2 ${pick(PLACES)} afterwards."
                        2 -> "When the $n1 feels ${pick(ADJ)}, ${pick(VERBS)} it and ${pick(VERBS)} the ${pick(ADJ)} $n2 ${pick(PLACES)} with care, step ${r.nextInt(900)}."
                        else -> "The ${pick(ADJ)} $n1 clearance should be ${(5 + r.nextInt(40)) / 10.0} mm on the $n2 ${pick(PLACES)}, as noted in table ${seed}-${s}-${r.nextInt(99)}."
                    }
                }
                sb.append(sents.joinToString(" ")).append("\n\n")
            }
        }
        return sb.toString()
    }

    fun corpus(nDocs: Int = 6, sections: Int = 5): List<SourceInput> = (1..nDocs).map { input("manual-$it.md", doc(it, sections), "text/markdown") }

    /** A project with ingested + rights-set sources and a built dataset; returns its id. */
    fun readyProject(s: Studio, nDocs: Int = 6, rights: RightsStatus = RightsStatus.OWNER_AUTHORED, name: String = "Motorcycle Mechanic", options: DatasetOptions = DatasetOptions()): ProjectId {
        val p = s.createProject(NewProject(name, "motorcycle service", "Diagnose and repair common faults")).ok().id
        val rep = s.ingest(p, corpus(nDocs), rights).ok()
        if (rep.ingested.size != nDocs) fail("expected $nDocs ingested, got ${rep.items.map { it.status to it.issues }}")
        s.buildDataset(p, options).ok()
        return p
    }

    const val MODEL = "Qwen3@Qwen3-1.7B (2504 generation, hybrid thinking)"

    /** Fetch + attest the license for [MODEL] through the studio so it becomes VERIFIED. */
    fun verifyLicense(rig: Rig, s: Studio, modelId: String = MODEL, perms: Map<Permission, Tri> = allYes(), text: String = "Apache License Version 2.0 (test text)"): LicenseInfo {
        val e = rig.http.resources.entries.firstOrNull()
        val url = (s.model(modelId).ok().license.authoritativeUrl)
        rig.http.resources[url] = text.toByteArray()
        val f = s.fetchLicenseText(modelId).ok()
        return s.attestLicense(modelId, LicenseAttestation(f.sha256, perms, true, "test attestation")).ok()
    }

    fun allYes(): Map<Permission, Tri> = Permission.values().associateWith { Tri.YES }
}
