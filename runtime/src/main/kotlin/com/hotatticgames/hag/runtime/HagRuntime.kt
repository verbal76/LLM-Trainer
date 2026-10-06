package com.hotatticgames.hag.runtime

/** How native libraries are loaded; injectable so the "engine unavailable" paths are testable on any device. */
fun interface LibraryLoader { fun load(name: String) }

/** Outcome of bringing up the native engine. The host boots either way; only [Ready] carries a usable engine. */
sealed class EngineStatus {
    class Ready(val engine: HagEngine, val version: String, val systemInfoJson: String) : EngineStatus()

    /** [stage]: which step failed (loadRuntime | cpuGate | loadEngine | bind | init). [reason] is human-readable. */
    class Unavailable(val stage: String, val reason: String) : EngineStatus()
}

/**
 * Brings up the engine in four guarded steps. Nothing here can crash the process:
 *  1. load libhagrt.so (tiny, baseline ISA)   -- failure => Unavailable("loadRuntime")
 *  2. CPU-feature gate                        -- missing ISA features => Unavailable("cpuGate"), engine library NEVER loaded
 *  3. load libhagengine.so + bind symbols     -- failure => Unavailable("loadEngine"/"bind")
 *  4. hag_engine_init                         -- failure => Unavailable("init")
 */
object HagRuntime {
    const val RUNTIME_LIB = "hagrt"
    const val ENGINE_LIB = "hagengine"

    fun load(
        loader: LibraryLoader = LibraryLoader { System.loadLibrary(it) },
        /** Test hook: replaces the native CPU gate result (null = pass). */
        cpuGateOverride: (() -> String?)? = null,
    ): EngineStatus {
        try {
            loader.load(RUNTIME_LIB)
        } catch (t: Throwable) {
            return EngineStatus.Unavailable("loadRuntime", "native runtime library missing or unloadable: ${t.message ?: t.javaClass.simpleName}")
        }
        try {
            val gate = if (cpuGateOverride != null) cpuGateOverride() else NativeBridge.nativeCpuCheck()?.utf8()
            if (gate != null) return EngineStatus.Unavailable("cpuGate", gate)
        } catch (t: Throwable) {
            return EngineStatus.Unavailable("cpuGate", "CPU feature check failed: ${t.message ?: t.javaClass.simpleName}")
        }
        try {
            loader.load(ENGINE_LIB)
        } catch (t: Throwable) {
            return EngineStatus.Unavailable("loadEngine", "engine library missing or unloadable: ${t.message ?: t.javaClass.simpleName}")
        }
        try {
            NativeBridge.nativeBind()?.let { return EngineStatus.Unavailable("bind", it.utf8()) }
            NativeBridge.nativeEngineInit()?.let { return EngineStatus.Unavailable("init", it.utf8()) }
            val engine = HagEngine()
            return EngineStatus.Ready(engine, engine.version(), engine.systemInfoJson())
        } catch (t: Throwable) {
            return EngineStatus.Unavailable("init", "engine bring-up failed: ${t.message ?: t.javaClass.simpleName}")
        }
    }

    /** "arm64-v8a" | "x86_64" | "unsupported"; null if even libhagrt cannot be used. */
    fun cpuAbi(): String? = try { NativeBridge.nativeCpuAbi().utf8() } catch (_: Throwable) { null }
}
