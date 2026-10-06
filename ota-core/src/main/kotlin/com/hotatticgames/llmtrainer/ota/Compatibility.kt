package com.hotatticgames.llmtrainer.ota

/**
 * The deliberate runtime/version compatibility rules. A bundle is installable iff this returns empty.
 *
 * Why these rules (see docs/OTA.md):
 *  - hostApi range: the Kotlin API surface the host exposes to bundles. Additive host changes keep the level;
 *    breaking ones bump it. Bundles declare the range they were built+tested against.
 *  - nativeAbi: exact. Anything that changes JNI signatures or shipped .so files bumps it. A bundle is
 *    pure Kotlin/dex+assets, so it can only ever talk to the native runtime through the host API.
 *  - capabilities: lets a bundle depend on a *feature* of the host (e.g. gguf inference) rather than
 *    on an exact APK, so one APK can serve many bundle versions and vice versa.
 */
object Compatibility {
    fun evaluate(req: Requires, host: HostInfo): List<Reject> {
        val out = mutableListOf<Reject>()
        if (host.hostApiLevel < req.hostApi.min) {
            out += Reject(RejectCode.HOST_API_TOO_OLD, "host api ${host.hostApiLevel} < required min ${req.hostApi.min}")
        }
        if (host.hostApiLevel > req.hostApi.max) {
            out += Reject(RejectCode.HOST_API_TOO_NEW, "host api ${host.hostApiLevel} > bundle max ${req.hostApi.max}")
        }
        if (host.nativeAbi != req.nativeAbi) {
            out += Reject(RejectCode.NATIVE_ABI_MISMATCH, "host native abi ${host.nativeAbi} != bundle ${req.nativeAbi}")
        }
        for (cap in req.capabilities) {
            if (cap !in host.capabilities) out += Reject(RejectCode.MISSING_CAPABILITY, cap)
        }
        if (host.sdkInt < req.minSdk) {
            out += Reject(RejectCode.SDK_TOO_LOW, "device sdk ${host.sdkInt} < ${req.minSdk}")
        }
        return out
    }

    /** True when the only reason to refuse is that the APK itself must be replaced. */
    fun needsNewHost(rejects: List<Reject>): Boolean =
        rejects.isNotEmpty() && rejects.all {
            it.code == RejectCode.HOST_API_TOO_OLD || it.code == RejectCode.NATIVE_ABI_MISMATCH ||
                it.code == RejectCode.MISSING_CAPABILITY || it.code == RejectCode.SDK_TOO_LOW
        }
}
