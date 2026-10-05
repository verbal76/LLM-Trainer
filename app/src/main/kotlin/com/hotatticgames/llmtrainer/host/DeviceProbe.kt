package com.hotatticgames.llmtrainer.host

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Cheap point-in-time device facts for device qualification (static + current state; no benchmarking here).
 * Every field beyond the original set is optional for consumers: bundles must tolerate absence.
 * Anything not measurable is omitted rather than guessed.
 */
object DeviceProbe {
    private val INTERESTING_CPU_FEATURES = setOf(
        "asimd", "asimdhp", "asimddp", "fphp", "i8mm", "bf16", "sve", "sve2", "aes", "sha2", "crc32", "neon", "vfpv4",
    )

    fun snapshotJson(ctx: Context): String {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val st = StatFs(ctx.filesDir.path)
        val o = JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("sdkInt", Build.VERSION.SDK_INT)
            .put("abis", Build.SUPPORTED_ABIS.joinToString(","))
            .put("abis64", Build.SUPPORTED_64_BIT_ABIS.joinToString(","))
            .put("totalRamBytes", mi.totalMem)
            .put("availRamBytes", mi.availMem)
            .put("lowMemoryThresholdBytes", mi.threshold)
            .put("lowMemory", mi.lowMemory)
            .put("isLowRamDevice", am.isLowRamDevice)
            .put("memoryClassMb", am.memoryClass)
            .put("largeMemoryClassMb", am.largeMemoryClass)
            .put("freeStorageBytes", st.availableBytes)
            .put("totalStorageBytes", st.totalBytes)
            .put("cpuCores", Runtime.getRuntime().availableProcessors())
        if (Build.VERSION.SDK_INT >= 31) {
            o.put("socModel", Build.SOC_MODEL)
            o.put("socManufacturer", Build.SOC_MANUFACTURER)
        }
        runCatching {
            val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
            if (Build.VERSION.SDK_INT >= 29) o.put("thermalStatus", pm.currentThermalStatus)
            o.put("powerSaveMode", pm.isPowerSaveMode)
        }
        runCatching {
            val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            val pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            if (pct in 0..100) o.put("batteryPct", pct)
        }
        runCatching {
            val feats = ctx.packageManager.systemAvailableFeatures
            feats.firstOrNull { it.name == "android.hardware.vulkan.level" }?.let { o.put("vulkanLevel", it.version) }
            feats.firstOrNull { it.name == "android.hardware.vulkan.version" }?.let { o.put("vulkanVersion", it.version) }
            o.put("hasVulkanCompute", ctx.packageManager.hasSystemFeature("android.hardware.vulkan.compute"))
        }
        runCatching { cpuFeatures()?.let { o.put("cpuFeatures", JSONArray(it.sorted())) } }
        runCatching { meminfo(o) }
        return o.toString()
    }

    /** ARM "Features" line from /proc/cpuinfo, restricted to the ones relevant to inference kernels. */
    private fun cpuFeatures(): List<String>? {
        val f = File("/proc/cpuinfo")
        if (!f.canRead()) return null
        val line = f.useLines { it.firstOrNull { l -> l.startsWith("Features") } } ?: return null
        return line.substringAfter(':').trim().split(' ').filter { it in INTERESTING_CPU_FEATURES }.distinct()
    }

    /** MemAvailable/Swap/Cached from /proc/meminfo (kB -> bytes); complements ActivityManager's view. */
    private fun meminfo(o: JSONObject) {
        val f = File("/proc/meminfo")
        if (!f.canRead()) return
        f.useLines { lines ->
            for (l in lines) {
                val key = l.substringBefore(':')
                val kb = l.substringAfter(':').trim().substringBefore(' ').toLongOrNull() ?: continue
                when (key) {
                    "MemAvailable" -> o.put("procMemAvailableBytes", kb * 1024)
                    "SwapTotal" -> o.put("swapTotalBytes", kb * 1024)
                    "SwapFree" -> o.put("swapFreeBytes", kb * 1024)
                    "Cached" -> o.put("cachedBytes", kb * 1024)
                }
            }
        }
    }
}
