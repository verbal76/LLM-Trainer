package com.hotatticgames.llmtrainer.host

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import org.json.JSONObject

/** Cheap point-in-time device facts. Feeds device qualification (static facts only; no benchmarking here). */
object DeviceProbe {
    fun snapshotJson(ctx: Context): String {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val st = StatFs(ctx.filesDir.path)
        val o = JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("sdkInt", Build.VERSION.SDK_INT)
            .put("abis", Build.SUPPORTED_ABIS.joinToString(","))
            .put("totalRamBytes", mi.totalMem)
            .put("availRamBytes", mi.availMem)
            .put("lowMemoryThresholdBytes", mi.threshold)
            .put("lowMemory", mi.lowMemory)
            .put("isLowRamDevice", am.isLowRamDevice)
            .put("memoryClassMb", am.memoryClass)
            .put("freeStorageBytes", st.availableBytes)
            .put("totalStorageBytes", st.totalBytes)
            .put("cpuCores", Runtime.getRuntime().availableProcessors())
        if (Build.VERSION.SDK_INT >= 31) o.put("socModel", Build.SOC_MODEL)
        if (Build.VERSION.SDK_INT >= 29) {
            val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
            o.put("thermalStatus", pm.currentThermalStatus)
        }
        return o.toString()
    }
}
