// Shared SDK lookup for modules compiled against android.jar without AGP (host-api, bundle).
val sdkRoot: String? = System.getenv("ANDROID_HOME") ?: System.getenv("ANDROID_SDK_ROOT")
    ?: file("local.properties").takeIf { it.exists() }?.readLines()
        ?.firstOrNull { it.startsWith("sdk.dir=") }?.removePrefix("sdk.dir=")
extra["androidSdkRoot"] = sdkRoot
extra["androidCompileSdk"] = 35
extra["androidJar"] = sdkRoot?.let { "$it/platforms/android-35/android.jar" }
