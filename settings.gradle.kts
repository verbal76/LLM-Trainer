pluginManagement {
    plugins {
        id("com.android.application") version "8.7.3"
        id("com.android.library") version "8.7.3"
        id("org.jetbrains.kotlin.android") version "2.0.21"
        id("org.jetbrains.kotlin.jvm") version "2.0.21"
        id("org.jetbrains.kotlin.plugin.serialization") version "2.0.21"
    }
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "LLM-Trainer"

// Pure-JVM, independently testable OTA/update core: always included.
include(":ota-core")
// Pure-JVM device-qualification core (Kotlin port of factory/llmtrainer/device.py); dexed into the OTA bundle.
include(":qualify", ":extract", ":studio-api", ":studio-core")

// Android modules need an Android SDK. CI always has one; a bare dev box may not.
val sdkDir: String? = System.getenv("ANDROID_HOME")
    ?: System.getenv("ANDROID_SDK_ROOT")
    ?: file("local.properties").takeIf { it.exists() }?.readLines()
        ?.firstOrNull { it.startsWith("sdk.dir=") }?.removePrefix("sdk.dir=")
if (sdkDir != null && file(sdkDir).isDirectory) {
    include(":host-api", ":bundle", ":app")
} else {
    logger.warn("No Android SDK found: building JVM modules only (:ota-core).")
}
