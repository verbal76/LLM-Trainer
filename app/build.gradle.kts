import java.io.File

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

fun prop(name: String, default: String) = (findProperty(name) as String?) ?: default

// Native v2 generation: APK versionName "2" (see docs/VERSIONING.md). Releases pass explicit -P values.
val hostVersionCode = prop("hostVersionCode", "2").toInt()
val hostVersionName = prop("hostVersionName", "2")
// The built-in layer of native v2 is OTA sequence #3: strictly newer than every v1-era slot (#1, #2), so
// UpdateStore.reconcileBuiltin drops stale v1 OTA slots on first v2 launch.
val builtinBundleVersion = prop("bundleVersion", "3").toInt()
// Source identity: CI exports GITHUB_SHA; local builds ask the repository; never fail the build over it.
val gitSha: String = (System.getenv("GITHUB_SHA")?.take(12)?.takeIf { it.isNotBlank() }
    ?: runCatching {
        providers.exec { commandLine("git", "rev-parse", "--short=12", "HEAD"); isIgnoreExitValue = true }
            .standardOutput.asText.get().trim()
    }.getOrNull()?.takeIf { it.isNotBlank() }
    ?: "unknown")
val otaKeyId = prop("otaKeyId", File(rootDir, "ota/keys/prod.keyid").readText().trim())
val otaPublicKey = File(prop("otaPublicKeyFile", File(rootDir, "ota/keys/prod.pub").path)).readText().trim()

android {
    namespace = "com.hotatticgames.llmtrainer"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.hotatticgames.llmtrainer"
        minSdk = 26
        targetSdk = 36
        versionCode = hostVersionCode
        versionName = hostVersionName
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Native runtime identity. Bump NATIVE_ABI whenever JNI surface or shipped .so files change
        // incompatibly: OTA bundles pin it exactly, so they can never reach an incompatible runtime.
        // 2 = the HAG engine runtime (libhagrt/libhagengine + host API level 2). Native v1 hosts are ABI 1.
        buildConfigField("int", "NATIVE_ABI", "2")
        // Fallback only: the live runtime id is the engine's own hag_engine_version() string (HostRuntime.hostInfo).
        buildConfigField("String", "NATIVE_RUNTIME_ID", "\"engine-unavailable\"")
        buildConfigField("String", "GIT_SHA", "\"$gitSha\"")
        buildConfigField("int", "BUILTIN_BUNDLE_VERSION", "$builtinBundleVersion")
        buildConfigField("String", "OTA_KEY_ID", "\"$otaKeyId\"")
        buildConfigField("String", "OTA_PUBLIC_KEY", "\"$otaPublicKey\"")
        buildConfigField(
            "String", "OTA_CHANNEL_URL",
            "\"https://github.com/verbal76/LLM-Trainer/releases/download/ota-stable/channel.json\"",
        )
    }
    buildFeatures { buildConfig = true }

    signingConfigs {
        val ks = System.getenv("ANDROID_KEYSTORE_FILE")
        if (ks != null) {
            create("release") {
                storeFile = file(ks)
                storePassword = System.getenv("ANDROID_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("ANDROID_KEY_ALIAS")
                keyPassword = System.getenv("ANDROID_KEY_PASSWORD")
            }
        }
    }
    packaging {
        // Native libs stay uncompressed + page-aligned inside the APK (mandatory for 16 KB page-size devices).
        jniLibs { useLegacyPackaging = false }
    }
    buildTypes {
        release {
            // Phones only: the x86_64 engine build exists for emulator qualification (debug/test APK) and never ships.
            ndk { abiFilters += "arm64-v8a" }
            isMinifyEnabled = false // bundles resolve kotlin-stdlib + host API from the host: never strip them
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    prop("otaFixturesDir", "").takeIf { it.isNotEmpty() }?.let { sourceSets["androidTest"].assets.srcDir(it) }
}

dependencies {
    implementation(project(":ota-core"))
    api(project(":host-api"))
    implementation(project(":runtime"))
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:core:1.6.1")
}

// Exact canonical studio logo + the built-in OTA bundle, copied byte-for-byte into assets/.
// Registered through the AGP variant API so every consumer of the assets (merge, lint, ...) depends on it.
abstract class StageHagAssets : DefaultTask() {
    @get:InputFile abstract val logo: RegularFileProperty
    @get:InputFile abstract val builtinBundle: RegularFileProperty
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun stage() {
        val out = outputDir.get().asFile
        out.deleteRecursively()
        File(out, "branding").mkdirs()
        File(out, "builtin").mkdirs()
        logo.get().asFile.copyTo(File(out, "branding/studio-logo.png"), overwrite = true)
        builtinBundle.get().asFile.copyTo(File(out, "builtin/llmtrainer-main.hagb"), overwrite = true)
    }
}

val stageHagAssets = tasks.register<StageHagAssets>("stageHagAssets") {
    dependsOn(":bundle:packBundle")
    logo.set(File(rootDir, "branding/hot-attic-games/studio-logo.png"))
    builtinBundle.set(project(":bundle").layout.buildDirectory.file("ota/LLM-Trainer-bundle-$builtinBundleVersion.hagb"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(stageHagAssets, StageHagAssets::outputDir)
    }
}
