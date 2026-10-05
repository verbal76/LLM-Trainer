import java.io.File

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

fun prop(name: String, default: String) = (findProperty(name) as String?) ?: default

val hostVersionCode = prop("hostVersionCode", "1").toInt()
val hostVersionName = prop("hostVersionName", "1")
val builtinBundleVersion = prop("bundleVersion", "1").toInt()
val otaKeyId = prop("otaKeyId", File(rootDir, "ota/keys/prod.keyid").readText().trim())
val otaPublicKey = File(prop("otaPublicKeyFile", File(rootDir, "ota/keys/prod.pub").path)).readText().trim()

android {
    namespace = "com.hotatticgames.llmtrainer"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.hotatticgames.llmtrainer"
        minSdk = 26
        targetSdk = 35
        versionCode = hostVersionCode
        versionName = hostVersionName
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Native runtime identity. Bump NATIVE_ABI whenever JNI surface or shipped .so files change
        // incompatibly: OTA bundles pin it exactly, so they can never reach an incompatible runtime.
        buildConfigField("int", "NATIVE_ABI", "1")
        buildConfigField("String", "NATIVE_RUNTIME_ID", "\"none-v1\"")
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
    buildTypes {
        release {
            isMinifyEnabled = false // bundles resolve kotlin-stdlib + host API from the host: never strip them
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/hag-assets"))
    prop("otaFixturesDir", "").takeIf { it.isNotEmpty() }?.let { sourceSets["androidTest"].assets.srcDir(it) }
}

dependencies {
    implementation(project(":ota-core"))
    api(project(":host-api"))
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}

// Exact canonical studio logo + the built-in OTA bundle, copied byte-for-byte into assets/.
val stageHagAssets = tasks.register<Copy>("stageHagAssets") {
    dependsOn(":bundle:packBundle")
    into(layout.buildDirectory.dir("generated/hag-assets"))
    from(File(rootDir, "branding/hot-attic-games/studio-logo.png")) { into("branding") }
    from(project(":bundle").layout.buildDirectory.file("ota/LLM-Trainer-bundle-$builtinBundleVersion.hagb")) {
        into("builtin"); rename { "llmtrainer-main.hagb" }
    }
}
tasks.matching { it.name.matches(Regex("merge.*Assets")) || it.name.matches(Regex("lint.*")) }
    .configureEach { dependsOn(stageHagAssets) }
