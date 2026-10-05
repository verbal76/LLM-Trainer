import java.io.File

// Generic, reusable HAG on-device LLM runtime: native engine (native/engine + llama.cpp, CPU only) + JNI + Kotlin binding.
// Nothing here is LLM-Trainer specific; the app module adapts it to its host API.
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

fun prop(name: String, default: String) = (findProperty(name) as String?) ?: default

val ndkPin = prop("hag.ndkVersion", "28.0.13004108")
val cmakePin = prop("hag.cmakeVersion", "3.22.1")
// ABIs built. Phones are arm64-v8a; x86_64 exists for emulator qualification (CI) only. The release APK additionally
// filters to arm64-v8a in app/build.gradle.kts, so x86_64 code never ships to users.
val hagAbis = prop("hag.abis", "arm64-v8a,x86_64").split(',').map { it.trim() }.filter { it.isNotEmpty() }

val engineDir = File(rootDir, "native/engine")
val fetchScript = File(rootDir, "native/fetch-llama.sh")
// LLAMA_CPP_SRC (a ready llama.cpp checkout) wins; otherwise native/fetch-llama.sh populates a cache inside build/.
val llamaSrcEnv: String? = System.getenv("LLAMA_CPP_SRC")?.takeIf { it.isNotBlank() }
val llamaDir: File = llamaSrcEnv?.let { File(it) } ?: layout.buildDirectory.dir("llama.cpp").get().asFile

android {
    namespace = "com.hotatticgames.hag.runtime"
    compileSdk = 36
    ndkVersion = ndkPin

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        ndk { abiFilters += hagAbis }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON", // 16 KB page-size compatible (default in NDK r28; explicit for clarity)
                    "-DHAG_ENGINE_DIR=${engineDir.absolutePath}",
                    "-DLLAMA_CPP_DIR=${llamaDir.absolutePath}",
                )
            }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = cmakePin
        }
    }
    packaging {
        // Uncompressed + page-aligned .so files, mapped straight from the APK (required for 16 KB devices).
        jniLibs { useLegacyPackaging = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    testOptions { targetSdk = 36 }
}

dependencies {
    testImplementation(kotlin("test"))
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:core:1.6.1")
}

// llama.cpp is fetched, never vendored. native/fetch-llama.sh <dest> pins the commit and is idempotent.
val fetchLlamaCpp = tasks.register<Exec>("fetchLlamaCpp") {
    group = "hag"
    description = "Fetch the pinned llama.cpp checkout (or validate \$LLAMA_CPP_SRC)"
    if (llamaSrcEnv == null) {
        inputs.file(fetchScript) // the pin lives in the script: a new pin re-runs the fetch
        outputs.dir(llamaDir)
        outputs.upToDateWhen { File(llamaDir, "CMakeLists.txt").isFile }
        commandLine("bash", fetchScript.absolutePath, llamaDir.absolutePath)
    } else {
        commandLine(
            "bash", "-c",
            "test -f '${llamaDir.absolutePath}/CMakeLists.txt' || { echo 'LLAMA_CPP_SRC=${llamaDir.absolutePath} is not a llama.cpp checkout' >&2; exit 1; }",
        )
    }
}

tasks.configureEach {
    if (name.startsWith("configureCMake") || name.startsWith("generateJsonModel") || name.startsWith("buildCMake")) {
        dependsOn(fetchLlamaCpp)
    }
}
