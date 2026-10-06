plugins { id("org.jetbrains.kotlin.jvm") }

// Pure-JVM device-qualification core (ADR 0011: Kotlin is the on-device authority, Python the executable spec).
// Runtime dependencies: kotlin-stdlib ONLY (the OTA host provides it). The bundle dexes this jar next to its
// own classes, so nothing here may need a library the host does not ship. kotlinx-serialization is test-only.
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        // Same flags as :bundle: no invokedynamic, so d8 desugaring stays trivial.
        freeCompilerArgs.addAll("-Xstring-concat=inline", "-Xlambdas=class", "-Xsam-conversions=class")
    }
}
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
}

tasks.jar {
    archiveBaseName.set("qualify")
    archiveVersion.set("")
}

tasks.test {
    useJUnit()
    // The SAME golden file the Python spec generates (factory/tools/gen_golden_device.py).
    systemProperty("golden.path", rootProject.file("factory/tests/golden/device_qualification.v1.json").path)
    // Device profiler v2 (per-artifact capabilities): factory/tools/gen_golden_capability.py
    systemProperty("golden.capability.path", rootProject.file("factory/tests/golden/device_capability.v1.json").path)
}
