plugins { id("org.jetbrains.kotlin.jvm") }

// Pure-JVM module bundled (dexed) into the OTA product bundle: implements the studio-api engine seams (InferenceBackend /
// TrainingBackend) over the host's EngineApi (host-api level 2). Runtime deps: kotlin-stdlib (host-provided), studio-api and the
// Android platform's own org.json. No android.* types, no native code, so it is unit-tested on a plain JVM with a fake EngineApi.
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        freeCompilerArgs.addAll("-Xstring-concat=inline", "-Xlambdas=class", "-Xsam-conversions=class")
    }
}
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

// EngineApi lives in host-api (the stable host <-> bundle contract). host-api needs the Android SDK to build, but its engine half is a
// single file without android.* imports, so this module compiles THAT FILE (copied at build time, never edited) for its own
// compile/test classpath only. The classes are excluded from this module's jar: at runtime the bundle gets the host's EngineApi
// through the parent class loader, and a second copy would make host and bundle disagree about the type.
val engineApiSrc = rootProject.layout.projectDirectory.file("host-api/src/main/kotlin/com/hotatticgames/llmtrainer/hostapi/EngineApi.kt")
val engineApiOut = layout.buildDirectory.dir("generated/engine-api/kotlin")
val syncEngineApi by tasks.registering(Sync::class) {
    from(engineApiSrc)
    into(engineApiOut)
}
sourceSets.main { kotlin.srcDir(engineApiOut) }
tasks.named("compileKotlin") { dependsOn(syncEngineApi) }

dependencies {
    compileOnly("org.json:json:20231013")          // on Android the platform provides org.json at runtime
    testImplementation("org.json:json:20231013")   // JVM tests need a real implementation
    testImplementation(kotlin("test"))
    api(project(":studio-api"))
    testImplementation(project(":studio-core"))   // smoke test: the real StudioCore wired to the adapter over a fake engine
}

tasks.jar {
    archiveBaseName.set("engine-adapter")
    archiveVersion.set("")
    exclude("com/hotatticgames/llmtrainer/hostapi/**")
}

tasks.test { useJUnit() }
