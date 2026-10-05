import java.io.File

plugins { id("org.jetbrains.kotlin.jvm") }

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        // No invokedynamic: keeps d8 desugaring trivial and bundle dex self-contained.
        freeCompilerArgs.addAll("-Xstring-concat=inline", "-Xlambdas=class", "-Xsam-conversions=class")
    }
}
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

val bundledModules = listOf(":qualify", ":extract", ":studio-api", ":studio-core", ":engine-adapter")
val androidJar = rootProject.extra["androidJar"] as String
val sdkRoot = rootProject.extra["androidSdkRoot"] as String

dependencies {
    compileOnly(project(":host-api"))
    // Dependency-free (kotlin-stdlib only) qualifier; its classes are dexed INTO the bundle (see dexBundle).
    // Pure-JVM modules dexed into the bundle (see dexBundle). Order irrelevant; all are stdlib/org.json-only.
    for (m in bundledModules) compileOnly(project(m))
    compileOnly("org.json:json:20231013")
    compileOnly(files(androidJar))
}

// ---- Build parameters (all overridable with -P) -------------------------------------------------
fun prop(name: String, default: String) = (findProperty(name) as String?) ?: default
val bundleVersion = prop("bundleVersion", "3")
val bundleVersionName = prop("bundleVersionName", "2.0") // convention <native>.<minor>; see docs/VERSIONING.md + ota/native-map.json
val faultMode = prop("faultMode", "none") // none | entry_throws | selftest_fails | slow_health
val hostApiMin = prop("hostApiMin", "1")
val hostApiMax = prop("hostApiMax", ((hostApiMin.toInt() / 100) * 100 + 99).toString())
val nativeAbi = prop("nativeAbi", "2")
val capabilities = prop("bundleCaps", "core.v1,device.snapshot.v1,update.check.v1").split(',').filter { it.isNotBlank() }
val bundleOut = prop("bundleOut", layout.buildDirectory.file("ota/LLM-Trainer-bundle-$bundleVersion.hagb").get().asFile.path)

val genDir = layout.buildDirectory.dir("generated/bundle-src")
val generateBuildInfo = tasks.register("generateBuildInfo") {
    val out = genDir.map { it.file("com/hotatticgames/llmtrainer/app/BuildInfo.kt") }
    inputs.property("v", bundleVersion); inputs.property("n", bundleVersionName); inputs.property("f", faultMode)
    outputs.file(out)
    doLast {
        out.get().asFile.apply { parentFile.mkdirs() }.writeText(
            """
            package com.hotatticgames.llmtrainer.app

            internal object BuildInfo {
                const val BUNDLE_VERSION = $bundleVersion
                const val BUNDLE_VERSION_NAME = "$bundleVersionName"
                /** Test-only fault injection used by OTA qualification fixtures. Always "none" in releases. */
                // Deliberately NOT const: a const is inlined into consumers at compile time, and a stale incremental build
                // could then keep an old fault mode in MainBundle. A plain val is always read at runtime from this class.
                @JvmField val FAULT_MODE: String = "$faultMode"
            }
            """.trimIndent() + "\n",
        )
    }
}
sourceSets.main { kotlin.srcDir(genDir) }
tasks.named("compileKotlin") { dependsOn(generateBuildInfo) }

// ---- dex + sign -> .hagb ---------------------------------------------------------------------------
val otaTool by configurations.creating
dependencies { otaTool(project(":ota-core")) }

val dexBundle = tasks.register<Exec>("dexBundle") {
    dependsOn(tasks.named("jar"))
    val jarFile = tasks.named<Jar>("jar").flatMap { it.archiveFile }
    val outDir = layout.buildDirectory.dir("ota/dex-$bundleVersion")
    inputs.file(jarFile)
    inputs.property("faultMode", faultMode)
    inputs.property("bundleVersion", bundleVersion)
    inputs.files(bundledModules.map { m -> project(m).layout.buildDirectory.file("libs/${m.removePrefix(":")}.jar") })
    outputs.dir(outDir)
    doFirst {
        val bt = File(sdkRoot, "build-tools").listFiles()!!.filter { File(it, "d8").exists() }.maxByOrNull { it.name }
            ?: error("no Android build-tools with d8 found under $sdkRoot")
        outDir.get().asFile.deleteRecursively(); outDir.get().asFile.mkdirs()
        val moduleJars = bundledModules.map { m ->
            project(m).layout.buildDirectory.file("libs/${m.removePrefix(":")}.jar").get().asFile
                .also { check(it.isFile) { "module jar missing: $it" } }
        }
        val stdlib = configurations.compileClasspath.get().files.firstOrNull { it.name.startsWith("kotlin-stdlib") }
        commandLine(
            listOfNotNull(
                File(bt, "d8").path, "--release", "--min-api", "26", "--lib", androidJar,
                "--classpath", project(":host-api").layout.buildDirectory.file("libs/host-api.jar").get().asFile.path,
                stdlib?.let { "--classpath" }, stdlib?.path,
                "--output", outDir.get().asFile.path, jarFile.get().asFile.path,
            ) + moduleJars.map { it.path },
        )
    }
    dependsOn(":host-api:jar")
    for (m in bundledModules) dependsOn("$m:jar")
}

tasks.register<JavaExec>("packBundle") {
    group = "ota"
    description = "Build, dex, sign: produces a .hagb. Needs -PotaPrivateKeyFile and -PotaKeyId."
    dependsOn(dexBundle)
    classpath = otaTool
    mainClass.set("com.hotatticgames.llmtrainer.ota.tool.MainKt")
    doFirst {
        val key = (findProperty("otaPrivateKeyFile") as String?) ?: error("-PotaPrivateKeyFile=<path> is required to sign a bundle")
        args(
            listOfNotNull(
                "pack", "--dex", layout.buildDirectory.dir("ota/dex-$bundleVersion").get().asFile.path,
                "--out", bundleOut, "--key", key, "--key-id", prop("otaKeyId", "k1"),
                "--version", bundleVersion, "--version-name", bundleVersionName,
                "--bundle-id", "llmtrainer-main", "--channel", prop("otaChannel", "stable"),
                "--entry", "com.hotatticgames.llmtrainer.app.MainBundle",
                "--host-api-min", hostApiMin, "--host-api-max", hostApiMax, "--native-abi", nativeAbi,
            ) + capabilities.flatMap { listOf("--cap", it) } +
                listOfNotNull(findProperty("bundleNotes")?.let { "--notes" }, findProperty("bundleNotes") as String?),
        )
    }
}
