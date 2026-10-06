plugins { id("org.jetbrains.kotlin.jvm") }

// Pure-JVM module bundled (dexed) into the OTA product bundle. Runtime deps: kotlin-stdlib (host-provided) and the
// Android platform's own org.json ONLY. Anything else must be pure Java/Kotlin and dexed in too (avoid).
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

dependencies {
    compileOnly("org.json:json:20231013")          // on Android the platform provides org.json at runtime
    testImplementation("org.json:json:20231013")   // JVM tests need a real implementation
    testImplementation(kotlin("test"))


}

tasks.jar {
    archiveBaseName.set("extract")
    archiveVersion.set("")
}

tasks.test { useJUnit() }
