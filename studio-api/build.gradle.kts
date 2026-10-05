plugins { id("org.jetbrains.kotlin.jvm") }

// Pure-JVM module bundled (dexed) into the OTA product bundle. Runtime deps: kotlin-stdlib (host-provided) and the
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
    testImplementation(kotlin("test"))


}

tasks.jar {
    archiveBaseName.set("studio-api")
    archiveVersion.set("")
}

tasks.test { useJUnit() }
