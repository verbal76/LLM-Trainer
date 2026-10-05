plugins { id("org.jetbrains.kotlin.jvm") }

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    // The platform API only; the real android.jar is on the device at runtime.
    compileOnly(files(rootProject.extra["androidJar"] as String))
}
