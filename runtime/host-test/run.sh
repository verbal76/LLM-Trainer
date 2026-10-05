#!/usr/bin/env bash
# Host-JVM test of the REAL JNI bridge (runtime/src/main/cpp/hag_jni.cpp) and the Kotlin binding against a STUB C engine
# (stub_engine.c implements hag_engine.h; it is never shipped). No Android SDK needed: runs with -Xcheck:jni on a plain JDK.
# Covers what emulator tests cannot localise quickly: struct/param marshalling, UTF-8 (supplementary chars), callbacks,
# exceptions from callbacks, cancel from another thread, deferred free inside callbacks, error -> exception mapping.
#   usage: runtime/host-test/run.sh            (needs gcc, g++, a JDK with include/jni.h)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
JH=${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}
[ -f "$JH/include/jni.h" ] || { echo "no jni.h under $JH"; exit 1; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/lib"
gcc -shared -fPIC -I"$ROOT/native/engine/include" "$HERE/stub_engine.c" -o "$W/lib/libhagengine.so"
g++ -std=c++17 -shared -fPIC -Wall -Wextra -Werror -I"$JH/include" -I"$JH/include/linux" -I"$ROOT/native/engine/include" \
  "$ROOT/runtime/src/main/cpp/hag_jni.cpp" -o "$W/lib/libhagrt.so" -ldl
cat > "$W/settings.gradle.kts" <<EOF
pluginManagement { plugins { id("org.jetbrains.kotlin.jvm") version "2.0.21" }
  repositories { gradlePluginPortal(); mavenCentral() } }
dependencyResolutionManagement { repositories { mavenCentral() } }
rootProject.name = "jnihost"
EOF
cat > "$W/build.gradle.kts" <<EOF
plugins { id("org.jetbrains.kotlin.jvm") }
sourceSets {
  main { kotlin.srcDir("$ROOT/runtime/src/main/kotlin") }
  test { kotlin.srcDir("$ROOT/runtime/src/test/kotlin"); kotlin.srcDir("$HERE") }
}
dependencies { testImplementation(kotlin("test")); testImplementation("junit:junit:4.13.2") }
tasks.test { useJUnit(); jvmArgs("-Djava.library.path=$W/lib", "-Xcheck:jni"); testLogging { showStandardStreams = true; events("failed","passed") } }
EOF
"$ROOT/gradlew" ${GRADLE_FLAGS:---no-daemon} -p "$W" test
