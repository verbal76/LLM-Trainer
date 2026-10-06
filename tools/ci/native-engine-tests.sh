#!/usr/bin/env bash
# Host-side native engine gate (no emulator): required by the release pipeline's `unit` job.
#   1. JNI bridge + Kotlin binding against a stub C engine on a plain JVM with -Xcheck:jni
#   2. Android-side JVM unit suites: host API rules, runtime handle/UTF-8/unavailable-path tests
# The REAL engine with REAL models runs in the emulator jobs (tools/ci/run-native-tests.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
runtime/host-test/run.sh
if [ -n "${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}" ]; then
  ./gradlew --no-daemon :host-api:test :runtime:testDebugUnitTest
else
  echo "note: no Android SDK; :host-api:test and :runtime:testDebugUnitTest are covered by the android CI job"
fi
