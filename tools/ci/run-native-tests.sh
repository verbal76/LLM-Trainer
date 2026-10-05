#!/usr/bin/env bash
# Runs the REAL-engine instrumented tests on the already-booted emulator (called as ONE line from android-emulator-runner: that action runs each script line in its own shell).
#   env: MODELS_DIR (dir with model-q8_0.gguf + model-f16.gguf), EXPECT_PAGE_SIZE (optional, e.g. 16384),
#        CI_KEY (prefix of the ephemeral OTA key), FIXTURES (dir containing ota-fixtures/), HAG_ABIS (default x86_64)
set -uo pipefail
MODELS_DIR=${MODELS_DIR:-$HOME/hag-models}
: "${CI_KEY:?}" "${FIXTURES:?}"
ABIS=${HAG_ABIS:-x86_64}

echo "== device"
adb shell getprop ro.build.fingerprint
adb shell getprop ro.build.version.sdk
PS=$(adb shell getconf PAGE_SIZE | tr -d '\r')
echo "PAGE_SIZE=$PS  (uname: $(adb shell uname -m | tr -d '\r'), kernel: $(adb shell uname -r | tr -d '\r'))"
adb shell cat /proc/cpuinfo | grep -m1 -E 'flags|Features' | cut -c1-400 || true
adb shell cat /proc/meminfo | head -2
adb shell df -h /data | tail -1
if [ -n "${EXPECT_PAGE_SIZE:-}" ] && [ "$PS" != "$EXPECT_PAGE_SIZE" ]; then
  echo "::error::expected PAGE_SIZE=$EXPECT_PAGE_SIZE but device reports $PS"; exit 1
fi

echo "== pushing verified models"
adb shell mkdir -p /data/local/tmp/hag
adb push "$MODELS_DIR/model-q8_0.gguf" /data/local/tmp/hag/model-q8_0.gguf
adb push "$MODELS_DIR/model-f16.gguf" /data/local/tmp/hag/model-f16.gguf
adb shell ls -l /data/local/tmp/hag

echo "== instrumented tests (real engine, real models)"
adb logcat -c
RUNNER="-Pandroid.testInstrumentationRunnerArguments"
./gradlew --no-daemon --continue -Phag.abis="$ABIS" :runtime:connectedDebugAndroidTest :app:connectedDebugAndroidTest \
  "$RUNNER.hagRequireModels=true" "$RUNNER.hagModelQ8=/data/local/tmp/hag/model-q8_0.gguf" \
  "$RUNNER.hagModelF16=/data/local/tmp/hag/model-f16.gguf" \
  ${GIT_SHA_SHORT:+-PgitSha=$GIT_SHA_SHORT} -PotaPrivateKeyFile="$CI_KEY.key" -PotaPublicKeyFile="$CI_KEY.pub" -PotaKeyId=ci -PotaFixturesDir="$FIXTURES"
rc=$?

echo "== diagnostics (rc=$rc)"
adb logcat -d -s HagTest:V HagEngine:V HagOta:V HagHost:V AndroidRuntime:E DEBUG:V libc:F | tail -150
adb logcat -d -b crash | tail -60
adb shell dmesg 2>/dev/null | grep -iE 'lowmemorykiller|oom|killed process' | tail -10 || true
exit $rc
