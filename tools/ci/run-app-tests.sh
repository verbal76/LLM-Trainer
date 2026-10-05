#!/usr/bin/env bash
# App-level instrumented tests (OTA qualification + host/engine wiring) on the already-booted emulator.
# Single entry point because android-emulator-runner runs every script line in a separate shell: a failure must come out
# of THIS script's exit status. env: CI_KEY (ephemeral OTA key prefix), FIXTURES (dir containing ota-fixtures/)
set -uo pipefail
: "${CI_KEY:?}" "${FIXTURES:?}"
adb shell getprop ro.build.fingerprint
echo "PAGE_SIZE=$(adb shell getconf PAGE_SIZE | tr -d '\r')"
adb logcat -c
./gradlew --no-daemon :app:connectedDebugAndroidTest \
  -PotaPrivateKeyFile="$CI_KEY.key" -PotaPublicKeyFile="$CI_KEY.pub" -PotaKeyId=ci -PotaFixturesDir="$FIXTURES"
rc=$?
adb logcat -d -s HagOta:V HagEngine:V AndroidRuntime:E DEBUG:V libc:F | tail -150
echo "app instrumented tests exit status: $rc"
exit $rc
