#!/usr/bin/env bash
# Install the Android Emulator package with retries. The hosted runner's download occasionally yields a corrupt zip
# ("Error on ZipFile unknown archive"); android-emulator-runner then finds the package present and skips the download.
set -u
SDKM="${ANDROID_HOME:-/usr/local/lib/android/sdk}/cmdline-tools/latest/bin/sdkmanager"
[ -x "$SDKM" ] || SDKM="$(command -v sdkmanager || true)"
[ -n "$SDKM" ] || { echo "sdkmanager not found; leaving it to android-emulator-runner"; exit 0; }
for attempt in 1 2 3 4; do
  yes | "$SDKM" --licenses >/dev/null 2>&1
  out=$("$SDKM" --install emulator --channel=0 2>&1)
  if ! echo "$out" | grep -qi "error on zipfile\|unknown archive\|failed"; then
    if [ -x "${ANDROID_HOME:-/usr/local/lib/android/sdk}/emulator/emulator" ]; then echo "emulator installed (attempt $attempt)"; exit 0; fi
  fi
  echo "emulator install attempt $attempt failed: $(echo "$out" | tail -n 2)"
  rm -rf "${ANDROID_HOME:-/usr/local/lib/android/sdk}/emulator"
  sleep $((attempt * 5))
done
echo "emulator pre-install did not succeed; android-emulator-runner will try once more"
exit 0
