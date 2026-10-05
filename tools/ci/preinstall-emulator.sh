#!/usr/bin/env bash
# Install the Android Emulator package with retries. The hosted runner's download occasionally yields a corrupt zip
# ("Error on ZipFile unknown archive"); android-emulator-runner then finds the package present and skips the download.
set -u
SDKM="${ANDROID_HOME:-/usr/local/lib/android/sdk}/cmdline-tools/latest/bin/sdkmanager"
[ -x "$SDKM" ] || SDKM="$(command -v sdkmanager || true)"
[ -n "$SDKM" ] || { echo "sdkmanager not found; leaving it to android-emulator-runner"; exit 0; }
# usage: preinstall-emulator.sh [system-image-package ...]   e.g. 'system-images;android-36;google_apis_ps16k;x86_64'
ROOT="${ANDROID_HOME:-/usr/local/lib/android/sdk}"
install_pkg() { # <package> <dir that must exist afterwards>
  local pkg="$1" dir="$2" out
  for attempt in 1 2 3 4; do
    yes | "$SDKM" --licenses >/dev/null 2>&1
    out=$("$SDKM" --install "$pkg" --channel=0 2>&1)
    if ! echo "$out" | grep -qi "error on zipfile\|unknown archive\|failed" && [ -e "$dir" ]; then
      echo "$pkg installed (attempt $attempt)"; return 0
    fi
    echo "$pkg install attempt $attempt failed: $(echo "$out" | tail -n 2)"
    rm -rf "$dir"
    sleep $((attempt * 5))
  done
  return 1
}
install_pkg emulator "$ROOT/emulator/emulator" || true
for img in "$@"; do
  # 'system-images;android-36;google_apis;x86_64' -> $ROOT/system-images/android-36/google_apis/x86_64
  install_pkg "$img" "$ROOT/$(echo "$img" | tr ';' '/')" || true
done
exit 0
