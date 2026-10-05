#!/usr/bin/env bash
# Verify a built APK: signer, identity, exact studio logo, built-in OTA bundle verifies against the trusted key.
# usage: verify-apk.sh <apk> <ota-public-key-file> <ota-key-id> <expected-version-name> [--allow-debug-signer]
set -euo pipefail
APK=$1; PUB=$2; KEYID=$3; VNAME=$4; ALLOW_DEBUG=${5:-}
LOGO_SHA=e3d9bb5653eafb783eede827606e7ac73a4e45564a1c25b1ed13ad1429f48c4e
BT=$(ls -d "$ANDROID_HOME"/build-tools/* | sort -V | tail -1)

"$BT/apksigner" verify --print-certs "$APK" | tee signer.txt
if grep -q "CN=Android Debug" signer.txt && [ "$ALLOW_DEBUG" != "--allow-debug-signer" ]; then
  echo "::error::APK is signed with the debug key"; exit 1
fi
"$BT/aapt2" dump badging "$APK" | tee badging.txt | grep -E "^package:|application-label:"
grep -q "package: name='com.hotatticgames.llmtrainer'" badging.txt
grep -q "versionName='$VNAME'" badging.txt || { echo "::error::versionName != $VNAME"; exit 1; }

unzip -p "$APK" assets/branding/studio-logo.png | sha256sum | grep -q "$LOGO_SHA" \
  || { echo "::error::studio logo in APK is not the canonical file"; exit 1; }
echo "logo OK ($LOGO_SHA)"

mkdir -p chk && unzip -p "$APK" assets/builtin/llmtrainer-main.hagb > chk/builtin.hagb
./gradlew --no-daemon -q :ota-core:run --args="verify --bundle $PWD/chk/builtin.hagb --pub $(realpath $PUB) --key-id $KEYID --host-api 2 --native-abi 2 --cap core.v1 --cap device.snapshot.v1 --cap update.check.v1 --cap inference.gguf.v1 --cap training.patch.v1 --sdk 34 --bundle-id llmtrainer-main --channel stable"
echo "built-in bundle verifies against trusted key $KEYID"

# Native v2 identity: minSdk 26, targetSdk 36 (16 KB packaging is qualified separately by tools/ci/check-16kb.sh).
grep -q "sdkVersion:'26'" badging.txt || { echo "::error::minSdk != 26"; exit 1; }
grep -q "targetSdkVersion:'36'" badging.txt || { echo "::error::targetSdk != 36"; exit 1; }
echo "package/minSdk/targetSdk OK"
