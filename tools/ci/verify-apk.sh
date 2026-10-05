#!/usr/bin/env bash
# Verify a built APK: identity, signer, SDK levels, exact studio logo, built-in OTA bundle, (optionally) native libs.
#
# usage: verify-apk.sh <apk> <ota-public-key-file> <ota-key-id> <expected-version-name> [options]
#   --version-code <n>      assert versionCode == n (release: n == the native version)
#   --pin-signer            assert the signer certificate is the permanent LLM Trainer identity (release builds)
#   --allow-debug-signer    do not fail on the Android debug certificate (CI builds only)
#   --require-native-libs   assert lib/arm64-v8a/*.so is present (release builds)
# env: NATIVE_ABI (default 2)  expected nativeAbi of the host and of the built-in bundle
#      EXPECT_SIGNER_SHA256    override the pinned signer (lowercase hex, no colons) - tests only
#
# Writes (cwd): signer.txt badging.txt chk/builtin.hagb
set -euo pipefail
APK=$1; PUB=$2; KEYID=$3; VNAME=$4; shift 4
VCODE=""; PIN=0; ALLOW_DEBUG=0; NEED_LIBS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --version-code) VCODE=$2; shift 2 ;;
    --pin-signer) PIN=1; shift ;;
    --allow-debug-signer) ALLOW_DEBUG=1; shift ;;
    --require-native-libs) NEED_LIBS=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
NATIVE_ABI=${NATIVE_ABI:-2}
LOGO_SHA=e3d9bb5653eafb783eede827606e7ac73a4e45564a1c25b1ed13ad1429f48c4e
# Permanent signing identity of LLM Trainer (same as v1). C9:02:BC:... without colons, lowercase.
PERMANENT_SIGNER=c902bc4193820df592e5438c5af5fa93669f34a9fc2b55bef081c9f2f57411c5
EXPECT_SIGNER=${EXPECT_SIGNER_SHA256:-$PERMANENT_SIGNER}
fail() { echo "::error::$*"; exit 1; }

if [ -n "${ANDROID_HOME:-}" ]; then
  BT=$(ls -d "$ANDROID_HOME"/build-tools/* | sort -V | tail -1)
else
  BT=$(dirname "$(command -v apksigner)")
fi

# ---- signer ---------------------------------------------------------------------------------------------------
"$BT/apksigner" verify --verbose --print-certs "$APK" | tee signer.txt
if grep -q "CN=Android Debug" signer.txt && [ "$ALLOW_DEBUG" != 1 ]; then fail "APK is signed with the debug key"; fi
SIGNER=$(grep -m1 "certificate SHA-256 digest" signer.txt | sed 's/.*: *//' | tr -d ':' | tr 'A-F' 'a-f')
[ -n "$SIGNER" ] || fail "could not read signer certificate digest"
echo "signer SHA-256: $SIGNER"
if [ "$PIN" = 1 ]; then
  [ "$SIGNER" = "$EXPECT_SIGNER" ] || fail "signer $SIGNER != permanent identity $EXPECT_SIGNER"
  echo "signer matches the permanent LLM Trainer identity"
fi
grep -q "Number of signers: 1" signer.txt || fail "expected exactly one signer"

# ---- identity / sdk levels ------------------------------------------------------------------------------------
"$BT/aapt2" dump badging "$APK" > badging.txt
grep -E "^package:|application-label:|sdkVersion|targetSdkVersion" badging.txt
grep -q "package: name='com.hotatticgames.llmtrainer'" badging.txt || fail "wrong package"
grep -q "versionName='$VNAME'" badging.txt || fail "versionName != $VNAME"
if [ -n "$VCODE" ]; then grep -q "versionCode='$VCODE'" badging.txt || fail "versionCode != $VCODE"; fi
grep -q "^sdkVersion:'26'" badging.txt || fail "minSdk != 26"
grep -q "^targetSdkVersion:'36'" badging.txt || fail "targetSdk != 36"

# ---- exact canonical studio logo --------------------------------------------------------------------------------
unzip -p "$APK" assets/branding/studio-logo.png | sha256sum | grep -q "$LOGO_SHA" \
  || fail "studio logo in APK is not the canonical file"
echo "logo OK ($LOGO_SHA)"

# ---- built-in OTA bundle ----------------------------------------------------------------------------------------
mkdir -p chk && unzip -p "$APK" assets/builtin/llmtrainer-main.hagb > chk/builtin.hagb
./gradlew --no-daemon -q :ota-core:run --args="verify --bundle $PWD/chk/builtin.hagb --pub $(realpath "$PUB") --key-id $KEYID --host-api 1 --native-abi $NATIVE_ABI --cap core.v1 --cap device.snapshot.v1 --cap update.check.v1 --sdk 34 --bundle-id llmtrainer-main --channel stable"
echo "built-in bundle verifies against trusted key $KEYID (native ABI $NATIVE_ABI)"
# Version identity of the layer inside the APK: '<native>.<minor>' whose major is the native version, built-in is .0
unzip -p chk/builtin.hagb manifest.json > chk/builtin-manifest.json
python3 - "$VNAME" "$NATIVE_ABI" <<'PY' || fail "built-in bundle identity inconsistent with the native version"
import json, sys
vname, abi = sys.argv[1], int(sys.argv[2])
m = json.load(open('chk/builtin-manifest.json'))
name = m['bundleVersionName']
print('built-in application version:', name, '| OTA sequence:', m['bundleVersion'], '| nativeAbi:', m['requires']['nativeAbi'])
assert m['requires']['nativeAbi'] == abi, 'built-in bundle nativeAbi != host abi'
if vname.isdigit():
    assert name.split('.')[0] == vname, 'application major %s != native version %s' % (name.split('.')[0], vname)
    assert name == vname + '.0', 'the layer built into the APK must be <native>.0, got ' + name
PY

# ---- native libs ------------------------------------------------------------------------------------------------
if unzip -l "$APK" | grep -E "lib/arm64-v8a/[^/]+\.so" ; then
  echo "arm64-v8a native libs present"
elif [ "$NEED_LIBS" = 1 ]; then
  fail "no lib/arm64-v8a/*.so in the APK (native engine missing)"
else
  echo "note: no native libs in this build (not required here)"
fi
echo "verify-apk: ALL CHECKS PASSED for $APK"
