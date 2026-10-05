#!/usr/bin/env bash
# Install-over gate: the PUBLISHED previous native APK is installed and run (so it creates real app data), then the
# release candidate is installed OVER it with `adb install -r` (no uninstall). Asserts:
#   - same signing certificate (otherwise Android refuses the update; we also compare explicitly);
#   - versionCode/versionName of the installed package are the candidate's;
#   - the OTA root of the previous app survived (data sanity) and the candidate boots its built-in layer healthily
#     (logcat 'healthy: builtin v<seq>') without a crash, with no stale previous-ABI slot left behind.
# Needs a rootable emulator (google_apis image): `adb root` is used only to read the app's own data directory.
#
# env: WANT_ABI (default 2) native ABI of the candidate
# usage: install-over-test.sh <previous.apk> <candidate.apk> <expected-versionCode> <expected-builtin-sequence>
set -uo pipefail
PREV=${1:?previous apk}; CAND=${2:?candidate apk}; WANT_CODE=${3:?versionCode}; WANT_SEQ=${4:?builtin OTA sequence}
PKG=com.hotatticgames.llmtrainer
fail() { echo "::error::install-over: $*"; adb logcat -d -s HagOta:V AndroidRuntime:E 2>/dev/null | tail -60; exit 1; }
BT=$(ls -d "$ANDROID_HOME"/build-tools/* | sort -V | tail -1)

cert() { "$BT/apksigner" verify --print-certs "$1" | grep -m1 "certificate SHA-256 digest" | sed 's/.*: *//'; }
[ "$(cert "$PREV")" = "$(cert "$CAND")" ] || fail "signer differs between previous ($(cert "$PREV")) and candidate ($(cert "$CAND"))"
echo "same signer: $(cert "$CAND")"

adb root >/dev/null 2>&1; sleep 2
adb uninstall $PKG >/dev/null 2>&1 || true
adb install "$PREV" >/dev/null || fail "cannot install the previous APK"
adb logcat -c
adb shell am start -n $PKG/.host.HostActivity >/dev/null; sleep 12
adb shell pidof $PKG >/dev/null || fail "previous APK is not running"
adb shell "ls /data/data/$PKG/files/ota/state.json" >/dev/null 2>&1 || echo "note: previous app wrote no state.json yet (clean built-in boot)"
PREV_CODE=$(adb shell dumpsys package $PKG | grep -m1 versionCode= | sed 's/.*versionCode=\([0-9]*\).*/\1/')
echo "previous versionCode=$PREV_CODE"
adb shell "echo marker > /data/data/$PKG/files/install-over-marker.txt" || fail "cannot write the data marker"

adb shell am force-stop $PKG
OUT=$(adb install -r "$CAND" 2>&1) || fail "adb install -r failed: $OUT"
echo "$OUT"
DUMP=$(adb shell dumpsys package $PKG)
echo "$DUMP" | grep -q "versionCode=$WANT_CODE " || echo "$DUMP" | grep -q "versionCode=$WANT_CODE$" || fail "installed versionCode != $WANT_CODE"
[ "$WANT_CODE" -gt "${PREV_CODE:-0}" ] || fail "candidate versionCode $WANT_CODE is not above the previous $PREV_CODE"
adb shell "cat /data/data/$PKG/files/install-over-marker.txt" | grep -q marker || fail "app data did not survive the upgrade"

adb logcat -c
adb shell am start -n $PKG/.host.HostActivity >/dev/null
ok=0
for i in $(seq 1 30); do
  sleep 2
  if adb logcat -d -s HagOta:V | grep -q "healthy: builtin v$WANT_SEQ"; then ok=1; break; fi
  adb logcat -d -s HagOta:V | grep -q "healthy: ota" && { ok=1; break; }
done
[ $ok = 1 ] || fail "upgraded app did not reach a healthy first frame (expected 'healthy: builtin v$WANT_SEQ')"
adb logcat -d -s AndroidRuntime:E | grep -q "FATAL EXCEPTION" && fail "crash after upgrade"
adb shell pidof $PKG >/dev/null || fail "upgraded app is not running"
# A slot left by the previous native generation must not be selectable by the new one.
STATE=$(adb shell "cat /data/data/$PKG/files/ota/state.json" 2>/dev/null || echo '{}')
echo "state.json after upgrade: $STATE" | head -c 600; echo
echo "$STATE" | WANT_ABI=${WANT_ABI:-2} python3 -c '
import json, os, sys
s = json.load(sys.stdin)
bad = [k for k, v in s.get("slots", {}).items() if v.get("nativeAbi", 1) != int(os.environ["WANT_ABI"])]
assert not bad, "stale previous-ABI slots survived the upgrade: %s" % bad
' || fail "stale slot after upgrade"
echo "INSTALL-OVER OK: $PREV_CODE -> $WANT_CODE, same signer, data intact, builtin healthy"
