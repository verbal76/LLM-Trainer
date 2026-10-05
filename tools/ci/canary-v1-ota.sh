#!/usr/bin/env bash
# Post-publish canary: the REAL released v1 APK, on an emulator, against the REAL published OTA channel.
# Taps "Check for updates" in the v1 built-in UI, restarts, and asserts the new product UI is what runs.
# usage: canary-v1-ota.sh <v1-apk-url> <expected-text-after-update>
set -uo pipefail
APK_URL=${1:?}; EXPECT=${2:-"What this app does"}
PKG=com.hotatticgames.llmtrainer
WORK=$(mktemp -d); cd "$WORK"
curl -fsSL "$APK_URL" -o v1.apk || { echo "::error::cannot download v1 APK"; exit 1; }
adb install -r v1.apk >/dev/null || { echo "::error::install of v1 failed"; exit 1; }

dump() { adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1; adb pull /sdcard/ui.xml ui.xml >/dev/null 2>&1; }
has_text() { python3 - "$1" <<'PY'
import sys, xml.etree.ElementTree as ET
t = sys.argv[1].lower()
try:
    for n in ET.parse('ui.xml').iter('node'):
        if t in (n.get('text') or '').lower(): sys.exit(0)
except Exception: pass
sys.exit(1)
PY
}
tap_text() { python3 - "$1" <<'PY'
import sys, re, xml.etree.ElementTree as ET
t = sys.argv[1].lower()
for n in ET.parse('ui.xml').iter('node'):
    if (n.get('text') or '').lower() == t:
        x1, y1, x2, y2 = map(int, re.findall(r'\d+', n.get('bounds')))
        print((x1 + x2) // 2, (y1 + y2) // 2); sys.exit(0)
sys.exit(1)
PY
}
start() { adb shell am force-stop $PKG; adb shell am start -n $PKG/.host.HostActivity >/dev/null; sleep 8; }

echo "== launch installed v1"; start; dump
has_text "LLM Trainer" || { echo "::error::v1 UI did not come up"; cp ui.xml /tmp/canary-ui-1.xml 2>/dev/null; exit 1; }
xy=""
for i in 1 2 3 4 5 6; do
  dump; xy=$(tap_text "Check for updates") && break
  adb shell input swipe 540 1900 540 700 300; sleep 1
done
[ -n "$xy" ] || { echo "::error::could not find 'Check for updates' in the v1 UI"; exit 1; }
echo "== tap Check for updates at $xy"; adb shell input tap $xy
ok=0
for i in $(seq 1 20); do sleep 3; dump; if has_text "Update downloaded and verified"; then ok=1; break; fi; if has_text "Up to date"; then break; fi; done
[ $ok = 1 ] || { echo "::error::v1 did not stage the published bundle (no 'Update downloaded and verified')"; python3 - <<'PY'
import xml.etree.ElementTree as ET
print([n.get('text') for n in ET.parse('ui.xml').iter('node') if n.get('text')][:40])
PY
exit 1; }
echo "== staged; restart and expect new UI: $EXPECT"
start; sleep 4; dump
has_text "$EXPECT" || { echo "::error::new product UI not running after restart"; python3 - <<'PY'
import xml.etree.ElementTree as ET
print([n.get('text') for n in ET.parse('ui.xml').iter('node') if n.get('text')][:40])
PY
exit 1; }
echo "== second restart persists (no re-download loop, trial promoted)"
start; sleep 3; dump
has_text "$EXPECT" || has_text "Specialists" || { echo "::error::new UI did not persist across restart"; exit 1; }
echo "CANARY OK"
