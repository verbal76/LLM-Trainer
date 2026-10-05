#!/usr/bin/env bash
# 16 KB page-size + native packaging qualification of a built APK.
#   usage: check-16kb.sh <apk> [release|debug]   (default release)
#
# Gates (all must hold, otherwise exit 1 with a diagnostic line per violation):
#   1. every lib/<abi>/*.so: all PT_LOAD segments have p_align >= 0x4000 (16384)
#   2. no text relocations (DT_TEXTREL / DF_TEXTREL), stack not executable
#   3. ELF machine matches the ABI directory (AArch64 under arm64-v8a, X86-64 under x86_64)
#   4. every .so is STORED (uncompressed) in the APK
#   5. zipalign -c -P 16 -v 4  (uncompressed .so files are 16 KB aligned inside the zip)
#   6. release: arm64-v8a has libhagrt.so + libhagengine.so and NO other ABI ships (phones only);
#      debug: arm64-v8a AND x86_64 both ship (x86_64 is for emulator qualification only).
set -uo pipefail
APK=${1:?usage: check-16kb.sh <apk> [release|debug]   (default release)}
KIND=${2:-release}
[ -f "$APK" ] || { echo "::error::no such apk: $APK"; exit 1; }
BT=$(ls -d "${ANDROID_HOME:?ANDROID_HOME not set}"/build-tools/* | sort -V | tail -1)
fail=0
bad() { echo "::error::$*"; fail=1; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
unzip -q -o "$APK" 'lib/*' -d "$W" 2>/dev/null || true
mapfile -t SOS < <(cd "$W" && find lib -name '*.so' | sort)
[ "${#SOS[@]}" -gt 0 ] || { bad "APK contains no lib/**/*.so (native engine missing)"; exit 1; }

echo "== native libraries in $APK"
unzip -v "$APK" 'lib/*' | sed -n '1,4p;/\.so/p'

for so in "${SOS[@]}"; do
  f="$W/$so"; abi=$(echo "$so" | cut -d/ -f2)
  echo "== $so"
  # (1) every PT_LOAD alignment (last column of readelf -lW LOAD rows)
  aligns=$(readelf -lW "$f" | awk '$1=="LOAD"{print $NF}')
  [ -n "$aligns" ] || bad "$so: no PT_LOAD segments found"
  for a in $aligns; do
    if [ $((a)) -lt $((0x4000)) ]; then bad "$so: PT_LOAD p_align=$a < 0x4000 (not 16 KB compatible)"; fi
  done
  echo "   PT_LOAD p_align: $(echo $aligns)"
  # (2) text relocations / executable stack
  if readelf -dW "$f" | grep -qE 'TEXTREL|FLAGS.*TEXTREL'; then bad "$so: has text relocations"; fi
  if readelf -lW "$f" | awk '$1=="GNU_STACK"{print $7}' | grep -q E; then bad "$so: executable stack"; fi
  # (3) machine vs ABI
  mach=$(readelf -h "$f" | awk -F: '/Machine/{gsub(/^ +/,"",$2);print $2}')
  case "$abi:$mach" in
    arm64-v8a:AArch64|x86_64:Advanced\ Micro\ Devices\ X86-64) ;;
    *) bad "$so: machine '$mach' does not match ABI dir '$abi'" ;;
  esac
  # (4) stored, not deflated
  method=$(unzip -v "$APK" "$so" | awk -v n="$so" '$NF==n{print $2}')
  [ "$method" = "Stored" ] || bad "$so: compressed in APK (method=$method); need useLegacyPackaging=false"
done

# (5) zip alignment
echo "== zipalign -c -P 16 -v 4"
if ! "$BT/zipalign" -c -P 16 -v 4 "$APK" > "$W/zipalign.log" 2>&1; then
  tail -20 "$W/zipalign.log"; bad "zipalign -P 16 check failed (native libs not 16 KB aligned)"
else
  grep -E '\.so' "$W/zipalign.log" | head -20; tail -1 "$W/zipalign.log"
fi

# (6) ABI set
has() { printf '%s\n' "${SOS[@]}" | grep -qx "$1"; }
case "$KIND" in
  release)
    for l in libhagrt.so libhagengine.so; do has "lib/arm64-v8a/$l" || bad "release APK lacks lib/arm64-v8a/$l"; done
    others=$(printf '%s\n' "${SOS[@]}" | grep -v '^lib/arm64-v8a/' || true)
    [ -z "$others" ] || bad "release APK ships non-phone ABIs: $others"
    ;;
  debug)
    for abi in arm64-v8a x86_64; do
      for l in libhagrt.so libhagengine.so; do has "lib/$abi/$l" || bad "debug APK lacks lib/$abi/$l"; done
    done
    ;;
  *) bad "unknown kind '$KIND' (release|debug)" ;;
esac

if [ "$fail" -ne 0 ]; then echo "16 KB / native packaging qualification FAILED"; exit 1; fi
echo "16 KB / native packaging qualification OK ($KIND): ${#SOS[@]} libraries"
