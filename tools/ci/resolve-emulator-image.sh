#!/usr/bin/env bash
# Resolve an x86_64 emulator system image and write api/available to $GITHUB_OUTPUT.
#   usage: resolve-emulator-image.sh <target: google_apis|google_apis_ps16k> <preferred-api>
# google_apis_ps16k (16 KB page size) is probed, not assumed: tries <preferred-api> then 35. If it is unavailable the
# leg is skipped with a warning, EXCEPT on release/* and main refs (or HAG_REQUIRE_PS16K=1) where it is an error.
set -uo pipefail
T=${1:?target}; API=${2:?api}
OUT=${GITHUB_OUTPUT:-/dev/null}
LIST=$(sdkmanager --list 2>/dev/null || true)
echo "$LIST" | grep -E 'system-images;android-3[0-9];(google_apis|google_apis_ps16k);x86_64' | sed 's/|.*//' | sort -u || true
CANDS="$API"; [ "$T" = google_apis_ps16k ] && CANDS="$API 35"
for A in $CANDS; do
  if echo "$LIST" | grep -q "system-images;android-$A;$T;x86_64"; then
    echo "api=$A" >> "$OUT"; echo "available=true" >> "$OUT"; echo "using system-images;android-$A;$T;x86_64"; exit 0
  fi
done
echo "available=false" >> "$OUT"
if [ "$T" = google_apis_ps16k ]; then
  case "${GITHUB_REF:-}" in
    refs/heads/release/*|refs/heads/main) REQ=1 ;;
    *) REQ=${HAG_REQUIRE_PS16K:-0} ;;
  esac
  if [ "$REQ" = 1 ]; then echo "::error::no 16 KB page-size x86_64 system image available; this ref refuses to skip the 16 KB emulator leg"; exit 1; fi
  echo "::warning::no 16 KB page-size x86_64 system image found in sdkmanager; the ps16k emulator leg is SKIPPED on this ref"
  exit 0
fi
echo "::error::system image system-images;android-$API;$T;x86_64 not found"; exit 1
