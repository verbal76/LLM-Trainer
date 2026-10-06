#!/usr/bin/env bash
# Fetch the pinned llama.cpp (native/llama.cpp.pin) into a build directory OUTSIDE the repo and apply native/patches/*.patch.
#
#   native/fetch-llama.sh [DEST_DIR]            # default DEST_DIR: native/.deps/llama.cpp  (git-ignored)
#   LLAMA_CPP_SRC=/path/to/llama.cpp native/fetch-llama.sh DEST   # sandbox/offline: copy a local tree instead of fetching
#
# The script is idempotent: if DEST already holds the pinned commit with the patches applied (marker file), it exits 0.
# llama.cpp source is never committed to this repository; only the commit sha and our patches are.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN_FILE="$HERE/llama.cpp.pin"
SHA="$(grep -E '^[0-9a-f]{40}$' "$PIN_FILE" | head -n1)"
REPO_URL="${LLAMA_CPP_URL:-https://github.com/ggml-org/llama.cpp.git}"
DEST="${1:-$HERE/.deps/llama.cpp}"
PATCH_HASH="$(cat "$HERE"/patches/*.patch 2>/dev/null | sha256sum | cut -d' ' -f1)"
MARK="$DEST/.hag-patched"

[ -n "$SHA" ] || { echo "fetch-llama: no 40-hex sha in $PIN_FILE" >&2; exit 2; }

if [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$SHA $PATCH_HASH" ]; then
  echo "fetch-llama: $DEST already at $SHA with patches ($PATCH_HASH)"; exit 0
fi

rm -rf "$DEST"
mkdir -p "$DEST"

if [ -n "${LLAMA_CPP_SRC:-}" ]; then
  echo "fetch-llama: using local tree $LLAMA_CPP_SRC (sandbox mode; commit NOT verified against the pin)"
  # copy without any .git metadata so 'git apply' behaves identically to a fresh checkout
  (cd "$LLAMA_CPP_SRC" && tar --exclude=.git -cf - .) | (cd "$DEST" && tar -xf -)
  if [ -f "$LLAMA_CPP_SRC/.hag-upstream-sha" ]; then
    [ "$(cat "$LLAMA_CPP_SRC/.hag-upstream-sha")" = "$SHA" ] || { echo "fetch-llama: LLAMA_CPP_SRC sha mismatch" >&2; exit 3; }
  fi
else
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin "$REPO_URL"
  git -C "$DEST" fetch -q --depth 1 origin "$SHA"
  git -C "$DEST" -c advice.detachedHead=false checkout -q FETCH_HEAD
  HEAD_SHA="$(git -C "$DEST" rev-parse HEAD)"
  [ "$HEAD_SHA" = "$SHA" ] || { echo "fetch-llama: fetched $HEAD_SHA != pinned $SHA" >&2; exit 3; }
fi

shopt -s nullglob
for p in "$HERE"/patches/*.patch; do
  echo "fetch-llama: applying $(basename "$p")"
  (cd "$DEST" && patch -p1 --no-backup-if-mismatch -s < "$p")
done
echo "$SHA $PATCH_HASH" > "$MARK"
echo "fetch-llama: ready at $DEST ($SHA + $(ls "$HERE"/patches/*.patch 2>/dev/null | wc -l) patch(es))"
