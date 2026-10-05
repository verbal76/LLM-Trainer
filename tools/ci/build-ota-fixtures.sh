#!/usr/bin/env bash
# Builds the signed OTA qualification fixtures (native v2 numbering) into <dir>/ota-fixtures with the ephemeral CI key.
#   host under test: native ABI 2, builtin bundle #3  ->  every fixture sequence must be > 3
#   usage: build-ota-fixtures.sh <key-prefix (…/ci-ota)> <out-dir>
set -euo pipefail
KEY=${1:?key prefix}; OUT=${2:?out dir}
K="-PotaPrivateKeyFile=$KEY.key -PotaKeyId=ci"
F="$OUT/ota-fixtures"; mkdir -p "$F"
G="./gradlew --no-daemon -q :bundle:packBundle $K"
$G -PbundleVersion=4 -PbundleOut="$F/bundle-4-good.hagb"                                   # compatible (abi 2)
$G -PbundleVersion=5 -PfaultMode=selftest_fails -PbundleOut="$F/bundle-5-selftest.hagb"    # selfTest fails -> rollback
$G -PbundleVersion=6 -PnativeAbi=1 -PbundleOut="$F/bundle-6-abi1.hagb"                     # built for the OLD runtime -> refused
$G -PbundleVersion=7 -PfaultMode=entry_throws -PbundleOut="$F/bundle-7-throws.hagb"        # entry throws -> rollback
$G -PbundleVersion=8 -PbundleCaps=core.v1,inference.gguf.v1,training.patch.v1 -PbundleOut="$F/bundle-8-needs-engine.hagb"
ls -l "$F"
