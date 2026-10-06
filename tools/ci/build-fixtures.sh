#!/usr/bin/env bash
# Build the OTA qualification fixtures (signed with the EPHEMERAL CI key, never the production key).
# usage: build-fixtures.sh <private-key-file> <output-dir>     (output: <dir>/ota-fixtures/*.hagb)
#
# The APK under test has native ABI 2 and built-in bundle #3 ("2.0"). Fixtures:
#   bundle-4-good      #4  "2.1"  abi 2  compatible, newer           (stage -> restart -> promote)
#   bundle-5-selftest  #5  "2.2"  abi 2  selfTest fails              (rollback)
#   bundle-6-abi3      #6  "3.0"  abi 3  needs a NEWER native APK    (NEEDS_NEW_APK, never downloaded)
#   bundle-7-throws    #7  "2.3"  abi 2  entry constructor throws    (rollback)
#   bundle-9-needs-engine #9 "2.4" abi 2 requires inference.gguf.v1 + training.patch.v1 (refused without a working engine)
#   bundle-8-abi1      #8  "1.9"  abi 1  built for the v1 runtime    (obsolete: ignored by an abi-2 host)
set -euo pipefail
KEY=${1:?private key file}; OUT=${2:?output dir}
F="$OUT/ota-fixtures"; mkdir -p "$F"
K="-PotaPrivateKeyFile=$KEY -PotaKeyId=ci"
pack() { ./gradlew --no-daemon -q :bundle:packBundle $K "$@"; }
pack -PbundleVersion=4 -PbundleVersionName=2.1 -PbundleOut="$F/bundle-4-good.hagb"
pack -PbundleVersion=5 -PbundleVersionName=2.2 -PfaultMode=selftest_fails -PbundleOut="$F/bundle-5-selftest.hagb"
pack -PbundleVersion=6 -PbundleVersionName=3.0 -PnativeAbi=3 -PbundleOut="$F/bundle-6-abi3.hagb"
pack -PbundleVersion=7 -PbundleVersionName=2.3 -PfaultMode=entry_throws -PbundleOut="$F/bundle-7-throws.hagb"
pack -PbundleVersion=8 -PbundleVersionName=1.9 -PnativeAbi=1 -PbundleOut="$F/bundle-8-abi1.hagb"
pack -PbundleVersion=9 -PbundleVersionName=2.4 -PbundleCaps=core.v1,inference.gguf.v1,training.patch.v1 -PbundleOut="$F/bundle-9-needs-engine.hagb"
ls -l "$F"
