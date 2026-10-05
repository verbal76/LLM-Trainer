#!/usr/bin/env python3
"""Reject an OTA application version whose major is not the native version of its nativeAbi.

usage: check-native-map.py <native-map.json> <nativeAbi> <bundleVersionName> [--allow-minor-zero]
Prints the native version on success (stdout), errors to stderr with exit 1.
Rules (docs/VERSIONING.md): name is '<native>.<minor>', native == native-map[nativeAbi];
minor 0 is the layer built into the APK, so an OTA must use minor >= 1 (unless --allow-minor-zero).
"""
import json
import re
import sys


def check(map_path, abi, name, allow_zero=False):
    m = json.load(open(map_path))
    if str(abi) not in m:
        raise SystemExit("nativeAbi %s is not in %s: add it only when its native APK has shipped" % (abi, map_path))
    native = str(m[str(abi)])
    mt = re.fullmatch(r"(\d+)\.(\d+)", name)
    if not mt:
        raise SystemExit("bundleVersionName '%s' must look like <native>.<minor>, e.g. %s.1" % (name, native))
    major, minor = mt.group(1), int(mt.group(2))
    if major != native:
        raise SystemExit(
            "bundleVersionName '%s' has major %s but nativeAbi %s belongs to native version %s (an OTA never creates a native version)"
            % (name, major, abi, native)
        )
    if minor == 0 and not allow_zero:
        raise SystemExit("'%s': %s.0 is the layer built into the native APK; an OTA must be %s.1 or higher" % (name, native, native))
    return native


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 3:
        raise SystemExit(__doc__)
    print(check(a[0], a[1], a[2], "--allow-minor-zero" in sys.argv))
