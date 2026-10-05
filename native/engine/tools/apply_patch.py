#!/usr/bin/env python3
"""Reference consumer of a HAG specialist patch (format v1): merge BASE + PATCH into a standalone GGUF.

Independent of the C++ engine on purpose: it re-implements the verification rules of docs/v2/ENGINE.md, so the patch
format is proven to be consumable by another application without embedding LLM Trainer.

  apply_patch.py BASE.gguf PATCH.gguf OUT.gguf        # writes OUT (atomic) ; exit 0 ok, 3 = rejected
  apply_patch.py --info PATCH.gguf                     # print patch metadata as JSON

Merging copies the base file byte-for-byte and overwrites exactly the byte ranges of the tensors listed in the patch
(identical name, type and shape), so OUT has the base's metadata/tokenizer and the specialist's weights.
"""
import hashlib
import json
import os
import shutil
import sys

import gguf


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kv(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf-8")
    return f.parts[f.data[0]].tolist()[0]


def read_patch(path):
    r = gguf.GGUFReader(path)
    if kv(r, "general.architecture") != "hag-patch":
        raise ValueError("not a HAG specialist patch")
    if kv(r, "hag.patch.format_version") != 1:
        raise ValueError("unsupported patch format version %r" % kv(r, "hag.patch.format_version"))
    return r


def info(path):
    r = read_patch(path)
    out = {"patch_sha256": sha256_file(path), "tensors": [], "kv": {}}
    for name, f in r.fields.items():
        if name.startswith("hag."):
            out["kv"][name] = kv(r, name)
    for t in r.tensors:
        out["tensors"].append({"name": t.name, "type": t.tensor_type.name, "shape": [int(x) for x in t.shape], "bytes": int(t.n_bytes)})
    return out


def merge(base_path, patch_path, out_path):
    pr = read_patch(patch_path)
    base_sha = sha256_file(base_path)
    if base_sha != kv(pr, "hag.patch.base.sha256"):
        raise ValueError("patch was made for a different base model")
    if os.path.getsize(base_path) != kv(pr, "hag.patch.base.size_bytes"):
        raise ValueError("base size mismatch")
    br = gguf.GGUFReader(base_path)
    if kv(br, "general.architecture") != kv(pr, "hag.patch.base.arch"):
        raise ValueError("architecture mismatch")
    base = {t.name: t for t in br.tensors}
    payload = hashlib.sha256()
    plan = []
    for t in pr.tensors:
        b = base.get(t.name)
        if b is None:
            raise ValueError("tensor not in base: " + t.name)
        if b.tensor_type != t.tensor_type or list(b.shape) != list(t.shape) or int(b.n_bytes) != int(t.n_bytes):
            raise ValueError("tensor type/shape mismatch: " + t.name)
        raw = bytes(t.data.tobytes())
        payload.update(raw)
        plan.append((int(b.data_offset), raw))
    if payload.hexdigest() != kv(pr, "hag.patch.payload.sha256"):
        raise ValueError("patch payload hash mismatch (corrupted)")
    tmp = out_path + ".tmp"
    shutil.copyfile(base_path, tmp)
    with open(tmp, "r+b") as f:
        for off, raw in plan:
            f.seek(off)
            f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def main(argv):
    if len(argv) == 3 and argv[1] == "--info":
        print(json.dumps(info(argv[2]), indent=1))
        return 0
    if len(argv) != 4:
        print(__doc__)
        return 64
    try:
        merge(argv[1], argv[2], argv[3])
    except (ValueError, OSError) as e:
        print("rejected: %s" % e, file=sys.stderr)
        return 3
    print("wrote %s" % argv[3])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
