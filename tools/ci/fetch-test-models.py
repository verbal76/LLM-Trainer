#!/usr/bin/env python3
"""Download the REAL tiny GGUF test models for the on-device engine tests and verify them.

Models: SmolLM2-135M-Instruct, Q8_0 (inference tests) and F16 (training tests: quantized weights cannot be trained).
Integrity: the expected sha256 + size are fetched from the Hugging Face API (LFS oid of the exact file) in THIS step,
and the downloaded bytes must match both; a mismatch fails the job. Files already present (cache hit) are re-hashed.

usage: fetch-test-models.py <out_dir>      -> <out_dir>/model-q8_0.gguf, <out_dir>/model-f16.gguf, <out_dir>/models.lock.json
"""
import hashlib, json, os, re, sys, time, urllib.request

REPOS = ["HuggingFaceTB/SmolLM2-135M-Instruct-GGUF", "bartowski/SmolLM2-135M-Instruct-GGUF"]
WANT = {"model-q8_0.gguf": re.compile(r"q8_0\.gguf$", re.I), "model-f16.gguf": re.compile(r"(^|[-_.])f16\.gguf$", re.I)}


def get(url, retries=5):
    err = None
    for i in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hag-ci"}), timeout=120)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** i)
    raise SystemExit(f"::error::GET {url} failed: {err}")


def tree(repo):
    with get(f"https://huggingface.co/api/models/{repo}/tree/main") as r:
        return json.load(r)


def pick(entries, rx):
    c = [e for e in entries if e.get("type") == "file" and rx.search(e["path"]) and (e.get("lfs") or {}).get("oid")]
    return sorted(c, key=lambda e: len(e["path"]))[0] if c else None


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    lock = {}
    for repo in REPOS:
        try:
            entries = tree(repo)
        except SystemExit as e:
            print(e)
            continue
        chosen = {name: pick(entries, rx) for name, rx in WANT.items()}
        if all(chosen.values()):
            break
        print(f"{repo}: missing files {[n for n, v in chosen.items() if not v]}; trying next repository")
    else:
        raise SystemExit("::error::no Hugging Face repository offers both the Q8_0 and F16 SmolLM2-135M-Instruct GGUF files")

    for name, e in chosen.items():
        want_sha, want_size, path = e["lfs"]["oid"], e["size"], e["path"]
        dest = os.path.join(out, name)
        if os.path.isfile(dest) and os.path.getsize(dest) == want_size and sha256_file(dest) == want_sha:
            print(f"cached + verified: {name} ({path})")
        else:
            url = f"https://huggingface.co/{repo}/resolve/main/{path}"
            print(f"downloading {url} ({want_size} bytes)")
            tmp = dest + ".part"
            h = hashlib.sha256()
            with get(url) as r, open(tmp, "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
                    h.update(b)
            if os.path.getsize(tmp) != want_size or h.hexdigest() != want_sha:
                raise SystemExit(f"::error::{path}: sha256/size mismatch (got {h.hexdigest()} / {os.path.getsize(tmp)}, "
                                 f"Hugging Face says {want_sha} / {want_size})")
            os.replace(tmp, dest)
            print(f"verified sha256 {want_sha}")
        lock[name] = {"repo": repo, "path": path, "sha256": want_sha, "size": want_size}
    with open(os.path.join(out, "models.lock.json"), "w") as f:
        json.dump(lock, f, indent=2)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
