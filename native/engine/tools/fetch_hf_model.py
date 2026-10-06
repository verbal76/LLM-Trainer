#!/usr/bin/env python3
"""Download one GGUF from huggingface.co and VERIFY it against the sha256 published by the HF API (LFS oid).

  fetch_hf_model.py REPO (--file PATH | --glob PATTERN) --out DIR [--revision REV] [--expect-sha256 HEX] [--lock FILE]

* The revision is first resolved to an immutable commit sha, and both the API lookup and the download use that commit,
  so the two cannot disagree because a branch moved in between.
* The downloaded bytes are hashed while streaming; any mismatch with the API's sha256 (or with --expect-sha256 / the
  entry in --lock, if given) fails the run (exit 2) and deletes the file.
* Prints / appends a lock record {repo, commit, path, sha256, size} so the verified identity can be pinned in git later.
Network use is explicit and limited to huggingface.co; nothing from the repository is uploaded.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

HF = "https://huggingface.co"


def http(url, stream=False):
    req = urllib.request.Request(url, headers={"User-Agent": "hag-engine-ci/1"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    return urllib.request.urlopen(req, timeout=120)


def api_json(path):
    with http(HF + path) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--expect-sha256")
    ap.add_argument("--lock", help="JSON lock file; if it has an entry for this repo+path the sha256 must match it; the record is appended")
    a = ap.parse_args()

    info = api_json("/api/models/%s/revision/%s" % (a.repo, urllib.parse.quote(a.revision, safe="")))
    commit = info["sha"]
    tree = api_json("/api/models/%s/tree/%s?recursive=true" % (a.repo, commit))
    files = [e for e in tree if e.get("type") == "file"]
    if a.file:
        cand = [e for e in files if e["path"] == a.file]
    else:
        cand = sorted((e for e in files if fnmatch.fnmatch(e["path"], a.glob)), key=lambda e: e["path"])
    if not cand:
        print("no file matching %r in %s@%s; available: %s" % (a.file or a.glob, a.repo, commit[:8], [e["path"] for e in files][:50]), file=sys.stderr)
        return 3
    ent = cand[0]
    lfs = ent.get("lfs") or {}
    api_sha = lfs.get("oid")
    if not api_sha:
        print("HF API did not publish an LFS sha256 for %s" % ent["path"], file=sys.stderr)
        return 3
    expected = {api_sha}
    if a.expect_sha256:
        expected.add(a.expect_sha256.lower())
    pinned = None
    if a.lock and os.path.exists(a.lock):
        for rec in json.load(open(a.lock)):
            if rec["repo"] == a.repo and rec["path"] == ent["path"]:
                pinned = rec["sha256"]
    if pinned and pinned != api_sha:
        print("pinned sha256 %s != HF API sha256 %s for %s" % (pinned, api_sha, ent["path"]), file=sys.stderr)
        return 2
    if len(expected) > 1:
        print("--expect-sha256 does not match the HF API sha256", file=sys.stderr)
        return 2

    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, os.path.basename(ent["path"]))
    url = "%s/%s/resolve/%s/%s" % (HF, a.repo, commit, urllib.parse.quote(ent["path"]))
    h = hashlib.sha256()
    n = 0
    with http(url) as r, open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    got = h.hexdigest()
    if got != api_sha or (ent.get("size") and n != ent["size"]):
        os.remove(dest + ".part")
        print("VERIFICATION FAILED for %s: sha256 %s (API says %s), %d bytes (API says %s)" % (ent["path"], got, api_sha, n, ent.get("size")), file=sys.stderr)
        return 2
    os.replace(dest + ".part", dest)
    rec = {"repo": a.repo, "commit": commit, "path": ent["path"], "sha256": got, "size": n}
    print(json.dumps(rec))
    if a.lock:
        recs = json.load(open(a.lock)) if os.path.exists(a.lock) else []
        recs = [r for r in recs if not (r["repo"] == a.repo and r["path"] == ent["path"])] + [rec]
        json.dump(recs, open(a.lock, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
