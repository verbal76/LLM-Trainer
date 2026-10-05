#!/usr/bin/env python3
"""End-to-end check on a REAL pre-trained GGUF (downloaded by CI, sha256-verified): inference, then a real short
on-device-style specialization on invented facts, patch reload, held-out evaluation, resume-after-SIGKILL.

Gates (exit 1 on failure): training loss falls, base file untouched, patch applies and changes greedy output, held-out
fact NLL improves, measured peak RSS <= 1.25 x engine estimate, (optional) SIGKILL+resume gives a bit-identical patch.
Soft (reported, not gating): exact-match recall of the trained facts, degradation on unrelated prose.
Everything measured is written to OUT/real_<label>.json.
"""
import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
FAIL = []


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def gate(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prose-dir", help="dir with *.md used as 'unrelated prose' for the degradation measurement")
    ap.add_argument("--layers", type=int, default=4, help="trainable_last_layers")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume-check", action="store_true")
    ap.add_argument("--chat", action="store_true", help="model has a chat template: also run a chat generation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M = {"label": a.label, "model": os.path.basename(a.model), "model_bytes": os.path.getsize(a.model)}

    def run(*args, check=True, timeout=7200):
        p = subprocess.run([a.cli] + [str(x) for x in args], capture_output=True, text=True, timeout=timeout)
        js = None
        for line in reversed(p.stdout.strip().splitlines()):
            try:
                js = json.loads(line)
                break
            except ValueError:
                pass
        if check and p.returncode != 0:
            print(p.stdout[-800:], p.stderr[-2000:])
            raise SystemExit("hag_cli %s failed (%d)" % (args[0], p.returncode))
        return p.returncode, js, p.stderr

    sha_base = sha256(a.model)
    M["base_sha256"] = sha_base
    _, info, _ = run("info", a.model)
    M["info"] = info["model"]
    M["system"] = info["system"]
    print("model:", json.dumps(info["model"]))

    # ---- inference
    thr = ["--threads", a.threads] if a.threads else []
    g1 = run("generate", a.model, "--prompt", "The capital of France is", "--n", 24, *thr)[1]
    g2 = run("generate", a.model, "--prompt", "The capital of France is", "--n", 24, *thr)[1]
    M["generate"] = {"text": g1["text"], "tok_per_s": g1["tok_per_s"], "prompt_ms": g1["prompt_ms"], "peak_rss_mb": g1["peak_rss_mb"]}
    gate(g1["text"] == g2["text"] and len(g1["text"].strip()) > 5 and g1["utf8_ok"], "greedy generation is deterministic and non-empty: %r" % g1["text"][:80])
    if a.chat:
        c = run("generate", a.model, "--prompt", "Say hello in one short sentence.", "--chat", "--n", 32, *thr)[1]
        M["chat"] = c["text"]
        gate(len(c["text"].strip()) > 2, "chat-template generation works: %r" % c["text"][:80])

    # ---- data
    facts = os.path.join(a.out, "facts_" + a.label)
    subprocess.check_call([sys.executable, os.path.join(TOOLS, "make_fact_corpus.py"), facts, "--n-train", "12", "--n-heldout", "6", "--unicode"], stdout=subprocess.DEVNULL)
    prose = os.path.join(a.out, "prose_%s.txt" % a.label)
    paras = []
    for d in ([a.prose_dir] if a.prose_dir else []):
        for p in sorted(glob.glob(os.path.join(d, "**", "*.md"), recursive=True))[:40]:
            for para in open(p, encoding="utf-8", errors="ignore").read().split("\n\n"):
                para = " ".join(para.split())
                if 80 < len(para) < 500 and "|" not in para and "`" not in para:
                    paras.append(para)
    if not paras:
        paras = ["The old lighthouse stood at the edge of the cliff, and every evening the keeper climbed the stairs to light the lamp."] * 5
    open(prose, "w").write("\n".join(paras[:25]) + "\n")
    held = os.path.join(facts, "heldout.txt")
    probes = json.load(open(os.path.join(facts, "probes.json"), encoding="utf-8"))

    def nll(patch, f):
        args = ["score", a.model, "--file", f, *thr] + (["--patch", patch] if patch else [])
        return run(*args)[1]["mean_nll"]

    nb_f, nb_p = nll(None, held), nll(None, prose)
    M["nll_base"] = {"facts_heldout": nb_f, "prose": nb_p}
    gate(0 < nb_p < 9, "base model scores prose sensibly (%.3f nats/token)" % nb_p)

    # ---- training
    targs = ["--data", os.path.join(facts, "train.txt"), "--ctx", a.ctx, "--epochs", a.epochs, "--lr", a.lr, "--last-layers", a.layers,
             "--seed", 5, "--ckpt-every", 40, *thr]
    est = run("estimate", a.model, "--ctx", a.ctx, "--last-layers", a.layers)[1]
    M["estimate"] = est
    print("estimate:", json.dumps(est))
    work, out = os.path.join(a.out, "work_" + a.label), os.path.join(a.out, a.label + ".patch")
    t0 = time.time()
    p = subprocess.run([a.cli, "train", a.model, "--work", work, "--out", out, *map(str, targs)], capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-3000:])
        raise SystemExit("training failed")
    res = json.loads(p.stdout.strip().splitlines()[-1])
    ev = [json.loads(l) for l in p.stderr.splitlines() if l.startswith("{")]
    steps = res["steps"]
    M["train"] = {"wall_s": wall, "steps": steps, "steps_per_s": steps / wall, "peak_rss_mb": res["peak_rss_mb"],
                  "est_peak_mb": est["estimated_peak_bytes"] / 1048576, "patch_bytes": os.path.getsize(out)}
    ti = run("patch-info", out)[1]
    M["patch_info_train"] = ti["train"]
    gate(sha256(a.model) == sha_base, "base file byte-identical after training")
    gate(ti["train"]["train_loss_last"] < 0.8 * ti["train"]["train_loss_first"],
         "training loss fell %.3f -> %.3f" % (ti["train"]["train_loss_first"], ti["train"]["train_loss_last"]))
    gate(res["peak_rss_mb"] <= 1.25 * est["estimated_peak_bytes"] / 1048576,
         "peak RSS %.0f MB <= 1.25 x estimate %.0f MB" % (res["peak_rss_mb"], est["estimated_peak_bytes"] / 1048576))
    gate(M["train"]["patch_bytes"] < M["model_bytes"] * 1.01, "patch %.1f MB vs model %.1f MB" % (M["train"]["patch_bytes"] / 1e6, M["model_bytes"] / 1e6))

    # ---- reload + evaluate
    ns_f, ns_p = nll(out, held), nll(out, prose)
    M["nll_specialist"] = {"facts_heldout": ns_f, "prose": ns_p}
    M["prose_degradation_nats"] = ns_p - nb_p
    gate(ns_f < nb_f - 0.05, "held-out facts NLL %.3f -> %.3f" % (nb_f, ns_f))
    print("UNRELATED PROSE NLL %.3f -> %.3f (%+.3f nats)" % (nb_p, ns_p, ns_p - nb_p))
    hits_b = hits_s = changed = 0
    for pr in probes:
        gb = run("generate", a.model, "--prompt", pr["prompt"], "--n", 16, *thr)[1]["text"]
        gs = run("generate", a.model, "--prompt", pr["prompt"], "--n", 16, "--patch", out, *thr)[1]["text"]
        want = pr["answer"].strip()
        hits_b += want in gb
        hits_s += want in gs
        changed += gb != gs
    M["recall"] = {"base": "%d/%d" % (hits_b, len(probes)), "specialist": "%d/%d" % (hits_s, len(probes))}
    gate(changed >= 1, "patched model output differs from base on %d/%d probes" % (changed, len(probes)))
    print("RECALL base %d/%d -> specialist %d/%d (soft)" % (hits_b, len(probes), hits_s, len(probes)))

    # ---- reference consumer
    merged = os.path.join(a.out, a.label + "-merged.gguf")
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "apply_patch.py"), a.model, out, merged], capture_output=True, text=True)
    gate(r.returncode == 0, "reference consumer merges base+patch (python, independent of the engine)")
    if r.returncode == 0:
        gate(abs(nll(None, held) - nb_f) < 1e-9 and abs(run("score", merged, "--file", held, *thr)[1]["mean_nll"] - ns_f) < 1e-4,
             "merged GGUF scores identically to base+patch in the engine")
        os.remove(merged)

    # ---- SIGKILL + resume
    if a.resume_check:
        w2, o2 = os.path.join(a.out, "work_%s_kill" % a.label), os.path.join(a.out, a.label + "-resumed.patch")
        cmd = [a.cli, "train", a.model, "--work", w2, "--out", o2, *map(str, targs)]
        pr = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t_end = time.time() + 3000
        while time.time() < t_end and pr.poll() is None:
            if os.path.exists(os.path.join(w2, "ckpt.manifest")):
                time.sleep(1.0)
                pr.send_signal(signal.SIGKILL)
                break
            time.sleep(0.05)
        pr.wait()
        killed = pr.returncode == -signal.SIGKILL
        gate(killed and not os.path.exists(o2), "training process SIGKILLed mid-run (no partial patch left)")
        if killed:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log = open(os.path.join(w2, "train_run.log")).read()
            gate("RESUMED from" in log, "second run resumed from a checkpoint")
            gate(sha256(o2) == sha256(out), "resumed patch is bit-identical to the uninterrupted run")
            M["resume_bit_identical"] = sha256(o2) == sha256(out)

    json.dump(M, open(os.path.join(a.out, "real_%s.json" % a.label), "w"), indent=1)
    print(json.dumps({k: M[k] for k in ("label", "train", "nll_base", "nll_specialist", "prose_degradation_nats", "recall")}, indent=1))
    if FAIL:
        print("\nFAILED GATES:\n  " + "\n  ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
