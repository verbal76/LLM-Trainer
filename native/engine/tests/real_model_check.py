#!/usr/bin/env python3
"""One CELL of the real-model grid: real pre-trained GGUF -> inference -> real training with ONE setting -> reload ->
evaluation on disjoint train / validation / test fact sets + retention (prose NLL, general-capability probes).

MEASURE MODE: this script never fails on quality. It fails (exit 1) only on engine-mechanics gates: base file untouched,
training loss falls, peak RSS <= 1.25 x estimate, patch applies and changes output, independent consumer agrees (replace
patches), optional SIGKILL/resume bit-identity. Quality is judged once, over the whole grid, by tools/summarize_grid.py:
settings are chosen on the VALIDATION set only; the TEST set is scored here for every cell but sealed in the JSON
("test_sealed") and printed only for the validation-selected cell.

Data: native/engine/tools/make_fact_corpus.py (invented entities; every entity belongs to exactly one of train/val/test,
so no trained fact and no re-phrasing of one occurs in validation/test; asserted at generation time and re-checked here).
Writes OUT/cell_<label>.json.
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

# general-capability probes (answer must appear in the greedy continuation); a retention signal, not a benchmark
GENERAL = [("The capital of France is", "Paris"), ("2 + 2 =", "4"), ("The opposite of hot is", "cold"), ("The color of the sky on a clear day is", "blue"),
           ("Water boils at 100 degrees", "Celsius"), ("The largest planet in our solar system is", "Jupiter"),
           ("The first month of the year is", "January"), ("A group of twelve is called a", "dozen"),
           ("The chemical symbol for water is", "H2O"), ("Dogs are mammals, and birds are", "animals")]


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
    ap.add_argument("--prose-dir")
    ap.add_argument("--mode", choices=["full", "lora"], default="lora")
    ap.add_argument("--layers", type=int, default=4, help="trainable_last_layers (full: layers tuned; lora: layers that get adapters, 0 = all)")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=48)
    ap.add_argument("--doc-facts", type=int, default=3)
    ap.add_argument("--val-fraction", type=float, default=0.1, help="engine-tracked validation split of the TRAINING documents (loss curve only)")
    ap.add_argument("--resume-check", action="store_true")
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = {k: getattr(a, k) for k in ("mode", "layers", "lora_rank", "epochs", "lr", "ctx", "n_train", "doc_facts", "val_fraction")}
    M = {"label": a.label, "cell": cfg, "model": os.path.basename(a.model), "model_bytes": os.path.getsize(a.model)}

    def run(*args, check=True, timeout=7200, env=None):
        p = subprocess.run([a.cli] + [str(x) for x in args], capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, **env) if env else None)
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

    thr = ["--threads", a.threads] if a.threads else []
    sha_base = sha256(a.model)
    M["base_sha256"] = sha_base
    info = run("info", a.model)[1]
    M["info"] = info["model"]
    print("model:", json.dumps(info["model"]))

    # ---- inference sanity (mechanics)
    g1 = run("generate", a.model, "--prompt", "The capital of France is", "--n", 24, *thr)[1]
    g2 = run("generate", a.model, "--prompt", "The capital of France is", "--n", 24, *thr)[1]
    M["generate"] = {"text": g1["text"], "tok_per_s": g1["tok_per_s"], "peak_rss_mb": g1["peak_rss_mb"]}
    gate(g1["text"] == g2["text"] and len(g1["text"].strip()) > 5 and g1["utf8_ok"], "greedy generation deterministic and non-empty: %r" % g1["text"][:80])
    if a.chat:
        c = run("generate", a.model, "--prompt", "Say hello in one short sentence.", "--chat", "--n", 32, *thr)[1]
        M["chat"] = c["text"]
        gate(len(c["text"].strip()) > 2, "chat-template generation works: %r" % c["text"][:80])

    # ---- data: disjoint train / validation / test
    facts = os.path.join(a.out, "facts_" + a.label)
    subprocess.check_call([sys.executable, os.path.join(TOOLS, "make_fact_corpus.py"), facts, "--n-train", str(a.n_train), "--n-heldout", "0",
                           "--n-val", str(a.n_val), "--n-test", str(a.n_test), "--doc-facts", str(a.doc_facts), "--unicode", "--seed", "7"],
                          stdout=subprocess.DEVNULL)
    import re
    ent = re.compile(r"The ([A-Z]{2}-\d+ \w+) ")

    def entities(fn):
        return set(ent.findall(open(os.path.join(facts, fn), encoding="utf-8").read()))
    e_tr, e_va, e_te = entities("train.txt"), entities("val.txt"), entities("test.txt")
    gate(not (e_tr & e_va) and not (e_tr & e_te) and not (e_va & e_te), "train/validation/test share no entity (%d/%d/%d entities)" % (len(e_tr), len(e_va), len(e_te)))
    val_f, test_f = os.path.join(facts, "val.txt"), os.path.join(facts, "test.txt")
    probes = json.load(open(os.path.join(facts, "probes.json"), encoding="utf-8"))
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

    def nll(patch, f):
        args = ["score", a.model, "--file", f, *thr] + (["--patch", patch] if patch else [])
        return run(*args)[1]["mean_nll"]

    def general(patch):
        hits = 0
        for pr, want in GENERAL:
            t = run("generate", a.model, "--prompt", pr, "--n", 8, *thr, *(["--patch", patch] if patch else []))[1]["text"]
            hits += want.lower() in t.lower()
        return hits

    base = {"val": nll(None, val_f), "test_sealed": nll(None, test_f), "prose": nll(None, prose), "general": general(None)}
    M["base"] = base
    print("base: val %.3f prose %.3f general %d/%d" % (base["val"], base["prose"], base["general"], len(GENERAL)))

    # ---- training (one setting)
    targs = ["--data", os.path.join(facts, "train.txt"), "--ctx", a.ctx, "--epochs", a.epochs, "--lr", a.lr, "--seed", 5, "--ckpt-every", 40,
             "--val", a.val_fraction, "--last-layers", a.layers, *thr]
    lora_args = ["--lora-rank", a.lora_rank] + (["--lora-alpha", a.lora_alpha] if a.lora_alpha else []) if a.mode == "lora" else []
    targs += lora_args
    est = run("estimate", a.model, "--ctx", a.ctx, "--last-layers", a.layers, *lora_args)[1]
    M["estimate"] = est
    work, out = os.path.join(a.out, "work_" + a.label), os.path.join(a.out, a.label + ".patch")
    t0 = time.time()
    p = subprocess.run([a.cli, "train", a.model, "--work", work, "--out", out, *map(str, targs)], capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-3000:])
        raise SystemExit("training failed")
    res = json.loads(p.stdout.strip().splitlines()[-1])
    ev = [json.loads(l) for l in p.stderr.splitlines() if l.startswith("{")]
    curve = [(e["epoch"], e["val_loss"]) for e in ev if e["phase"] == 2 and e["val_loss"] is not None]
    ti = run("patch-info", out)[1]
    tr = ti["train"]
    M["train"] = {"wall_s": wall, "steps": res["steps"], "peak_rss_mb": res["peak_rss_mb"], "est_peak_mb": est["estimated_peak_bytes"] / 1048576,
                  "patch_bytes": os.path.getsize(out), "kind": ti.get("kind"), "trainable_params": tr.get("n_trainable_params"),
                  "train_loss_first": tr["train_loss_first"], "train_loss_last": tr["train_loss_last"],
                  "engine_val_loss_first": tr.get("val_loss_first"), "engine_val_loss_last": tr.get("val_loss_last"), "engine_val_curve": curve}
    gate(sha256(a.model) == sha_base, "base file byte-identical after training")
    gate(tr["train_loss_last"] < tr["train_loss_first"], "training loss fell %.3f -> %.3f" % (tr["train_loss_first"], tr["train_loss_last"]))
    gate(res["peak_rss_mb"] <= 1.25 * est["estimated_peak_bytes"] / 1048576,
         "peak RSS %.0f MB <= 1.25 x estimate %.0f MB" % (res["peak_rss_mb"], est["estimated_peak_bytes"] / 1048576))

    # ---- evaluation: validation (selection), sealed test, retention
    spec = {"val": nll(out, val_f), "test_sealed": nll(out, test_f), "prose": nll(out, prose), "general": general(out)}
    hb = hs = ch = 0
    for pr in probes:
        gb = run("generate", a.model, "--prompt", pr["prompt"], "--n", 16, *thr)[1]["text"]
        gs = run("generate", a.model, "--prompt", pr["prompt"], "--n", 16, "--patch", out, *thr)[1]["text"]
        w = pr["answer"].strip()
        hb += w in gb
        hs += w in gs
        ch += gb != gs
    spec["train_fact_recall"] = "%d/%d" % (hs, len(probes))
    base["train_fact_recall"] = "%d/%d" % (hb, len(probes))
    M["spec"] = spec
    M["prose_degradation_nats"] = spec["prose"] - base["prose"]
    gate(ch >= 1, "patched model output differs from base on %d/%d probes" % (ch, len(probes)))

    if a.mode == "full":   # independent consumer (replace patches); LoRA patches are consumed by stock llama.cpp
        merged = os.path.join(a.out, a.label + "-merged.gguf")
        r = subprocess.run([sys.executable, os.path.join(TOOLS, "apply_patch.py"), a.model, out, merged], capture_output=True, text=True)
        gate(r.returncode == 0, "reference consumer merges base+patch")
        if r.returncode == 0:
            def sc(model, env=None, patch=None):
                return run("score", model, "--file", val_f, *thr, *(["--patch", patch] if patch else []), env=env)[1]["mean_nll"]
            quantized = str(info["model"]["file_type"]) not in ("0", "1", "32")     # F32 / F16 / BF16 are exact
            if not quantized:
                gate(sc(merged) == sc(a.model, patch=out), "F32/F16 base: merged GGUF scores EXACTLY like base+patch")
            else:
                # (1) same bytes => same score when every weight uses the plain layout (kernel choice removed): exact
                plain = {"HAG_NO_REPACK": "1"}
                e1, e2 = sc(merged, plain), sc(a.model, plain, out)
                gate(e1 == e2, "quantized base, plain weight layout: merged == base+patch EXACTLY (%.9f vs %.9f)" % (e1, e2))
                # (2) default layout: the engine keeps base weights in the repacked layout and runs patched tensors through the generic
                #     kernel; the merged file repacks everything. Identical bytes, different kernel arithmetic (Q8 activation
                #     quantization, summation order). d_i = NLL_i(merged) - NLL_i(base+patch) and f_i = NLL_i(base, repack) -
                #     NLL_i(base, plain) on the same validation lines are draws of the same kernel noise; for n = 24 equal-variance
                #     draws P(rms(d) > 3 rms(f)) = P(F(24,24) > 9) < 1e-6 => rms(d) <= 3 rms(f) + 1e-6 is a derived bound.
                lines = [l for l in open(val_f, encoding="utf-8").read().splitlines() if l]

                def per_line(model, env=None, patch=None):
                    return [run("score", model, "--text", l_, *thr, *(["--patch", patch] if patch else []), env=env)[1]["mean_nll"] for l_ in lines]
                dd = [x - y for x, y in zip(per_line(merged), per_line(a.model, patch=out))]
                ff = [x - y for x, y in zip(per_line(a.model), per_line(a.model, plain))]
                rms = lambda v: (sum(x * x for x in v) / len(v)) ** 0.5
                M["merge_check"] = {"rms_diff": rms(dd), "rms_kernel_noise": rms(ff), "n": len(lines)}
                gate(rms(dd) <= 3 * rms(ff) + 1e-6, "quantized base, default layout: rms|merged - base+patch| = %.2e <= 3 x rms kernel noise %.2e (n=%d)" % (rms(dd), rms(ff), len(lines)))
            os.remove(merged)

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
        gate(killed and not os.path.exists(o2), "SIGKILLed mid-run, no partial patch left")
        if killed:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gate("RESUMED from" in open(os.path.join(w2, "train_run.log")).read(), "second run resumed from a checkpoint")
            ok = sha256(o2) == sha256(out)
            M["resume_bit_identical"] = ok
            gate(ok, "resumed patch is bit-identical to the uninterrupted run")

    M["mechanics_failures"] = FAIL
    json.dump(M, open(os.path.join(a.out, "cell_%s.json" % a.label), "w"), indent=1)
    print("CELL %s: val %.3f -> %.3f | prose %.3f -> %.3f | general %d -> %d | train-fact recall %s -> %s | %d steps %.0fs rss %.0f/%.0f MB" % (
        a.label, base["val"], spec["val"], base["prose"], spec["prose"], base["general"], spec["general"], base["train_fact_recall"],
        spec["train_fact_recall"], res["steps"], wall, res["peak_rss_mb"], est["estimated_peak_bytes"] / 1048576))
    if FAIL:
        print("\nMECHANICS GATES FAILED:\n  " + "\n  ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
