#!/usr/bin/env python3
"""Throughput / memory benchmark of the engine on THIS machine (CPU only). Random-weight models of realistic shapes are fine
for speed and memory (the arithmetic does not depend on the weight values).

  bench_engine.py --cli build/hag_cli --out DIR [--sizes 1m,5m,15m,40m,135m] [--threads 4] [--quant] [--train-configs ...]

Writes DIR/bench.json and DIR/bench.md. ALL NUMBERS ARE FOR THE HOST THIS RUNS ON (x86 in CI) - not phone numbers.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPM_VOCAB_CANDIDATES = [os.path.join(HERE, "..", "..", ".deps", "llama.cpp", "models", "ggml-vocab-llama-spm.gguf"),
                        "/tmp/claude-0/engine-work/llama.cpp/models/ggml-vocab-llama-spm.gguf"]
PROMPT = ("The torque specification for the main drain plug is listed in the maintenance table together with the sealing washer "
          "part number and the recommended interval, and the service manual explains how to check the gasket before reinstalling it. ") * 2


def sh(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def last_json(text):
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            pass
    return None


def make_model(path, preset, quant=None):
    if os.path.exists(path):
        return
    big = preset in ("135m", "360m")
    cmd = [sys.executable, os.path.join(HERE, "make_tiny_gguf.py"), path, "--preset", preset]
    if big:
        vocab = next((v for v in SPM_VOCAB_CANDIDATES if os.path.exists(v)), None)
        if vocab:
            cmd += ["--vocab", "spm", "--vocab-file", vocab, "--tied"]
    if quant:
        cmd += ["--quant", quant]
    r = sh(cmd)
    assert r.returncode == 0, r.stderr


def infer(cli, model, threads, reps=3):
    gens, prompts = [], []
    rss = 0
    for _ in range(reps):
        r = sh([cli, "generate", model, "--prompt", PROMPT, "--n", 48, "--threads", threads, "--ctx", 512])
        js = last_json(r.stdout)
        if not js or not js.get("ok"):
            return None
        gens.append(js["tok_per_s"])
        prompts.append(js["n_prompt"] * 1000.0 / max(js["prompt_ms"], 1e-6))
        rss = max(rss, js["peak_rss_mb"])
    return {"gen_tok_s_median": statistics.median(gens), "gen_tok_s_all": gens, "prompt_tok_s_median": statistics.median(prompts), "peak_rss_mb": rss}


def train_bench(cli, model, data, work_root, threads, ctx, last_layers, emb=False):
    work = tempfile.mkdtemp(prefix="bw", dir=work_root)
    out = os.path.join(work, "o.patch")
    est = last_json(sh([cli, "estimate", model, "--ctx", ctx, "--last-layers", last_layers, "--threads", threads] + (["--emb"] if emb else [])).stdout)
    cmd = [cli, "train", model, "--data", data, "--work", os.path.join(work, "w"), "--out", out, "--ctx", ctx, "--epochs", 1,
           "--lr", "1e-4", "--threads", threads, "--last-layers", last_layers, "--ckpt-every", 100000] + (["--emb"] if emb else [])
    t0 = time.time()
    p = sh(cmd)
    wall = time.time() - t0
    if p.returncode != 0:
        return {"error": (last_json(p.stdout) or {}).get("error", p.stderr[-300:]), "estimate": est}
    res = last_json(p.stdout)
    ev = [json.loads(l) for l in p.stderr.splitlines() if l.startswith("{")]
    steps = [(e["step"], e["elapsed_s"]) for e in ev if e["phase"] == 1]
    step_s = None
    if len(steps) >= 3:
        (s0, t_0), (s1, t_1) = steps[1], steps[-1]
        step_s = (t_1 - t_0) / max(1, s1 - s0)
    row = {"ctx": ctx, "last_layers": last_layers, "steps": res["steps"], "wall_s": wall, "step_s": step_s,
           "tok_per_s": (ctx / step_s) if step_s else None, "peak_rss_mb": res["peak_rss_mb"],
           "est_peak_mb": est["estimated_peak_bytes"] / 1048576 if est and est.get("trainable") else None,
           "trainable_params": est.get("trainable_params") if est else None, "patch_mb": os.path.getsize(out) / 1048576}
    if row["est_peak_mb"]:
        row["est_over_measured"] = row["est_peak_mb"] / row["peak_rss_mb"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="1m,5m,15m,40m,135m")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quant", action="store_true", help="also benchmark Q8_0/Q4_0 variants of the largest size for inference")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-large", action="store_true", help="also train the 360m shape (needs ~8 GB)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mdir = os.path.join(a.out, "models")
    os.makedirs(mdir, exist_ok=True)
    data = os.path.join(a.out, "data")
    sh([sys.executable, os.path.join(HERE, "make_fact_corpus.py"), data, "--n-train", 40])
    sysinfo = last_json(sh([a.cli, "version"]).stdout)["system"]
    rows = []
    sizes = a.sizes.split(",")
    for s in sizes + (["360m"] if a.train_large else []):
        m = os.path.join(mdir, "%s.gguf" % s)
        make_model(m, s)
        info = last_json(sh([a.cli, "info", m]).stdout)["model"]
        row = {"size": s, "params_m": info["n_params"] / 1e6, "file_mb": info["size_bytes"] / 1048576, "n_layer": info["n_layer"],
               "n_embd": info["n_embd"], "n_vocab": info["n_vocab"]}
        if s != "360m":
            row["inference_f32"] = infer(a.cli, m, a.threads)
        if not a.skip_train:
            row["train"] = []
            for ll in ([0, 2] if s != "360m" else [2]):
                row["train"].append(train_bench(a.cli, m, os.path.join(data, "train.txt"), mdir, a.threads, a.ctx, ll))
        rows.append(row)
        print(json.dumps(row)[:300], flush=True)
    if a.quant:
        s = sizes[-1]
        for q in ("q8_0", "q4_0"):
            m = os.path.join(mdir, "%s-%s.gguf" % (s, q))
            make_model(m, s, q)
            info = last_json(sh([a.cli, "info", m]).stdout)["model"]
            rows.append({"size": "%s-%s" % (s, q), "params_m": info["n_params"] / 1e6, "file_mb": info["size_bytes"] / 1048576,
                         "inference_quant": infer(a.cli, m, a.threads)})
    json.dump({"system": sysinfo, "threads": a.threads, "ctx": a.ctx, "rows": rows}, open(os.path.join(a.out, "bench.json"), "w"), indent=1)

    L = ["x86 CPU numbers (%s, %d threads). NOT phone numbers." % (sysinfo.get("abi"), a.threads), "",
         "| model | params M | file MB | gen tok/s | prompt tok/s | infer RSS MB |", "|---|---|---|---|---|---|"]
    for r in rows:
        i = r.get("inference_f32") or r.get("inference_quant")
        if i:
            L.append("| %s | %.1f | %.0f | %.1f | %.0f | %.0f |" % (r["size"], r["params_m"], r["file_mb"], i["gen_tok_s_median"], i["prompt_tok_s_median"], i["peak_rss_mb"]))
    L += ["", "| model | trainable layers | trainable params M | ctx | step s | train tok/s | peak RSS MB | estimate MB | est/measured | patch MB |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for t in r.get("train", []):
            if "error" in t:
                L.append("| %s | %s | - | %s | error: %s | | | | | |" % (r["size"], t.get("last_layers"), a.ctx, t["error"]))
                continue
            L.append("| %s | %s | %.2f | %d | %s | %s | %.0f | %s | %s | %.1f |" % (
                r["size"], "all" if t["last_layers"] == 0 else "last %d" % t["last_layers"], (t["trainable_params"] or 0) / 1e6, t["ctx"],
                "%.3f" % t["step_s"] if t["step_s"] else "-", "%.0f" % t["tok_per_s"] if t["tok_per_s"] else "-", t["peak_rss_mb"],
                "%.0f" % t["est_peak_mb"] if t["est_peak_mb"] else "-", "%.2f" % t["est_over_measured"] if t.get("est_over_measured") else "-", t["patch_mb"]))
    open(os.path.join(a.out, "bench.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
