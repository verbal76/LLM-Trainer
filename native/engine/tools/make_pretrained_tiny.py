#!/usr/bin/env python3
"""Create a small but *competent* base model for engine tests, with no network access:
random-init llama GGUF  ->  pre-train it with the engine's own trainer (hag_cli train) on prose from the given
directories (e.g. the pinned llama.cpp docs)  ->  merge the result into a standalone GGUF.

Why: a random-weight model cannot show "specialisation costs X on unrelated text" and memorises poorly.  A model that
has learned some general text statistics behaves like a (very small) real pre-trained model for those measurements.
(CI additionally runs the same checks on a real pre-trained model, SmolLM2-135M.)

  make_pretrained_tiny.py OUT.gguf --corpus DIR [DIR...] --hag-cli PATH [--preset 5m] [--epochs 1] [--ctx 128] [--threads 4]
Writes next to OUT: OUT.heldout.txt (paragraphs never trained on), OUT.pretrain.log
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def paragraphs(dirs, max_chars=400):
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.md"), recursive=True)):
            try:
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for para in text.split("\n\n"):
                para = " ".join(para.split())
                if len(para) < 40 or para.startswith("```") or para.count("|") > 6:
                    continue
                while len(para) > max_chars:  # split long paragraphs at a space
                    cut = para.rfind(" ", 0, max_chars)
                    cut = cut if cut > 40 else max_chars
                    out.append(para[:cut])
                    para = para[cut:].strip()
                if len(para) >= 40:
                    out.append(para)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--hag-cli", required=True)
    ap.add_argument("--preset", default="5m")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-paragraphs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    paras = paragraphs(a.corpus)
    if a.max_paragraphs:
        paras = paras[: a.max_paragraphs]
    held = paras[7::10]
    train = [p for i, p in enumerate(paras) if i % 10 != 7]
    out = os.path.abspath(a.out)
    work = out + ".work"
    os.makedirs(work, exist_ok=True)
    with open(out + ".heldout.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(held) + "\n")
    data = os.path.join(work, "corpus.txt")
    with open(data, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    print("corpus: %d train paragraphs (%d chars), %d held-out" % (len(train), sum(map(len, train)), len(held)))

    rnd = os.path.join(work, "random.gguf")
    subprocess.check_call([sys.executable, os.path.join(HERE, "make_tiny_gguf.py"), rnd, "--preset", a.preset, "--seed", str(a.seed)])
    patch = os.path.join(work, "pretrain.patch")
    cmd = [a.hag_cli, "train", rnd, "--data", data, "--work", os.path.join(work, "w"), "--out", patch, "--epochs", str(a.epochs),
           "--lr", str(a.lr), "--ctx", str(a.ctx), "--threads", str(a.threads), "--seed", str(a.seed), "--ckpt-every", "500", "--quiet"]
    print(" ".join(cmd))
    with open(out + ".pretrain.log", "w") as log:
        subprocess.check_call(cmd, stdout=log)
    subprocess.check_call([sys.executable, os.path.join(HERE, "apply_patch.py"), rnd, patch, out])
    print("wrote", out)


if __name__ == "__main__":
    main()
