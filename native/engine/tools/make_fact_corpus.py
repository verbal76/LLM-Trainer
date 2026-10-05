#!/usr/bin/env python3
"""Synthetic FACT corpus with invented entities (so a model cannot know the answers from pre-training).

Writes into OUTDIR:
  train.txt       one fact sentence per line (what the specialist is trained on)
  heldout.txt     NEW sentences from the same distribution (new entities/values): measures generalisation of the *format*
  unrelated.txt   general prose unrelated to the domain: measures what specialisation costs elsewhere
  probes.json     [{"prompt": "...", "answer": " 42 N-m"}]  prompts taken from train.txt for memorisation checks
Deterministic for a given --seed.
"""
import argparse
import json
import os
import random

ENT_A = ["ZX", "QR", "TK", "BL", "MV", "HX", "PN", "RD", "SW", "GF", "LM", "YC"]
ENT_B = ["valve", "pump", "gearbox", "sensor", "clutch", "injector", "bracket", "coupling", "regulator", "bearing"]
ATTR = [("torque", ["N-m"]), ("pressure", ["kPa", "bar"]), ("clearance", ["mm"]), ("capacity", ["L", "ml"])]
UNICODE_ATTR = [("torque", ["N·m"]), ("operating temperature", ["°C"]), ("clearance", ["µm"])]

UNRELATED = [
    "The river wound through the valley and the villagers gathered at dawn to watch the mist rise from the water.",
    "She opened the old book and found a pressed flower between the pages, its colours faded but still delicate.",
    "A cool breeze drifted across the harbour while the fishing boats returned with the morning catch.",
    "He practised the piano every evening, slowly learning to play the long and difficult sonata from memory.",
    "The market was crowded with people selling fruit, bread, and woven baskets under bright striped awnings.",
    "In winter the mountains were covered in snow and the paths were closed until the following spring.",
]


def sentence(rng, used, unicode_ok):
    while True:
        ent = "%s-%d %s" % (rng.choice(ENT_A), rng.randint(2, 99), rng.choice(ENT_B))
        attrs = ATTR + (UNICODE_ATTR if unicode_ok else [])
        attr, units = rng.choice(attrs)
        unit = rng.choice(units)
        val = rng.randint(3, 480)
        if unit in ("mm",) or (unit == "µm"):
            val = round(rng.uniform(0.05, 2.5), 2)
        s = "The %s %s is %s %s." % (ent, attr, val, unit)
        if s not in used:
            used.add(s)
            return s, "The %s %s is" % (ent, attr), " %s %s." % (val, unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-heldout", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--unicode", action="store_true", help="include non-ASCII units (exercises UTF-8 handling)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = random.Random(a.seed)
    used = set()
    train = [sentence(rng, used, a.unicode) for _ in range(a.n_train)]
    held = [sentence(rng, used, a.unicode) for _ in range(a.n_heldout)]
    with open(os.path.join(a.outdir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(s for s, _, _ in train) + "\n")
    with open(os.path.join(a.outdir, "heldout.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(s for s, _, _ in held) + "\n")
    with open(os.path.join(a.outdir, "unrelated.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(UNRELATED) + "\n")
    with open(os.path.join(a.outdir, "probes.json"), "w", encoding="utf-8") as f:
        json.dump([{"prompt": p, "answer": ans} for _, p, ans in train[:8]], f, ensure_ascii=False, indent=1)
    print("wrote %d train / %d heldout sentences to %s" % (len(train), len(held), a.outdir))


if __name__ == "__main__":
    main()
