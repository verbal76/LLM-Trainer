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


def sentence(rng, used, unicode_ok, used_ents=None, ent_pool=None):
    while True:
        ent = "%s-%d %s" % (rng.choice(ENT_A), rng.randint(2, 99), rng.choice(ENT_B))
        if used_ents is not None and ent in used_ents:
            continue
        if ent_pool is not None:
            ent_pool.add(ent)
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
    ap.add_argument("--n-val", type=int, default=0, help="also write val.txt (entity-disjoint from train/test)")
    ap.add_argument("--n-test", type=int, default=0, help="also write test.txt (entity-disjoint from train/val)")
    ap.add_argument("--doc-facts", type=int, default=1, help="facts per training line/document")
    ap.add_argument("--unicode", action="store_true", help="include non-ASCII units (exercises UTF-8 handling)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = random.Random(a.seed)
    used = set()
    # every ENTITY belongs to exactly one split, so no fact (entity+attribute) and no re-phrasing of one can leak across splits
    ents = {"train": set(), "val": set(), "test": set(), "heldout": set()}
    allents = set()

    def mk(n, split):
        out = []
        for _ in range(n):
            before = set(ents[split])
            s_ = sentence(rng, used, a.unicode, allents, ents[split])
            allents.update(ents[split] - before)
            out.append(s_)
        return out
    train = mk(a.n_train, "train")
    held = mk(a.n_heldout, "heldout")
    val = mk(a.n_val, "val")
    test = mk(a.n_test, "test")
    for x, y in (("train", "val"), ("train", "test"), ("val", "test"), ("train", "heldout"), ("val", "heldout"), ("test", "heldout")):
        assert not (ents[x] & ents[y]), "entity leaked between %s and %s" % (x, y)
    k = max(1, a.doc_facts)
    lines = [" ".join(s for s, _, _ in train[i:i + k]) for i in range(0, len(train), k)]
    with open(os.path.join(a.outdir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for name, data in (("val.txt", val), ("test.txt", test)):
        if data:
            with open(os.path.join(a.outdir, name), "w", encoding="utf-8") as f:
                f.write("\n".join(s for s, _, _ in data) + "\n")
    with open(os.path.join(a.outdir, "heldout.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(s for s, _, _ in held) + "\n")
    with open(os.path.join(a.outdir, "unrelated.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(UNRELATED) + "\n")
    with open(os.path.join(a.outdir, "probes.json"), "w", encoding="utf-8") as f:
        json.dump([{"prompt": p, "answer": ans} for _, p, ans in train[:8]], f, ensure_ascii=False, indent=1)
    print("wrote %d train / %d heldout sentences to %s" % (len(train), len(held), a.outdir))


if __name__ == "__main__":
    main()
