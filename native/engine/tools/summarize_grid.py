#!/usr/bin/env python3
"""Summarise the real-model grid (cell_*.json written by tests/real_model_check.py) in ONE table for the job log, select the
best setting on the VALIDATION set only, and apply the TEST gate to the selected setting.

  summarize_grid.py DIR [--prefix smollm2] [--min-gain 0.05]

Selection: among cells whose validation NLL beats the base (preferring those whose prose NLL rises by at most --forget-budget,
a soft retention preference), the lowest validation NLL; ties (< 0.01) go to the smaller prose
degradation. Test NLL is shown ONLY for the selected cell (it is sealed in every cell's JSON and never used to choose).
Hard gates (exit 1): every cell's mechanics gates passed; the selected setting's test NLL is lower than the base's by at least
--min-gain nats. If no setting improves validation NLL, that is reported plainly and the job FAILS: the specialist then does not
generalise to unseen facts and the report says so. Retention (prose NLL, general probes) is always printed for every cell.
"""
import argparse
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--prefix", default="smollm2")
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--forget-budget", type=float, default=0.5, help="preferred max prose-NLL increase (nats) for the selected setting; soft")
    a = ap.parse_args()
    cells = []
    for f in sorted(glob.glob(os.path.join(a.dir, "**", "cell_*.json"), recursive=True)):
        c = json.load(open(f))
        if c["label"].startswith(a.prefix):
            cells.append(c)
    if not cells:
        print("no cells found under", a.dir)
        return 1
    ref = cells[0]["base"]
    print("GRID SUMMARY (%d cells) - base model: val NLL %.3f, prose NLL %.3f, general probes %d/10" % (len(cells), ref["val"], ref["prose"], ref["general"]))
    hdr = "%-28s %-5s %4s %6s %3s | %7s %7s | %7s %7s %6s | %7s | %s | %5s %5s" % (
        "cell", "mode", "r/L", "lr", "ep", "valBase", "valSpec", "proseB", "proseS", "dProse", "recall", "general", "steps", "sec")
    print(hdr)
    print("-" * len(hdr))
    bad = []
    for c in sorted(cells, key=lambda c: c["label"]):
        k, b, s, t = c["cell"], c["base"], c["spec"], c["train"]
        print("%-28s %-5s %4s %6.0e %3d | %7.3f %7.3f | %7.3f %7.3f %+6.2f | %7s | %2d->%-2d | %5d %5.0f" % (
            c["label"], k["mode"], k["lora_rank"] if k["mode"] == "lora" else k["layers"], k["lr"], k["epochs"], b["val"], s["val"],
            b["prose"], s["prose"], s["prose"] - b["prose"], s["train_fact_recall"], b["general"], s["general"], t["steps"], t["wall_s"]))
        if c.get("mechanics_failures"):
            bad.append((c["label"], c["mechanics_failures"]))
    print()
    ok = [c for c in cells if c["spec"]["val"] < c["base"]["val"]]
    rc = 0
    if bad:
        print("MECHANICS FAILURES:", json.dumps(bad))
        rc = 1
    if not ok:
        print("NO SETTING IMPROVES VALIDATION NLL on unseen facts (best: %s at %.3f vs base %.3f)." % (
            min(cells, key=lambda c: c["spec"]["val"])["label"], min(c["spec"]["val"] for c in cells), ref["val"]))
        print("=> In this grid the specialist memorises its training facts but does not generalise to unseen ones; test gate FAILS.")
        return 1
    within = [c for c in ok if c["spec"]["prose"] - c["base"]["prose"] <= a.forget_budget]
    pool = within or ok
    if not within:
        print("WARNING: every setting that helps validation also forgets more than %.2f nats of general prose; selecting among them anyway." % a.forget_budget)
    best = min(pool, key=lambda c: (round(c["spec"]["val"], 2), c["spec"]["prose"] - c["base"]["prose"]))
    b, s = best["base"], best["spec"]
    print("SELECTED on validation: %s (val NLL %.3f -> %.3f)" % (best["label"], b["val"], s["val"]))
    gain = b["test_sealed"] - s["test_sealed"]
    print("TEST (unseen entities, scored once for the selected setting): %.3f -> %.3f  (gain %+.3f nats/token)" % (b["test_sealed"], s["test_sealed"], gain))
    print("RETENTION for the selected setting: prose NLL %.3f -> %.3f (%+.3f nats), general probes %d/10 -> %d/10, trained-fact recall %s -> %s" % (
        b["prose"], s["prose"], s["prose"] - b["prose"], b["general"], s["general"], b["train_fact_recall"], s["train_fact_recall"]))
    if gain < a.min_gain:
        print("TEST GATE FAILED: the validation-selected setting does not improve unseen-fact NLL by %.2f nats." % a.min_gain)
        rc = 1
    else:
        print("TEST GATE PASSED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
