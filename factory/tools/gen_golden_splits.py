"""Generates studio-core/src/test/resources/golden/splits.v1.json from the Python reference (llmtrainer.splits).

The Kotlin port in studio-core (Similarity.kt) must reproduce these values exactly: BLAKE2b-64, shingles, MinHash,
near-duplicate pairs, containment, group assignment. Run: python factory/tools/gen_golden_splits.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "factory"))
from llmtrainer import splits as sp  # noqa: E402

STRINGS = ["", "a", "a0", "a1", "b63", "hello world", "the quick brown fox jumps over the lazy dog " * 4,
           "x" * 127, "x" * 128, "x" * 129, "x" * 300, "torque µ café – N·m"]

TEXTS = {
    "a": "Tighten the rear axle nut to 85 N-m and then fit a new split pin before refitting the wheel on the swing arm of the machine.",
    "b": "Tighten the rear axle nut to 85 N-m and then fit a new split pin before refitting the wheel on the swing arm of the machine today.",
    "c": "Bleed the front brake by pumping the lever slowly while an assistant opens the bleed nipple and watches the fluid level in the reservoir.",
    "d": "Check the valve clearance with a feeler gauge when the engine is cold and record every reading before adjusting any shim at all.",
    "e": "short text here",
    "f": "Check the valve clearance with a feeler gauge when the engine is cold and record every reading before adjusting any shim at all.",
}

out: dict = {"blake2b8": [{"s": s, "h": sp._h64(s)} for s in STRINGS]}
out["perm_a0"] = sp._PERMS[0][0]
out["perm_b0"] = sp._PERMS[0][1]
out["perm_a63"] = sp._PERMS[63][0]
out["shingles"] = {k: sorted(sp.shingles(t, 5)) for k, t in TEXTS.items()}
out["minhash"] = {k: list(sp.minhash(sp.shingles(t, 5))) for k, t in TEXTS.items()}
out["texts"] = TEXTS
out["near_dups_08"] = [[a, b, j] for a, b, j in sp.find_near_duplicates(TEXTS, 0.8, 5)]
out["near_dups_05"] = [[a, b, j] for a, b, j in sp.find_near_duplicates(TEXTS, 0.5, 5)]
cand = {"x1": "Tighten the rear axle nut to 85 N-m and then fit a new split pin", "x2": "An unrelated sentence about nothing at all in this garage"}
out["containment"] = {"candidates": cand, "protected": {"p": TEXTS["a"]},
                      "result": [[a, b, c] for a, b, c in sp.containment_leaks(cand, {"p": TEXTS["a"]}, 0.8, 5)]}
weights = {f"g{i}": (i * 7) % 11 + 3 for i in range(14)}
out["assign"] = []
for seed in (0, 1, 1234, 99999):
    for ratios in ({"train": 0.7, "validation": 0.15, "test": 0.15}, {"train": 0.8, "validation": 0.1, "test": 0.1}):
        out["assign"].append({"seed": seed, "ratios": ratios, "weights": weights, "result": sp.assign_groups(weights, ratios, seed)})
dest = ROOT / "studio-core" / "src" / "test" / "resources" / "golden" / "splits.v1.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
print("wrote", dest)
