"""Export the trained RandomForest to model.json so index.html can run it in the browser.

Run once (needs the CSV next to this file):
    python export_model.py
Then upload index.html + model.json to GitHub Pages.
"""
import json
import os

import numpy as np

# Importing app trains the model exactly as the Flask backend does.
from app import model, MAE, FEATURES, LIMITS

trees = []
for est in model.estimators_:
    t = est.tree_
    trees.append({
        "l": t.children_left.tolist(),    # -1 means leaf
        "r": t.children_right.tolist(),
        "f": t.feature.tolist(),
        "t": t.threshold.tolist(),
        "v": [round(float(x), 4) for x in t.value[:, 0, 0]],
    })

out = {"features": FEATURES, "mae": round(float(MAE), 2), "trees": trees}
with open("model.json", "w") as fh:
    json.dump(out, fh, separators=(",", ":"))


# ---- Verify: re-implement the browser's tree walk and compare with sklearn ----
def walk(x32):
    total = 0.0
    for t in trees:
        i = 0
        while t["l"][i] != -1:
            i = t["l"][i] if x32[t["f"][i]] <= t["t"][i] else t["r"][i]
        total += t["v"][i]
    return total / len(trees)


rng = np.random.default_rng(0)
worst = 0.0
for _ in range(500):
    row = [rng.uniform(*LIMITS[name]) for name in FEATURES]
    x32 = np.float32(row)  # sklearn casts inputs to float32; the JS uses Math.fround
    expected = model.predict(np.array([row]))[0]
    worst = max(worst, abs(walk(x32) - expected))

size_mb = os.path.getsize("model.json") / 1e6
print(f"model.json written: {len(trees)} trees, {size_mb:.2f} MB")
print(f"Max difference vs sklearn over 500 random inputs: {worst:.5f}")
assert worst < 1e-3, "Export does not match sklearn. Do not deploy."
print("OK: export matches sklearn.")
