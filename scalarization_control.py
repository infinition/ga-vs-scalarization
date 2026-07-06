"""Fairness control on compose_rotation: does the GeoEquivariant win hold up
against a reinforced scalarization (triple product, more capacity)?
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from geonet_ext import EXT_TASKS, ScalarizationNet, nparams, train_timed

OUT = Path("results/scalarization_control.json")
SEEDS = 5
EPOCHS = 200

VARIANTS = {
    "scal_h32": dict(hidden=32, triple=False),
    "scal_h32_triple": dict(hidden=32, triple=True),
    "scal_h128_triple": dict(hidden=128, triple=True),
    "scal_h128_d3_triple": dict(hidden=128, depth=3, triple=True),
}

task = EXT_TASKS["compose_rotation"]
out = {}
for vname, kw in VARIANTS.items():
    out[vname] = {"params": None, "configs": {}}
    for tag, n_train, train_split, test_split in task.protocol:
        rec = {"nmse": [], "seconds": []}
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            x_train, y_train = task.make(n_train, seed, train_split)
            x_test, y_test = task.make(1000, seed + 1000, test_split)
            model = ScalarizationNet(k=task.k, **kw)
            out[vname]["params"] = nparams(model)
            r = train_timed(model, x_train, y_train, x_test, y_test, epochs=EPOCHS)
            rec["nmse"].append(r["nmse"])
            rec["seconds"].append(round(r["seconds"], 2))
        out[vname]["configs"][tag] = rec
        t = torch.tensor(rec["nmse"], dtype=torch.float64)
        print(vname, tag, f"nmse mean={t.nanmean():.4g} min={t.min():.4g} max={t.max():.4g}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", OUT)
