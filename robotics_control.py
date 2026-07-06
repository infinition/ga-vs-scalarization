"""Reinforced scalarization control on the robotics tasks.

Objective: verify whether the possible advantage of GeoEquivariant survives a
wider scalarization, with triple invariant when k=3.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from geonet_ext import EXT_TASKS, ScalarizationNet, nparams, train_timed

OUT = Path("results/robotics_control.json")
SEEDS = 5
EPOCHS = 200
TASKS = ["local_force", "torque_rotated_force"]

VARIANTS = {
    "scal_h32": dict(hidden=32, triple=False),
    "scal_h128": dict(hidden=128, triple=False),
    "scal_h128_d3": dict(hidden=128, depth=3, triple=False),
    "scal_h128_d3_triple": dict(hidden=128, depth=3, triple=True),
}


def main() -> None:
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for task_name in TASKS:
        task = EXT_TASKS[task_name]
        out.setdefault(task_name, {})
        for vname, kw in VARIANTS.items():
            if task.k < 3 and kw.get("triple"):
                continue
            rec_model = out[task_name].setdefault(vname, {"params": None, "configs": {}})
            for tag, n_train, train_split, test_split in task.protocol:
                if tag in rec_model["configs"]:
                    continue
                rec = {"nmse": [], "seconds": []}
                for seed in range(SEEDS):
                    torch.manual_seed(seed)
                    x_train, y_train = task.make(n_train, seed, train_split)
                    x_test, y_test = task.make(1000, seed + 1000, test_split)
                    model = ScalarizationNet(k=task.k, **kw)
                    rec_model["params"] = nparams(model)
                    r = train_timed(model, x_train, y_train, x_test, y_test, epochs=EPOCHS)
                    rec["nmse"].append(r["nmse"])
                    rec["seconds"].append(round(r["seconds"], 2))
                rec_model["configs"][tag] = rec
                t = torch.tensor(rec["nmse"], dtype=torch.float64)
                print(
                    task_name,
                    vname,
                    tag,
                    f"nmse mean={t.nanmean():.4g} min={t.min():.4g} max={t.max():.4g}",
                    flush=True,
                )
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
