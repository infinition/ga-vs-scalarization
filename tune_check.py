"""Control of flaw F3: is the MLP score an artifact of under-tuning?

Grid lr x epochs on the MLP (and Scalarization as control), 3 seeds,
n1000_iid configurations of rotation and central_force. If the best tuned MLP
stays far from GeoEquivariant, the benchmark comparison is defensible.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from geonet_ext import EXT_TASKS, build_model_ext, train_timed

OUT = Path("results/tune_check.json")
LRS = [1e-3, 5e-3, 2e-2]
EPOCHS = [500, 2000]
SEEDS = 3

out = {}
for task_name in ["rotation", "central_force"]:
    task = EXT_TASKS[task_name]
    out[task_name] = {}
    for model_name in ["MLP", "Scalarization"]:
        out[task_name][model_name] = {}
        for lr in LRS:
            for ep in EPOCHS:
                key = f"lr{lr}_ep{ep}"
                nmses = []
                for seed in range(SEEDS):
                    torch.manual_seed(seed)
                    x_train, y_train = task.make(1000, seed, "iid")
                    x_test, y_test = task.make(1000, seed + 1000, "iid")
                    model = build_model_ext(model_name, task)
                    r = train_timed(model, x_train, y_train, x_test, y_test, epochs=ep, lr=lr)
                    nmses.append(r["nmse"])
                t = torch.tensor(nmses, dtype=torch.float64)
                out[task_name][model_name][key] = {
                    "nmse_mean": float(t.nanmean()),
                    "nmse_per_seed": nmses,
                }
                print(task_name, model_name, key, f"nmse={float(t.nanmean()):.4g}", flush=True)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", OUT)
