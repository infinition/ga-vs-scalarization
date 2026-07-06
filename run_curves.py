"""Data efficiency curves for the main figure of the paper.

NMSE as a function of n_train on the compositional tasks, 10 seeds.
Output in the format expected by paper/make_figures.py:
    {task: {model: {str(n): [nmse per seed]}}}
Supports --device cuda. Incremental writing, resumable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from geonet_ext import EXT_TASKS, build_model_ext, train_timed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="compose_rotation,torque_rotated_force")
    parser.add_argument("--models", type=str, default="Scalarization,GeoEquivariant")
    parser.add_argument("--ns", type=str, default="30,100,300,1000,3000")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/curves.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}

    for task_name in args.tasks.split(","):
        task = EXT_TASKS[task_name]
        res_task = out.setdefault(task_name, {})
        for model_name in args.models.split(","):
            res_model = res_task.setdefault(model_name, {})
            for n in [int(x) for x in args.ns.split(",")]:
                key = str(n)
                if key in res_model:
                    continue
                nmses = []
                t0 = time.perf_counter()
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    x_train, y_train = task.make(n, seed, "iid")
                    x_test, y_test = task.make(args.n_test, seed + 1000, "iid")
                    x_train, y_train = x_train.to(device), y_train.to(device)
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    model = build_model_ext(model_name, task).to(device)
                    r = train_timed(model, x_train, y_train, x_test, y_test, epochs=args.epochs)
                    nmses.append(r["nmse"])
                res_model[key] = nmses
                t = torch.tensor(nmses, dtype=torch.float64)
                print(
                    f"{task_name} {model_name} n={n} "
                    f"nmse mean={t.nanmean():.4g} min={t.min():.4g} max={t.max():.4g} "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
                args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
