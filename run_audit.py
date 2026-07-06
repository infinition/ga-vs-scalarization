"""Audit benchmark: 5 seeds, NMSE, timing, additional baselines, new tasks.

Incremental JSON writing, resumable if interrupted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geonet_ext import (
    EXT_TASKS,
    augment_so3_ext,
    build_model_ext,
    equivariance_error_ext,
    nparams,
    train_timed,
)

MODELS = ["MLP", "MLP-Aug", "Scalarization", "GeoBilinear", "GeoEquivariant"]


def run(args: argparse.Namespace) -> dict:
    if args.out.exists() and args.resume:
        out = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        out = {
            "meta": {
                "epochs": args.epochs,
                "seeds": args.seeds,
                "n_test": args.n_test,
                "aug_factor": args.aug_factor,
                "lr": args.lr,
                "torch": torch.__version__,
            },
            "results": {},
        }
    task_names = args.tasks.split(",") if args.tasks else list(EXT_TASKS)
    model_names = args.models.split(",") if args.models else MODELS
    for task_name in task_names:
        task = EXT_TASKS[task_name]
        res_task = out["results"].setdefault(task_name, {})
        for model_name in model_names:
            res_model = res_task.setdefault(model_name, {"params": None, "configs": {}, "equivariance": []})
            for tag, n_train, train_split, test_split in task.protocol:
                if tag in res_model["configs"]:
                    continue
                rec = {"mse": [], "nmse": [], "seconds": []}
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    x_train, y_train = task.make(n_train, seed, train_split)
                    x_test, y_test = task.make(args.n_test, seed + 1000, test_split)
                    if model_name == "MLP-Aug":
                        x_train, y_train = augment_so3_ext(x_train, y_train, args.aug_factor, seed)
                    model = build_model_ext(model_name, task)
                    res_model["params"] = nparams(model)
                    r = train_timed(model, x_train, y_train, x_test, y_test, epochs=args.epochs, lr=args.lr)
                    rec["mse"].append(r["mse"])
                    rec["nmse"].append(r["nmse"])
                    rec["seconds"].append(round(r["seconds"], 2))
                    if tag == "n1000_iid":
                        res_model["equivariance"].append(equivariance_error_ext(model, task, seed=seed))
                res_model["configs"][tag] = rec
                t = torch.tensor(rec["nmse"], dtype=torch.float64)
                print(
                    f"{task_name} {model_name} {tag} "
                    f"nmse mean={t.nanmean():.4g} min={t.min():.4g} max={t.max():.4g} "
                    f"sec/run={sum(rec['seconds'])/len(rec['seconds']):.1f}",
                    flush=True,
                )
                args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--aug-factor", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--out", type=Path, default=Path("results/audit_5seeds.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
