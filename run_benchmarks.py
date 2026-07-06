"""Run Exp0 geometric equivariance benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geonet_lib import TASKS, augment_so3, build_model, equivariance_error, nparams, summarize, train_model


def protocol_for(task_name: str):
    if task_name == "rotation":
        return [
            ("n100_iid", 100, "iid", "iid"),
            ("n1000_iid", 1000, "iid", "iid"),
            ("ood_angle", 2000, "small_angle", "large_angle"),
            ("ood_axis", 2000, "north", "south"),
        ]
    if task_name == "cross":
        return [
            ("n100_iid", 100, "iid", "iid"),
            ("n1000_iid", 1000, "iid", "iid"),
            ("ood_axis", 1000, "north", "south"),
        ]
    if task_name == "central_force":
        return [
            ("n100_iid", 100, "iid", "iid"),
            ("n1000_iid", 1000, "iid", "iid"),
            ("ood_radius", 1500, "near", "far"),
        ]
    raise ValueError(task_name)


def run(args: argparse.Namespace) -> dict:
    if args.out.exists():
        out = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        out = {
            "meta": {
                "epochs": args.epochs,
                "seeds": args.seeds,
                "torch": torch.__version__,
                "aug_factor": args.aug_factor,
            },
            "results": {},
        }
    selected_models = args.models.split(",") if args.models else ["MLP", "MLP-Aug", "GeoBilinear", "GeoEquivariant"]
    for task_name, task in TASKS.items():
        out["results"].setdefault(task_name, {})
        for model_name in selected_models:
            out["results"][task_name].setdefault(
                model_name,
                {"params": None, "scores": {}, "equivariance_rmse": []},
            )
            for tag, n_train, train_split, test_split in protocol_for(task_name):
                if tag in out["results"][task_name][model_name]["scores"]:
                    continue
                scores = []
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    x_train, y_train = task.make(n_train, seed, train_split)
                    x_test, y_test = task.make(args.n_test, seed + 1000, test_split)
                    if model_name == "MLP-Aug":
                        x_train, y_train = augment_so3(x_train, y_train, args.aug_factor, seed)
                    model = build_model(model_name, task)
                    out["results"][task_name][model_name]["params"] = nparams(model)
                    score = train_model(model, x_train, y_train, x_test, y_test, epochs=args.epochs)
                    scores.append(score)
                out["results"][task_name][model_name]["scores"][tag] = summarize(scores)
                print(task_name, model_name, tag, out["results"][task_name][model_name]["scores"][tag], flush=True)
                args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
            if isinstance(out["results"][task_name][model_name]["equivariance_rmse"], list):
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    model = build_model(model_name, task)
                    x_train, y_train = task.make(512, seed, "iid")
                    x_test, y_test = task.make(512, seed + 1000, "iid")
                    if model_name == "MLP-Aug":
                        x_train, y_train = augment_so3(x_train, y_train, args.aug_factor, seed)
                    train_model(model, x_train, y_train, x_test, y_test, epochs=max(50, args.epochs // 2))
                    out["results"][task_name][model_name]["equivariance_rmse"].append(
                        equivariance_error(model, task, seed=seed)
                    )
                out["results"][task_name][model_name]["equivariance_rmse"] = summarize(
                    out["results"][task_name][model_name]["equivariance_rmse"]
                )
                args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--aug-factor", type=int, default=8)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--out", type=Path, default=Path("results/geonet_benchmarks.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
