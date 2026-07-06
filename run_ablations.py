"""Ablations and controls requested by the review.

1. Network depth versus composition depth: GeoEquivariant of
   depth 1 to 4 on chains of 1 to 4 rotations (n=1000 iid).
2. EquiNorm / GradeGate ablation on rotation, cross and compose_rotation.
3. Non-Clifford control: scalarization with multiplicative coefficients
   (BilinearUnit) on the compositional tasks.
4. nested_cross: composition without rotation, all main models.

Output: results/ablations.json, incremental writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geonet_ext import (
    EXT_TASKS,
    GeoBiNetX,
    ScalarizationNet,
    build_model_ext,
    nparams,
    train_timed,
)


def run_cell(out, section, key, builder, task, n_train, train_split, test_split, args):
    rec_sec = out.setdefault(section, {})
    if key in rec_sec:
        return
    rec = {"nmse": [], "params": None}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        x_train, y_train = task.make(n_train, seed, train_split)
        x_test, y_test = task.make(args.n_test, seed + 1000, test_split)
        device = torch.device(args.device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_test, y_test = x_test.to(device), y_test.to(device)
        model = builder().to(device)
        rec["params"] = nparams(model)
        r = train_timed(model, x_train, y_train, x_test, y_test, epochs=args.epochs)
        rec["nmse"].append(r["nmse"])
    rec_sec[key] = rec
    t = torch.tensor(rec["nmse"], dtype=torch.float64)
    print(f"{section} {key} nmse mean={t.nanmean():.4g} min={t.min():.4g} max={t.max():.4g}", flush=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/ablations.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}

    # 1. network depth x chain length
    for length in range(1, 5):
        task = EXT_TASKS[f"chain_{length}"]
        for depth in range(1, 5):
            run_cell(
                out, "depth_vs_chain", f"L{length}_depth{depth}",
                lambda t=task, d=depth: GeoBiNetX(cin=t.k, depth=d, equi=True, embed=t.embed,
                                                  out_indices=t.out_indices, out_signs=t.out_signs),
                task, 1000, "iid", "iid", args,
            )
        run_cell(
            out, "depth_vs_chain", f"L{length}_scalarization",
            lambda t=task: ScalarizationNet(k=t.k),
            task, 1000, "iid", "iid", args,
        )

    # 2. ablation EquiNorm / GradeGate
    for task_name in ["rotation", "cross", "compose_rotation"]:
        task = EXT_TASKS[task_name]
        for label, gate, norm in [("full", True, True), ("no_gate", False, True),
                                  ("no_norm", True, False), ("neither", False, False)]:
            run_cell(
                out, "norm_gate", f"{task_name}_{label}",
                lambda t=task, g=gate, m=norm: GeoBiNetX(cin=t.k, equi=True, embed=t.embed,
                                                         out_indices=t.out_indices, out_signs=t.out_signs,
                                                         use_gate=g, use_norm=m),
                task, 1000, "iid", "iid", args,
            )

    # 3. multiplicative scalarization on the compositional tasks
    for task_name in ["compose_rotation", "torque_rotated_force"]:
        task = EXT_TASKS[task_name]
        for tag, n_train, tr, te in [("n100_iid", 100, "iid", "iid"),
                                     ("n1000_iid", 1000, "iid", "iid"),
                                     ("ood_axis", 2000, "north", "south")]:
            run_cell(
                out, "scal_multiplicative", f"{task_name}_{tag}",
                lambda t=task: ScalarizationNet(k=t.k, multiplicative=True),
                task, n_train, tr, te, args,
            )

    # 4. composition without rotation
    task = EXT_TASKS["nested_cross"]
    for model_name in ["MLP", "Scalarization", "VN-Cross", "GeoEquivariant"]:
        for tag, n_train, tr, te in task.protocol:
            run_cell(
                out, "nested_cross", f"{model_name}_{tag}",
                lambda m=model_name, t=task: build_model_ext(m, t),
                task, n_train, tr, te, args,
            )

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
