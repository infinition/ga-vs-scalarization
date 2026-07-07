"""Generates the paper figures from the result JSON files.

Dependencies: matplotlib (no torch). Usage, from the repository root:

    python paper/make_figures.py

Figures produced in paper/figures/:
- overview.pdf: NMSE per task and model, log scale, 5-seed error bars.
- sample_efficiency.pdf: only if results/curves.json exists (n_train runs
  to launch, see TODO.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

MODELS = ["MLP", "MLP-Aug", "Scalarization", "VN-Cross", "E3NN", "GeoEquivariant"]
COLORS = {
    "MLP": "#888888",
    "MLP-Aug": "#b8860b",
    "Scalarization": "#1f77b4",
    "VN": "#8c564b",
    "VN-Cross": "#9467bd",
    "E3NN": "#2ca02c",
    "GeoEquivariant": "#d62728",
}

TASKS = [
    ("rotation", "audit_5seeds.json", "n100_iid"),
    ("cross", "audit_5seeds.json", "n100_iid"),
    ("central_force", "audit_5seeds.json", "n100_iid"),
    ("two_body", "audit_5seeds.json", "n100_iid"),
    ("compose_rotation", "audit_5seeds.json", "n100_iid"),
    ("torque_rotated_force", "robotics_5seeds.json", "n100_iid"),
]

LABELS = {
    "rotation": "rotation",
    "cross": "cross",
    "central_force": "central\nforce",
    "two_body": "two\nbody",
    "compose_rotation": "composed\nrotations",
    "torque_rotated_force": "torque",
}


def load(fname: str) -> dict:
    """Merges the main file of the task with the VN baselines."""
    data = json.loads((ROOT / "results" / fname).read_text(encoding="utf-8"))["results"]
    vn = ROOT / "results" / "vn_5seeds.json"
    if vn.exists():
        for task, models in json.loads(vn.read_text(encoding="utf-8"))["results"].items():
            data.setdefault(task, {}).update(models)
    return data


def overview() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, tag, title in [
        (axes[0], "n100_iid", "n = 100, iid"),
        (axes[1], None, "key OOD split"),
    ]:
        width = 0.15
        for mi, model in enumerate(MODELS):
            xs, ys, errs = [], [], []
            for ti, (task, fname, _) in enumerate(TASKS):
                data = load(fname)[task]
                if model not in data:
                    continue
                cfg = data[model]["configs"]
                use_tag = tag
                if use_tag is None:
                    ood = [t for t in cfg if t.startswith("ood")]
                    use_tag = ood[0]
                v = cfg[use_tag]["nmse"]
                mean = sum(v) / len(v)
                std = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
                xs.append(ti + (mi - 2.0) * width)
                ys.append(mean)
                errs.append(std)
            ax.bar(xs, ys, width=width, yerr=errs, label=model, color=COLORS[model], error_kw={"lw": 0.8})
        ax.set_yscale("log")
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([LABELS[t] for t, _, _ in TASKS], fontsize=8)
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3, which="both")
    axes[0].set_ylabel("NMSE (log)")
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "overview.pdf")
    print("wrote", FIGDIR / "overview.pdf")


def sample_efficiency() -> None:
    src = ROOT / "results" / "curves.json"
    if not src.exists():
        print("results/curves.json missing, sample_efficiency figure not generated")
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 3.4), sharey=True, squeeze=False)
    for ax, (task, models) in zip(axes[0], data.items()):
        for model, pts in models.items():
            ns = sorted(int(n) for n in pts)
            means, los, his = [], [], []
            for n in ns:
                v = pts[str(n)]
                m = sum(v) / len(v)
                s = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
                means.append(m)
                los.append(max(m - s, 1e-6))
                his.append(m + s)
            color = COLORS.get(model)
            ax.plot(ns, means, marker="o", label=model, color=color)
            ax.fill_between(ns, los, his, color=color, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n train")
        ax.set_title(task, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("NMSE (log)")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "sample_efficiency.pdf")
    print("wrote", FIGDIR / "sample_efficiency.pdf")


if __name__ == "__main__":
    overview()
    sample_efficiency()
