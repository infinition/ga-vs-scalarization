"""Visual demo: what exact rotation symmetry buys.

Two networks learn the same task from the same 100 examples: rotate a vector
by a given axis and angle. The plain MLP memorizes the examples it saw. The
equivariant network respects rotation symmetry by construction, so it is
correct for every orientation, including ones far from the training data.

The animation sweeps a rotation while both networks predict the result.
Green: exact answer. Blue: equivariant network (stays on target). Red: MLP.

Usage, from the repository root (needs torch and matplotlib):

    python demo/demo.py

Output: demo/demo.gif
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

from geonet_ext import EXT_TASKS, build_model_ext, train_timed

N_TRAIN = 100
EPOCHS = 500
FRAMES = 120
OUT = os.path.join(os.path.dirname(__file__), "demo.gif")


def train_models():
    task = EXT_TASKS["rotation"]
    torch.manual_seed(0)
    x_train, y_train = task.make(N_TRAIN, 0, "iid")
    x_test, y_test = task.make(1000, 1000, "iid")
    models = {}
    for name in ["MLP", "GeoEquivariant"]:
        torch.manual_seed(0)
        model = build_model_ext(name, task)
        r = train_timed(model, x_train, y_train, x_test, y_test, epochs=EPOCHS)
        print(f"{name}: NMSE {r['nmse']:.4f} after {EPOCHS} epochs on {N_TRAIN} examples")
        models[name] = model
    return models


def make_frames():
    """A rotation whose axis precesses while the angle sweeps."""
    p = torch.tensor([1.0, 0.35, 0.55])
    p = p / p.norm() * 1.25
    inputs = []
    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        phi = 2 * math.pi * t
        axis = torch.tensor([math.cos(phi) * 0.8, math.sin(phi) * 0.8, 0.6])
        axis = axis / axis.norm()
        angle = 0.3 + 2.6 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
        inputs.append(torch.cat([axis * angle, p]))
    return torch.stack(inputs), p


def rodrigues(u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    theta = u.norm(dim=1, keepdim=True).clamp_min(1e-9)
    k = u / theta
    c, s = torch.cos(theta), torch.sin(theta)
    return p * c + torch.cross(k, p.expand_as(k), dim=1) * s + k * (k * p).sum(1, keepdim=True) * (1 - c)


def main():
    models = train_models()
    x, p = make_frames()
    with torch.no_grad():
        preds = {name: m(x) for name, m in models.items()}
    truth = rodrigues(x[:, :3], p)
    err = {name: (preds[name] - truth).norm(dim=1) / truth.norm(dim=1) for name in preds}

    fig = plt.figure(figsize=(9, 4.2))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axerr = fig.add_subplot(1, 2, 2)
    colors = {"truth": "#2ca02c", "GeoEquivariant": "#1f77b4", "MLP": "#d62728"}

    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=colors[k], lw=3) for k in ["truth", "GeoEquivariant", "MLP"]]
    labels = ["exact answer", "equivariant net", "MLP"]

    def draw(i):
        ax3d.clear()
        ax3d.set_xlim(-1.3, 1.3), ax3d.set_ylim(-1.3, 1.3), ax3d.set_zlim(-1.3, 1.3)
        ax3d.set_axis_off()
        ax3d.view_init(elev=18, azim=25 + 0.4 * i)
        ax3d.set_title("Rotate a vector: predictions", fontsize=10)
        u = x[i, :3]
        ax3d.quiver(0, 0, 0, *(u / u.norm() * 1.25), color="#aaaaaa", lw=1.2, arrow_length_ratio=0.06)
        ax3d.text(*(u / u.norm() * 1.32), "axis", color="#888888", fontsize=8)
        for name, v in [("truth", truth[i]), ("GeoEquivariant", preds["GeoEquivariant"][i]), ("MLP", preds["MLP"][i])]:
            ax3d.quiver(0, 0, 0, *v, color=colors[name], lw=3.5, arrow_length_ratio=0.16)
        lo = max(0, i - 25)
        ax3d.plot(*truth[lo: i + 1].T, color=colors["truth"], lw=1.0, alpha=0.3)
        ax3d.plot(*preds["MLP"][lo: i + 1].T, color=colors["MLP"], lw=1.0, alpha=0.3)
        ax3d.legend(handles, labels, fontsize=8, loc="lower left", frameon=False)

        axerr.clear()
        for name in ["MLP", "GeoEquivariant"]:
            axerr.plot(err[name][: i + 1], color=colors[name], label=name)
        axerr.set_xlim(0, FRAMES), axerr.set_ylim(0, max(1.05, float(err["MLP"].max()) * 1.05))
        axerr.set_title("Relative prediction error", fontsize=10)
        axerr.set_xlabel("frame", fontsize=8)
        axerr.legend(fontsize=8, loc="upper right", frameon=False)
        axerr.grid(alpha=0.3)
        fig.suptitle("Same task, same 100 training examples", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=FRAMES)
    anim.save(OUT, writer=PillowWriter(fps=20), dpi=80)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
