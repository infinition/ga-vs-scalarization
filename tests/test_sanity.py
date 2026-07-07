"""Sanity checks: exact equivariance, task correctness, sparse vs dense product.

Runs with plain python, exits nonzero on failure.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from geonet_lib import GeoBilinear, random_rotation_matrix, rotate_batch
from geonet_ext import (
    EXT_TASKS,
    FastGeoBilinear,
    ScalarizationNet,
    build_model_ext,
    compose_rotation_task,
    equivariance_error_ext,
    local_force_task,
    train_timed,
    torque_rotated_force_task,
    two_body_task,
)

FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    print(("PASS" if ok else "FAIL"), name, detail)
    if not ok:
        FAILURES.append(name)


def rotmat(u: torch.Tensor) -> torch.Tensor:
    theta = u.norm(dim=1, keepdim=True).clamp_min(1e-9)
    k = u / theta
    K = torch.zeros(u.shape[0], 3, 3)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    c = theta.cos().view(-1, 1, 1)
    s = theta.sin().view(-1, 1, 1)
    eye = torch.eye(3).expand_as(K)
    aa = k[:, :, None] * k[:, None, :]
    return c * eye + s * K + (1 - c) * aa


def main() -> int:
    torch.manual_seed(0)

    # 1. Exact equivariance of constrained models on every task.
    for tname, task in EXT_TASKS.items():
        for m in ["Scalarization", "GeoEquivariant"]:
            err = equivariance_error_ext(build_model_ext(m, task), task, seed=7)["relative"]
            check(f"equivariance {tname} {m}", err < 1e-5, f"rel_err={err:.2e}")

    # 2. Sparse geometric product equals dense einsum.
    old = GeoBilinear(2, 8, True)
    new = FastGeoBilinear(2, 8, True)
    new.load_state_dict(old.state_dict())
    x = torch.randn(200, 2, 8)
    with torch.no_grad():
        diff = float((old(x) - new(x)).abs().max())
    check("sparse gp equals dense", diff < 1e-4, f"max_diff={diff:.2e}")

    # 3. compose_rotation matches matrix composition.
    x, y = compose_rotation_task(300, 3, "iid")
    y_ref = torch.einsum(
        "bij,bj->bi", rotmat(x[:, 3:6]), torch.einsum("bij,bj->bi", rotmat(x[:, :3]), x[:, 6:])
    )
    diff = float((y - y_ref).abs().max())
    check("compose_rotation matches matrices", diff < 1e-4, f"max_diff={diff:.2e}")

    # 3b. local_force and torque_rotated_force match matrix formulas.
    x, y = local_force_task(300, 4, "iid")
    y_ref = torch.einsum("bij,bj->bi", rotmat(x[:, :3]), x[:, 3:6])
    diff = float((y - y_ref).abs().max())
    check("local_force matches matrices", diff < 1e-4, f"max_diff={diff:.2e}")

    x, y = torque_rotated_force_task(300, 6, "iid")
    f_world = torch.einsum("bij,bj->bi", rotmat(x[:, 3:6]), x[:, 6:9])
    y_ref = torch.cross(x[:, :3], f_world, dim=1)
    diff = float((y - y_ref).abs().max())
    check("torque_rotated_force matches formula", diff < 1e-4, f"max_diff={diff:.2e}")

    # 4. two_body target rotates with its inputs.
    x, y = two_body_task(300, 5, "iid")
    r = random_rotation_matrix(300, 99)
    d = rotate_batch(x[:, :3], r) - rotate_batch(x[:, 3:], r)
    dist = d.norm(dim=1, keepdim=True).clamp_min(0.15)
    diff = float((rotate_batch(y, r) - (-d / (dist.pow(3) + 0.05))).abs().max())
    check("two_body data equivariance", diff < 1e-4, f"max_diff={diff:.2e}")

    # 5. OOD splits use disjoint regions.
    xn, _ = two_body_task(500, 11, "near")
    xf, _ = two_body_task(500, 11, "far")
    sn = (xn[:, :3] - xn[:, 3:]).norm(dim=1).max()
    sf = (xf[:, :3] - xf[:, 3:]).norm(dim=1).min()
    check("two_body splits disjoint", float(sn) < float(sf), f"near_max={sn:.2f} far_min={sf:.2f}")

    # 6. Triple product variant stays equivariant.
    task = EXT_TASKS["compose_rotation"]
    model = ScalarizationNet(k=3, triple=True)
    err = equivariance_error_ext(model, task, seed=7)["relative"]
    check("scalarization triple equivariance", err < 1e-5, f"rel_err={err:.2e}")

    # 7. Short training beats the trivial predictor on an easy task.
    task = EXT_TASKS["cross"]
    torch.manual_seed(0)
    x_train, y_train = task.make(500, 0, "iid")
    x_test, y_test = task.make(500, 1000, "iid")
    r = train_timed(build_model_ext("Scalarization", task), x_train, y_train, x_test, y_test, epochs=100)
    check("scalarization learns cross", r["nmse"] < 0.5, f"nmse={r['nmse']:.3f}")

    # 8. E3NN baseline: equivariance on a few tasks, skipped if e3nn is absent.
    try:
        import geonet_e3nn
        has_e3nn = geonet_e3nn.E3NN_AVAILABLE
    except ImportError:
        has_e3nn = False
    if has_e3nn:
        for tname in ["rotation", "cross", "compose_rotation"]:
            task = EXT_TASKS[tname]
            err = equivariance_error_ext(build_model_ext("E3NN", task), task, seed=7)["relative"]
            check(f"equivariance {tname} E3NN", err < 1e-4, f"rel_err={err:.2e}")
    else:
        print("SKIP e3nn baseline checks (e3nn not installed)")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
