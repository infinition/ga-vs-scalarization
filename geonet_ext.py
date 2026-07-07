"""Extensions for the audit: scalarization baseline, compositional tasks, NMSE, timing.

This module only reads from geonet_lib, it modifies nothing of the existing code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geonet_lib import (
    EquiLinear,
    EquiNorm,
    FullLinear,
    GeoBilinear,
    GradeGate,
    IDX_BIV,
    IDX_VEC,
    central_force_task,
    cross_task,
    embed_axis_vector,
    embed_vectors,
    random_rotation_matrix,
    rotate_batch,
    rotation_task,
)


BIV_SIGNS = torch.tensor([1.0, -1.0, 1.0])

from geonet_lib import T3

_NZ = T3.nonzero()
GP_A = _NZ[:, 0].contiguous()
GP_B = _NZ[:, 1].contiguous()
GP_C = _NZ[:, 2].contiguous()
GP_SIGN = T3[GP_A, GP_B, GP_C].contiguous()


_GP_TABLES = {}


def _gp_tables(device: torch.device):
    key = str(device)
    if key not in _GP_TABLES:
        _GP_TABLES[key] = tuple(t.to(device) for t in (GP_A, GP_B, GP_C, GP_SIGN))
    return _GP_TABLES[key]


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """gp(a, b) via the sparse Cayley table (64 non-zeros), math identical to the dense einsum."""
    ga, gb, gc, gs = _gp_tables(a.device)
    prod = a[..., ga] * b[..., gb] * gs
    out = torch.zeros_like(a)
    out.index_add_(-1, gc, prod)
    return out


class FastGeoBilinear(nn.Module):
    """Same computation as geonet_lib.GeoBilinear, sparse geometric product."""

    def __init__(self, cin: int, cout: int, equi: bool = True):
        super().__init__()
        linear = EquiLinear if equi else FullLinear
        self.a = linear(cin, cout)
        self.b = linear(cin, cout)
        self.c = linear(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return geometric_product(self.a(x), self.b(x)) + self.c(x)


class BilinearUnit(nn.Module):
    """Multiplicative unit: (Ax) * (Bx) + Cx. Non-Clifford control of the products."""

    def __init__(self, din: int, dout: int):
        super().__init__()
        self.a = nn.Linear(din, dout)
        self.b = nn.Linear(din, dout)
        self.c = nn.Linear(din, dout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) * self.b(x) + self.c(x)


class ScalarizationNet(nn.Module):
    """Minimal equivariant baseline.

    Invariant features (all dot products between the k input vectors)
    -> MLP -> coefficients on the equivariant basis {v_i} + {v_i x v_j, i<j}.
    Exactly SO(3)-equivariant by construction.
    """

    def __init__(self, k: int = 2, hidden: int = 32, depth: int = 2, triple: bool = False,
                 multiplicative: bool = False):
        super().__init__()
        self.k = k
        self.triple = triple and k >= 3
        self.multiplicative = multiplicative
        n_inv = k * (k + 1) // 2
        if self.triple:
            n_inv += k * (k - 1) * (k - 2) // 6
        n_basis = k + k * (k - 1) // 2
        dims = [n_inv] + [hidden] * depth
        unit = BilinearUnit if multiplicative else nn.Linear
        self.layers = nn.ModuleList([unit(dims[i], dims[i + 1]) for i in range(depth)])
        self.head = nn.Linear(hidden, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        vs = x.view(b, self.k, 3)
        invs = []
        basis = []
        for i in range(self.k):
            basis.append(vs[:, i])
            for j in range(i, self.k):
                invs.append((vs[:, i] * vs[:, j]).sum(-1))
        for i in range(self.k):
            for j in range(i + 1, self.k):
                basis.append(torch.cross(vs[:, i], vs[:, j], dim=-1))
        if self.triple:
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    for l in range(j + 1, self.k):
                        invs.append((torch.cross(vs[:, i], vs[:, j], dim=-1) * vs[:, l]).sum(-1))
        h = torch.stack(invs, dim=-1)
        for layer in self.layers:
            h = layer(h) if self.multiplicative else F.relu(layer(h))
        coeff = self.head(h)
        return (coeff.unsqueeze(-1) * torch.stack(basis, dim=1)).sum(1)


class GeoBiNetX(nn.Module):
    """Copy of GeoBiNet with a parametrizable number of input channels (cin)."""

    def __init__(
        self,
        cin: int = 2,
        channels: int = 8,
        depth: int = 3,
        equi: bool = True,
        embed: Callable[[torch.Tensor], torch.Tensor] = embed_vectors,
        out_indices: Sequence[int] = IDX_VEC,
        out_signs: Sequence[float] = (1.0, 1.0, 1.0),
        use_gate: bool = True,
        use_norm: bool = True,
    ):
        super().__init__()
        self.embed = embed
        self.out_indices = list(out_indices)
        self.register_buffer("out_signs", torch.tensor(out_signs, dtype=torch.float32))
        blocks = []
        c = cin
        for _ in range(depth):
            gate = GradeGate(channels) if use_gate else nn.Identity()
            norm = EquiNorm() if use_norm else nn.Identity()
            blocks.append(nn.ModuleList([FastGeoBilinear(c, channels, equi), gate, norm]))
            c = channels
        self.blocks = nn.ModuleList(blocks)
        self.out = EquiLinear(channels, 1) if equi else FullLinear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for bilinear, gate, norm in self.blocks:
            h = norm(gate(bilinear(h)))
        y = self.out(h)[:, 0, self.out_indices]
        return y * self.out_signs.to(y.device)


class MLPX(nn.Module):
    """MLP with a parametrizable input dimension."""

    def __init__(self, in_dim: int = 6, hidden: int = 58, depth: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.head(h)


def embed_biv_biv_vec(x: torch.Tensor) -> torch.Tensor:
    """Two axis-angles as bivectors + one point as vector (9D input, 3 channels)."""
    out = torch.zeros(x.shape[0], 3, 8, device=x.device, dtype=x.dtype)
    signs = BIV_SIGNS.to(x.device)
    out[:, 0, IDX_BIV] = x[:, :3] * signs
    out[:, 1, IDX_BIV] = x[:, 3:6] * signs
    out[:, 2, IDX_VEC] = x[:, 6:]
    return out


def embed_biv_vec(x: torch.Tensor) -> torch.Tensor:
    """One axis-angle as bivector + one vector as input (6D, 2 channels)."""
    out = torch.zeros(x.shape[0], 2, 8, device=x.device, dtype=x.dtype)
    signs = BIV_SIGNS.to(x.device)
    out[:, 0, IDX_BIV] = x[:, :3] * signs
    out[:, 1, IDX_VEC] = x[:, 3:6]
    return out


def embed_vec_biv_vec(x: torch.Tensor) -> torch.Tensor:
    """A lever vector, an axis-angle bivector, a local force vector (9D)."""
    out = torch.zeros(x.shape[0], 3, 8, device=x.device, dtype=x.dtype)
    signs = BIV_SIGNS.to(x.device)
    out[:, 0, IDX_VEC] = x[:, :3]
    out[:, 1, IDX_BIV] = x[:, 3:6] * signs
    out[:, 2, IDX_VEC] = x[:, 6:9]
    return out


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    return p * c + torch.cross(axis, p, dim=1) * s + axis * (axis * p).sum(1, keepdim=True) * (1 - c)


def _unit_axis(n: int, gen: torch.Generator, hemi: str | None) -> torch.Tensor:
    axis = torch.randn(n, 3, generator=gen)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if hemi == "north":
        axis[:, 2] = axis[:, 2].abs()
    elif hemi == "south":
        axis[:, 2] = -axis[:, 2].abs()
    return axis


def two_body_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    """Force on body 1: F = -(r1-r2)/(|r1-r2|^3 + 0.05). OOD on the separation."""
    gen = torch.Generator().manual_seed(seed)
    mid = torch.randn(n, 3, generator=gen)
    d = torch.randn(n, 3, generator=gen)
    if split == "near":
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-8) * (0.5 + 1.0 * torch.rand(n, 1, generator=gen))
    elif split == "far":
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-8) * (2.0 + 2.0 * torch.rand(n, 1, generator=gen))
    r1 = mid + d / 2
    r2 = mid - d / 2
    dist = d.norm(dim=1, keepdim=True).clamp_min(0.15)
    y = -d / (dist.pow(3) + 0.05)
    return torch.cat([r1, r2], dim=1), y


def compose_rotation_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    """Composition R2 R1 applied to p. Input (k1*a1, k2*a2, p), 9D. OOD on the axes."""
    gen = torch.Generator().manual_seed(seed)
    hemi = split if split in ("north", "south") else None
    axis1 = _unit_axis(n, gen, hemi)
    axis2 = _unit_axis(n, gen, hemi)
    ang1 = torch.rand(n, 1, generator=gen) * math.pi
    ang2 = torch.rand(n, 1, generator=gen) * math.pi
    p = torch.randn(n, 3, generator=gen)
    y = _rodrigues(axis2, ang2, _rodrigues(axis1, ang1, p))
    return torch.cat([axis1 * ang1, axis2 * ang2, p], dim=1), y


def local_force_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    """Local force transformed into the world frame: (orientation, f_local) -> R f_local.

    This is the minimal robotics case: action or force expressed in the local frame
    of a tool, expected effect in the world frame. OOD on the axes or the angle.
    """
    gen = torch.Generator().manual_seed(seed)
    hemi = split if split in ("north", "south") else None
    axis = _unit_axis(n, gen, hemi)
    if split == "small_angle":
        angle = torch.rand(n, 1, generator=gen) * (math.pi / 2)
    elif split == "large_angle":
        angle = math.pi / 2 + torch.rand(n, 1, generator=gen) * (math.pi / 2)
    else:
        angle = torch.rand(n, 1, generator=gen) * math.pi
    f_local = torch.randn(n, 3, generator=gen)
    y = _rodrigues(axis, angle, f_local)
    return torch.cat([axis * angle, f_local], dim=1), y


def torque_rotated_force_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    """World torque: (r, orientation, f_local) -> r x (R f_local).

    This task composes rotation and cross product. It represents a local force
    applied to a lever in the world frame.
    """
    gen = torch.Generator().manual_seed(seed)
    hemi = split if split in ("north", "south") else None
    r = torch.randn(n, 3, generator=gen)
    axis = _unit_axis(n, gen, hemi)
    if split == "small_angle":
        angle = torch.rand(n, 1, generator=gen) * (math.pi / 2)
    elif split == "large_angle":
        angle = math.pi / 2 + torch.rand(n, 1, generator=gen) * (math.pi / 2)
    else:
        angle = torch.rand(n, 1, generator=gen) * math.pi
    f_local = torch.randn(n, 3, generator=gen)
    f_world = _rodrigues(axis, angle, f_local)
    y = torch.cross(r, f_world, dim=1)
    return torch.cat([r, axis * angle, f_local], dim=1), y


def embed_vecs(k: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """k simple vectors as k grade-1 channels."""
    def embed(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], k, 8, device=x.device, dtype=x.dtype)
        for i in range(k):
            out[:, i, IDX_VEC] = x[:, 3 * i:3 * i + 3]
        return out
    return embed


def embed_bivs_vec(n_biv: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """n_biv axis-angles as bivectors followed by one vector."""
    def embed(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], n_biv + 1, 8, device=x.device, dtype=x.dtype)
        signs = BIV_SIGNS.to(x.device)
        for i in range(n_biv):
            out[:, i, IDX_BIV] = x[:, 3 * i:3 * i + 3] * signs
        out[:, n_biv, IDX_VEC] = x[:, 3 * n_biv:]
        return out
    return embed


def make_chain_rotation_task(length: int) -> Callable[[int, int, str], Tuple[torch.Tensor, torch.Tensor]]:
    """Chain of rotations: p -> R_L ... R_1 p. Input (u_1, ..., u_L, p)."""
    def task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        hemi = split if split in ("north", "south") else None
        parts = []
        rotations = []
        for _ in range(length):
            axis = _unit_axis(n, gen, hemi)
            angle = torch.rand(n, 1, generator=gen) * math.pi
            parts.append(axis * angle)
            rotations.append((axis, angle))
        p = torch.randn(n, 3, generator=gen)
        y = p
        for axis, angle in rotations:
            y = _rodrigues(axis, angle, y)
        parts.append(p)
        return torch.cat(parts, dim=1), y
    return task


def nested_cross_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    """Composition without rotation: (a, b, c) -> a x (b x c)."""
    gen = torch.Generator().manual_seed(seed)
    vs = []
    for _ in range(3):
        v = torch.randn(n, 3, generator=gen)
        if split == "north":
            v[:, 2] = v[:, 2].abs()
        elif split == "south":
            v[:, 2] = -v[:, 2].abs()
        vs.append(v)
    a, b, c = vs
    return torch.cat(vs, dim=1), torch.cross(a, torch.cross(b, c, dim=1), dim=1)


@dataclass
class ExtTask:
    name: str
    k: int
    embed: Callable[[torch.Tensor], torch.Tensor]
    make: Callable[[int, int, str], Tuple[torch.Tensor, torch.Tensor]]
    protocol: List[Tuple[str, int, str, str]]
    out_indices: Tuple[int, int, int] = tuple(IDX_VEC)
    out_signs: Tuple[float, float, float] = (1.0, 1.0, 1.0)


EXT_TASKS: Dict[str, ExtTask] = {
    "rotation": ExtTask(
        "rotation", 2, embed_axis_vector, rotation_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_angle", 2000, "small_angle", "large_angle"), ("ood_axis", 2000, "north", "south")],
    ),
    "cross": ExtTask(
        "cross", 2, embed_vectors, cross_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_axis", 1000, "north", "south")],
        tuple(IDX_BIV), (1.0, -1.0, 1.0),
    ),
    "central_force": ExtTask(
        "central_force", 2, embed_vectors, central_force_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_radius", 1500, "near", "far")],
    ),
    "two_body": ExtTask(
        "two_body", 2, embed_vectors, two_body_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_sep", 1500, "near", "far")],
    ),
    "compose_rotation": ExtTask(
        "compose_rotation", 3, embed_biv_biv_vec, compose_rotation_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_axis", 2000, "north", "south")],
    ),
    "local_force": ExtTask(
        "local_force", 2, embed_biv_vec, local_force_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_angle", 2000, "small_angle", "large_angle"), ("ood_axis", 2000, "north", "south")],
    ),
    "torque_rotated_force": ExtTask(
        "torque_rotated_force", 3, embed_vec_biv_vec, torque_rotated_force_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_angle", 2000, "small_angle", "large_angle"), ("ood_axis", 2000, "north", "south")],
        tuple(IDX_BIV), (1.0, -1.0, 1.0),
    ),
    "nested_cross": ExtTask(
        "nested_cross", 3, embed_vecs(3), nested_cross_task,
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid"),
         ("ood_axis", 1000, "north", "south")],
    ),
}

for _L in range(1, 5):
    EXT_TASKS[f"chain_{_L}"] = ExtTask(
        f"chain_{_L}", _L + 1, embed_bivs_vec(_L), make_chain_rotation_task(_L),
        [("n100_iid", 100, "iid", "iid"), ("n1000_iid", 1000, "iid", "iid")],
    )


class VNLinear(nn.Module):
    """Vector Neurons linear layer: combinations of vector channels (Deng et al. 2021)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin) / math.sqrt(cin))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, 3)
        return torch.einsum("oi,bic->boc", self.weight, x)


class VNReLU(nn.Module):
    """Vector Neurons non-linearity: q if <q,k> >= 0, else projection onto k orthogonal."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.q = VNLinear(cin, cout)
        self.k = VNLinear(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        k_hat = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot = (q * k_hat).sum(-1, keepdim=True)
        return torch.where(dot >= 0, q, q - dot * k_hat)


class VNNet(nn.Module):
    """Minimal Vector Neurons network for vector -> vector regression.

    Faithful to the VN principle: only linear combinations of vector
    channels and the non-linearity by directional projection, so the output
    stays in the linear span of the input channels. With cross=True, the
    cross products of the input pairs are added as channels, which
    lifts this limitation (common practice).
    """

    def __init__(self, k: int = 2, channels: int = 24, depth: int = 3, cross: bool = False):
        super().__init__()
        self.k = k
        self.cross = cross
        cin = k + (k * (k - 1) // 2 if cross else 0)
        layers = []
        for _ in range(depth):
            layers.append(VNReLU(cin, channels))
            cin = channels
        self.layers = nn.ModuleList(layers)
        self.out = VNLinear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        vs = list(x.view(b, self.k, 3).unbind(1))
        if self.cross:
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    vs.append(torch.cross(vs[i], vs[j], dim=-1))
        h = torch.stack(vs, dim=1)
        for layer in self.layers:
            h = layer(h)
        return self.out(h)[:, 0]


def build_model_ext(kind: str, task: ExtTask) -> nn.Module:
    in_dim = 3 * task.k
    if kind in {"MLP", "MLP-Aug"}:
        return MLPX(in_dim=in_dim)
    if kind == "Scalarization":
        return ScalarizationNet(k=task.k)
    if kind == "VN":
        return VNNet(k=task.k, cross=False)
    if kind == "VN-Cross":
        return VNNet(k=task.k, cross=True)
    if kind == "GeoBilinear":
        return GeoBiNetX(cin=task.k, equi=False, embed=task.embed,
                         out_indices=task.out_indices, out_signs=task.out_signs)
    if kind == "GeoEquivariant":
        return GeoBiNetX(cin=task.k, equi=True, embed=task.embed,
                         out_indices=task.out_indices, out_signs=task.out_signs)
    if kind == "E3NN":
        from geonet_e3nn import E3NNNet
        return E3NNNet(task.name, depth=3)
    raise ValueError(kind)


def rotate_blocks(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Applies the same per-sample rotation to each 3D block of the input."""
    out = x.clone()
    for i in range(x.shape[1] // 3):
        out[:, 3 * i:3 * i + 3] = rotate_batch(x[:, 3 * i:3 * i + 3], r)
    return out


def augment_so3_ext(x: torch.Tensor, y: torch.Tensor, factor: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if factor <= 1:
        return x, y
    xs, ys = [x], [y]
    n = x.shape[0]
    for kk in range(factor - 1):
        r = random_rotation_matrix(n, seed + 1009 * (kk + 1), device=x.device)
        xs.append(rotate_blocks(x, r))
        ys.append(rotate_batch(y, r))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def equivariance_error_ext(model: nn.Module, task: ExtTask, seed: int = 123, n: int = 256) -> Dict[str, float]:
    """Equivariance RMSE, raw and normalized by the RMS of the outputs."""
    x, _ = task.make(n, seed, "iid")
    r = random_rotation_matrix(n, seed + 100)
    with torch.no_grad():
        y1 = model(rotate_blocks(x, r))
        y2 = rotate_batch(model(x), r)
        rmse = float((y1 - y2).pow(2).mean().sqrt())
        rms = float(y2.pow(2).mean().sqrt())
    return {"rmse": rmse, "relative": rmse / max(rms, 1e-12)}


def train_timed(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 500,
    lr: float = 5e-3,
) -> Dict[str, float]:
    """Same loop as geonet_lib.train_model, with time and NMSE as output."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    diverged = False
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(x_train), y_train)
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    seconds = time.perf_counter() - t0
    if diverged:
        return {"mse": float("nan"), "nmse": float("nan"), "seconds": seconds}
    with torch.no_grad():
        mse = float(F.mse_loss(model(x_test), y_test))
        ref = float((y_test - y_test.mean(0, keepdim=True)).pow(2).mean())
    return {"mse": mse, "nmse": mse / max(ref, 1e-12), "seconds": seconds}


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
