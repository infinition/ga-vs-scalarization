"""Geometric algebra layers for small SO(3)-equivariant experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_num_threads(4)


def cayley(p: int = 3) -> torch.Tensor:
    n = 2**p
    table = torch.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            sign = 1
            res = a ^ b
            bb = b
            while bb:
                low = bb & -bb
                i = low.bit_length() - 1
                sign *= (-1) ** bin(a >> (i + 1)).count("1")
                bb ^= low
            table[a, b, res] = sign
    return table


T3 = cayley(3)
GRADE = torch.tensor([bin(i).count("1") for i in range(8)])
IDX_VEC = [1, 2, 4]
IDX_BIV = [6, 5, 3]


def embed_vectors(x: torch.Tensor) -> torch.Tensor:
    """Embed two 3D vectors into two vector-grade multivector channels."""
    out = torch.zeros(x.shape[0], 2, 8, device=x.device, dtype=x.dtype)
    out[:, 0, IDX_VEC] = x[:, :3]
    out[:, 1, IDX_VEC] = x[:, 3:]
    return out


def embed_axis_vector(x: torch.Tensor) -> torch.Tensor:
    """Embed axis-angle as bivector channel and point as vector channel."""
    out = torch.zeros(x.shape[0], 2, 8, device=x.device, dtype=x.dtype)
    out[:, 0, IDX_BIV] = x[:, :3] * torch.tensor([1.0, -1.0, 1.0], device=x.device)
    out[:, 1, IDX_VEC] = x[:, 3:]
    return out


class EquiLinear(nn.Module):
    """Channel mixing with one scalar per grade, hence SO(3)-equivariant."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, 4) / math.sqrt(cin))
        self.register_buffer("grade", GRADE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight[:, :, self.grade]
        return torch.einsum("oic,bic->boc", w, x)


class FullLinear(nn.Module):
    """Unconstrained multivector linear layer, used as a non-equivariant control."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, 8, 8) / math.sqrt(cin * 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oicd,nid->noc", self.weight, x)


class GeoBilinear(nn.Module):
    """Geometric product bilinear block, y = gp(Ax, Bx) + Cx."""

    def __init__(self, cin: int, cout: int, equi: bool = True):
        super().__init__()
        linear = EquiLinear if equi else FullLinear
        self.a = linear(cin, cout)
        self.b = linear(cin, cout)
        self.c = linear(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a(x)
        b = self.b(x)
        return torch.einsum("noa,nob,abc->noc", a, b, T3.to(x.device)) + self.c(x)


class GradeGate(nn.Module):
    """Equivariant nonlinearity, one scalar gate per grade norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(channels, 4))
        self.b = nn.Parameter(torch.zeros(channels, 4))
        self.register_buffer("grade", GRADE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norms = []
        for g in range(4):
            mask = self.grade == g
            norms.append((x[:, :, mask].pow(2).sum(-1) + 1e-8).sqrt())
        n = torch.stack(norms, dim=-1)
        gate = torch.sigmoid(self.a * n + self.b)
        return x * gate[:, :, self.grade]


class EquiNorm(nn.Module):
    """Normalize each multivector channel to stabilize bilinear products."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1.0)


class GeoBiNet(nn.Module):
    def __init__(
        self,
        channels: int = 8,
        depth: int = 3,
        equi: bool = True,
        embed: Callable[[torch.Tensor], torch.Tensor] = embed_vectors,
        out_indices: Sequence[int] = IDX_VEC,
        out_signs: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.embed = embed
        self.out_indices = list(out_indices)
        self.register_buffer("out_signs", torch.tensor(out_signs, dtype=torch.float32))
        blocks = []
        cin = 2
        for _ in range(depth):
            blocks.append(nn.ModuleList([GeoBilinear(cin, channels, equi), GradeGate(channels), EquiNorm()]))
            cin = channels
        self.blocks = nn.ModuleList(blocks)
        self.out = EquiLinear(channels, 1) if equi else FullLinear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for bilinear, gate, norm in self.blocks:
            h = norm(gate(bilinear(h)))
        y = self.out(h)[:, 0, self.out_indices]
        return y * self.out_signs.to(y.device)


class MLP(nn.Module):
    def __init__(self, hidden: int = 58, depth: int = 3):
        super().__init__()
        dims = [6] + [hidden] * depth
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.head(h)


def random_rotation_matrix(n: int, seed: int, device: Optional[torch.device] = None) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    axis = torch.randn(n, 3, generator=gen, device=device)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
    angle = torch.rand(n, 1, generator=gen, device=device) * math.pi
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [
            zero, -kz, ky,
            kz, zero, -kx,
            -ky, kx, zero,
        ],
        dim=1,
    ).view(n, 3, 3)
    I = torch.eye(3, device=device).expand(n, 3, 3)
    aa = axis[:, :, None] * axis[:, None, :]
    c = torch.cos(angle).view(n, 1, 1)
    s = torch.sin(angle).view(n, 1, 1)
    return c * I + s * K + (1.0 - c) * aa


def rotate_batch(v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bj->bi", r, v)


def augment_so3(x: torch.Tensor, y: torch.Tensor, factor: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate rotated copies for a rotation-augmentation MLP baseline."""
    if factor <= 1:
        return x, y
    xs = [x]
    ys = [y]
    n = x.shape[0]
    for k in range(factor - 1):
        r = random_rotation_matrix(n, seed + 1009 * (k + 1), device=x.device)
        xr = x.clone()
        xr[:, :3] = rotate_batch(x[:, :3], r)
        xr[:, 3:] = rotate_batch(x[:, 3:], r)
        xs.append(xr)
        ys.append(rotate_batch(y, r))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@dataclass
class Task:
    name: str
    embed: Callable[[torch.Tensor], torch.Tensor]
    make: Callable[[int, int, str], Tuple[torch.Tensor, torch.Tensor]]
    out_indices: Tuple[int, int, int] = tuple(IDX_VEC)
    out_signs: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def rotation_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    axis = torch.randn(n, 3, generator=gen)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if split == "north":
        axis[:, 2] = axis[:, 2].abs()
    elif split == "south":
        axis[:, 2] = -axis[:, 2].abs()
    if split == "small_angle":
        angle = torch.rand(n, 1, generator=gen) * (math.pi / 2)
    elif split == "large_angle":
        angle = math.pi / 2 + torch.rand(n, 1, generator=gen) * (math.pi / 2)
    else:
        angle = torch.rand(n, 1, generator=gen) * math.pi
    point = torch.randn(n, 3, generator=gen)
    c = torch.cos(angle)
    s = torch.sin(angle)
    y = point * c + torch.cross(axis, point, dim=1) * s + axis * (axis * point).sum(1, keepdim=True) * (1 - c)
    return torch.cat([axis * angle, point], dim=1), y


def cross_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 3, generator=gen)
    b = torch.randn(n, 3, generator=gen)
    if split == "north":
        a[:, 2] = a[:, 2].abs()
        b[:, 2] = b[:, 2].abs()
    elif split == "south":
        a[:, 2] = -a[:, 2].abs()
        b[:, 2] = -b[:, 2].abs()
    return torch.cat([a, b], dim=1), torch.cross(a, b, dim=1)


def central_force_task(n: int, seed: int, split: str = "iid") -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    r = torch.randn(n, 3, generator=gen)
    if split == "near":
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8) * (0.25 + 0.75 * torch.rand(n, 1, generator=gen))
    elif split == "far":
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8) * (1.25 + 1.75 * torch.rand(n, 1, generator=gen))
    aux = torch.randn(n, 3, generator=gen)
    dist = r.norm(dim=1, keepdim=True).clamp_min(0.15)
    y = -r / (dist.pow(3) + 0.05)
    return torch.cat([r, aux], dim=1), y


TASKS: Dict[str, Task] = {
    "rotation": Task("rotation", embed_axis_vector, rotation_task),
    "cross": Task("cross", embed_vectors, cross_task, tuple(IDX_BIV), (1.0, -1.0, 1.0)),
    "central_force": Task("central_force", embed_vectors, central_force_task),
}


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 500,
    lr: float = 5e-3,
) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(x_train), y_train)
        if not torch.isfinite(loss):
            return float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        return float(F.mse_loss(model(x_test), y_test))


def build_model(kind: str, task: Task) -> nn.Module:
    if kind in {"MLP", "MLP-Aug"}:
        return MLP()
    if kind == "GeoBilinear":
        return GeoBiNet(
            channels=8,
            depth=3,
            equi=False,
            embed=task.embed,
            out_indices=task.out_indices,
            out_signs=task.out_signs,
        )
    if kind == "GeoEquivariant":
        return GeoBiNet(
            channels=8,
            depth=3,
            equi=True,
            embed=task.embed,
            out_indices=task.out_indices,
            out_signs=task.out_signs,
        )
    raise ValueError(kind)


def equivariance_error(model: nn.Module, task: Task, seed: int = 123, n: int = 256) -> float:
    x, _ = task.make(n, seed, "iid")
    r = random_rotation_matrix(n, seed + 100)
    xr = x.clone()
    xr[:, :3] = rotate_batch(x[:, :3], r)
    xr[:, 3:] = rotate_batch(x[:, 3:], r)
    with torch.no_grad():
        y1 = model(xr)
        y2 = rotate_batch(model(x), r)
    return float((y1 - y2).pow(2).mean().sqrt())


def summarize(values: List[float]) -> Dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "n": len(values)}
