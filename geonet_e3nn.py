"""External O(3)-equivariant baseline built from e3nn irreducible representations.

Optional dependency: import fails gracefully if e3nn is not installed, so the
rest of the benchmark works without it. Vector inputs are embedded as l=1
irreps, all under a single uniform parity (see E3NN_PARITIES below for why).
Each block is a self tensor product gated by scalar channels, followed by an
equivariant BatchNorm to control the magnitude growth of stacked tensor
products, the same role EquiNorm plays in the Clifford network.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from e3nn import o3
    from e3nn.math import soft_one_hot_linspace
    from e3nn.nn import BatchNorm, Gate

    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False

RADIAL_BASIS = 8
RADIAL_MAX = 6.0


HIDDEN_IRREPS_STR = "6x0e + 6x1e"

# All inputs and outputs are declared with a single, uniform parity ("e").
# This is a deliberate simplification: under proper rotations (the only
# transformations tested here, this benchmark is SO(3), not full O(3)), the
# Wigner D-matrix for a given l is identical regardless of the e/o parity
# label (verified numerically: D_from_matrix for "1o" and "1e" agree exactly
# for any rotation with det=+1). Mixing parities only matters for reflections,
# which are never tested, and empirically caused specific tensor-product
# output channels to be pruned to structural zero by e3nn's parity selection
# rule, silently killing the signal for some tasks (observed on "cross").
# A uniform parity avoids that failure mode entirely, at the cost of not
# encoding the true/pseudo-vector distinction. This is noted in the paper.
E3NN_PARITIES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "rotation": (("e", "e"), "e"),
    "cross": (("e", "e"), "e"),
    "central_force": (("e", "e"), "e"),
    "two_body": (("e", "e"), "e"),
    "compose_rotation": (("e", "e", "e"), "e"),
    "local_force": (("e", "e"), "e"),
    "torque_rotated_force": (("e", "e", "e"), "e"),
    "nested_cross": (("e", "e", "e"), "e"),
}
for _L in range(1, 5):
    E3NN_PARITIES[f"chain_{_L}"] = (tuple(["e"] * (_L + 1)), "e")


if E3NN_AVAILABLE:

    class TPBlock(nn.Module):
        """One equivariant block: self tensor product gated by scalar channels,
        followed by equivariant BatchNorm to control magnitude growth across depth."""

        def __init__(self, irreps_in: "o3.Irreps", irreps_hidden: "o3.Irreps"):
            super().__init__()
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in irreps_hidden if ir.l == 0])
            irreps_gated = o3.Irreps([(m, ir) for m, ir in irreps_hidden if ir.l > 0])
            irreps_gates = o3.Irreps([(m, "0e") for m, _ in irreps_gated])

            self.tp = o3.FullyConnectedTensorProduct(
                irreps_in, irreps_in, irreps_scalars + irreps_gates + irreps_gated
            )
            self.gate = Gate(
                irreps_scalars, [torch.tanh] * len(irreps_scalars),
                irreps_gates, [torch.sigmoid] * len(irreps_gates),
                irreps_gated,
            )
            self.norm = BatchNorm(self.gate.irreps_out)
            self.irreps_out = self.gate.irreps_out

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.gate(self.tp(x, x)))

    class E3NNNet(nn.Module):
        """Minimal e3nn network: k input vectors concatenated -> one output vector.

        Each input vector's norm is expanded into a small Gaussian radial
        basis (as in NequIP/Tensor Field Networks) and fed as extra invariant
        scalar channels alongside the raw l=1 vectors. Without this, the only
        source of scalar information is what a self tensor product of the raw
        vectors can construct, which struggled to represent functions of an
        angle magnitude (e.g. rotation by axis-angle) in early testing.
        """

        def __init__(self, task_name: str, depth: int = 2):
            super().__init__()
            parities, out_parity = E3NN_PARITIES[task_name]
            self.k = len(parities)
            self.irreps_in = o3.Irreps(" + ".join(f"1{p}" for p in parities))
            irreps_first = o3.Irreps(f"{self.k * RADIAL_BASIS}x0e") + self.irreps_in
            hidden = o3.Irreps(HIDDEN_IRREPS_STR)
            blocks = []
            irreps = irreps_first
            for _ in range(depth):
                block = TPBlock(irreps, hidden)
                blocks.append(block)
                irreps = block.irreps_out
            self.blocks = nn.ModuleList(blocks)
            self.out = o3.Linear(irreps, o3.Irreps(f"1{out_parity}"))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            vecs = x.view(b, self.k, 3)
            norms = vecs.norm(dim=-1)
            radial = soft_one_hot_linspace(
                norms, start=0.0, end=RADIAL_MAX, number=RADIAL_BASIS, basis="gaussian", cutoff=False
            ).reshape(b, -1)
            h = torch.cat([radial, x], dim=-1)
            for block in self.blocks:
                h = block(h)
            return self.out(h)

else:

    class E3NNNet(nn.Module):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("e3nn is not installed. pip install e3nn to use the E3NN baseline.")
