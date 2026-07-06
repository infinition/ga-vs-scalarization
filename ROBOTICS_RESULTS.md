# Robotics Composition Results

Date: 2026-07-05.
Branch: `robotics-composition`.
Data:

- `results/robotics_5seeds.json`, 5 seeds, 200 epochs, NMSE, all four models.
- `results/robotics_control.json`, strengthened scalarization controls.

Metric: NMSE, lower is better. NMSE 1.0 equals the constant mean predictor.

## Question

The audit found that scalarization is enough on simple single-product laws, but
GeoEquivariant wins on composed rotations. The robotics question is whether this
advantage appears in a practical local-to-world setting:

- local action or force expressed in a tool frame,
- orientation maps it to the world frame,
- optional lever arm composes this with a torque.

## Tasks

### `local_force`

Input: `(axis * angle, f_local)`. Target: `R(axis, angle) f_local`.

Caveat: this task is mathematically identical to the existing `rotation` task,
same sampling order, same splits, same targets. The per-seed results are equal
to the `rotation` rows of `results/audit_5seeds.json`. It is kept only as the
robotics framing of that task and counts as zero new evidence. Any claim based
on it is a claim about the rotation task.

### `torque_rotated_force` (new evidence)

Input: `(r, axis * angle, f_local)`. Target: `r x (R(axis, angle) f_local)`.

This composes a rotation with a cross product, modeling a local force applied at
a world-frame lever arm. This is the second genuinely compositional task after
`compose_rotation`.

## Main results (5 seeds, mean NMSE)

### local_force (identical to rotation, shown for completeness)

| Model | Params | n100 iid | n1000 iid | OOD angle | OOD axis | s/run |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7427 | 0.53 | 0.10 | 0.69 | 0.61 | 1.9 |
| MLP-Aug | 7427 | 0.14 | 0.054 | 0.64 | 0.064 | 4.7 |
| Scalarization | 1283 | 0.011 | 0.0040 | 1.08 | 0.0033 | 1.3 |
| GeoEquivariant | 1952 | 0.0036 | 0.0031 | 0.26 | 0.0031 | 22.4 |

### torque_rotated_force

| Model | Params | n100 iid | n1000 iid | OOD angle | OOD axis | s/run |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7601 | 1.35 | 0.66 | 1.08 | 1.16 | 1.0 |
| MLP-Aug | 7601 | 0.85 | 0.37 | 0.87 | 0.40 | 4.1 |
| Scalarization | 1478 | 0.145 | 0.029 | 0.64 | 0.020 | 0.8 |
| GeoEquivariant | 2048 | 0.059 | 0.030 | 0.23 | 0.031 | 18.3 |

Readout:

- The unconstrained MLP fails on this task even iid at n=1000 (NMSE 0.66).
  Augmentation helps but stays above 0.37 everywhere.
- GeoEquivariant wins in low data (2.5x vs scalarization) and on OOD angle (2.8x).
- Scalarization is equal at n=1000 iid and better on OOD axis (0.020 vs 0.031).
- The differentiator is angle shift, consistent with the rotation and
  compose_rotation results: NMSE 0.23 vs 0.64, both still far from solved.

## Strengthened scalarization controls

Scalarization widened to hidden 128, depth 3, triple product invariant where k=3.

| Task | Variant | Params | n100 iid | n1000 iid | OOD angle | OOD axis |
|---|---|---:|---:|---:|---:|---:|
| local_force | scal_h128 | 17411 | 0.0083 | 0.0017 | 0.49 | 0.0010 |
| local_force | scal_h128_d3 | 33923 | 0.010 | 0.00095 | 1.82 | 0.00074 |
| torque | scal_h128 | 18182 | 0.139 | 0.028 | 0.46 | 0.020 |
| torque | scal_h128_d3 | 34694 | 0.114 | 0.022 | 0.58 | 0.019 |
| torque | scal_h128_d3_triple | 34822 | 0.146 | 0.020 | 0.43 | 0.015 |
| torque | GeoEquivariant (ref) | 2048 | 0.059 | 0.030 | 0.23 | 0.031 |

Readout:

- Widening scalarization improves iid and OOD axis, where it already led.
- It does not close OOD angle: best variant 0.43 against 0.23, still 1.9x behind
  with 17x more parameters. On local_force the h128_d3 variant even degrades
  OOD angle to 1.82, worse than trivial, a sign of unstable extrapolation of the
  learned coefficient functions.

## Interpretation

The robotics tasks refine the audit conclusion:

Scalarization stays the best cheap choice for simple equivariant laws and for
axis-shift generalization.Cl(3,0) bilinear layers earn their cost in two
regimes: low data on compositional targets, and angle shift when a local
quantity must be mapped through a rotation. Neither approach solves angle
extrapolation, GeoEquivariant only degrades more slowly (NMSE 0.23 to 0.26
against 0.43 to 1.8).

## Notes

- `torque_rotated_force` is a second compositional task, independent of
  `compose_rotation`, and confirms the same pattern.
- `local_force` is the rotation task under a robotics framing (same draws, same
  splits, identical per-seed values). It does not count as new evidence.
- GeoEquivariant is 10 to 20x slower to train than scalarization at these sizes.
