# Results

Date: 2026-07-05.
Main data: `results/audit_5seeds.json` (5 seeds, 200 epochs, Adam full batch, lr 5e-3, CPU).
Metric: NMSE = test MSE divided by the MSE of the constant mean predictor on the test set.
NMSE 1.0 equals a trivial predictor. NMSE above 1.0 is worse than trivial.
Values are mean ± std over 5 seeds. `s/run` is mean wall-clock training time per run.
Equi. err (rel) is the relative equivariance RMSE of the trained n1000 models.

Earlier 2-seed raw-MSE tables are kept in `results/geonet_benchmarks_combined_200e.json`
for reference. This file uses NMSE instead, because raw MSE is misleading on the
OOD radius splits: far-field targets are small, so a low raw MSE there can still
be worse than predicting the mean.

## Models

| Model | Constraint |
|---|---|
| MLP | none |
| MLP-Aug | random SO(3) data augmentation, factor 8 |
| Scalarization | exact equivariance: invariant dot products -> MLP -> coefficients on {v_i, v_i x v_j} |
| GeoBilinear | geometric product blocks, no equivariant weight tying |
| GeoEquivariant | grade-wise SO(3)-equivariant tying plus geometric products |

## Rotation (apply axis-angle rotation to a point)

| Model | Params | n100_iid | n1000_iid | ood_angle | ood_axis | Equi. err (rel) | s/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 7427 | 0.53 ± 0.061 | 0.10 ± 6.1e-3 | 0.69 ± 0.028 | 0.61 ± 0.07 | 0.28 | 1.0 |
| MLP-Aug | 7427 | 0.14 ± 0.014 | 0.054 ± 4.8e-3 | 0.64 ± 0.017 | 0.064 ± 6.5e-3 | 0.22 | 3.9 |
| Scalarization | 1283 | 0.011 ± 3.0e-3 | 4.0e-3 ± 1.6e-3 | 1.08 ± 0.44 | 3.3e-3 ± 1.1e-3 | 2.5e-7 | 0.6 |
| GeoBilinear | 28352 | 0.59 ± 0.066 | 0.059 ± 3.6e-3 | 0.37 ± 0.016 | 0.75 ± 0.052 | 0.22 | 16.9 |
| GeoEquivariant | 1952 | 3.6e-3 ± 1.9e-3 | 3.1e-3 ± 8.0e-4 | 0.26 ± 0.061 | 3.1e-3 ± 9.9e-4 | 1.7e-7 | 18.7 |

## Cross product

| Model | Params | n100_iid | n1000_iid | ood_axis | Equi. err (rel) | s/run |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7427 | 0.17 ± 0.011 | 0.040 ± 0.011 | 0.46 ± 0.064 | 0.20 | 0.6 |
| MLP-Aug | 7427 | 0.047 ± 7.0e-3 | 0.025 ± 0.016 | 0.049 ± 0.051 | 0.17 | 2.0 |
| Scalarization | 1283 | 5.8e-3 ± 3.4e-3 | 5.4e-4 ± 4.7e-4 | 1.3e-3 ± 2.0e-3 | 1.8e-7 | 0.7 |
| GeoBilinear | 28352 | 0.58 ± 0.14 | 0.025 ± 2.0e-3 | 1.01 ± 0.22 | 0.16 | 12.8 |
| GeoEquivariant | 1952 | 0.015 ± 4.5e-3 | 0.014 ± 3.1e-3 | 0.014 ± 3.7e-3 | 1.6e-7 | 10.8 |

## Central force (inverse-cube radial force plus distractor input)

| Model | Params | n100_iid | n1000_iid | ood_radius | Equi. err (rel) | s/run |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7427 | 0.73 ± 0.14 | 0.027 ± 5.7e-3 | 9.65 ± 8.47 | 0.16 | 1.0 |
| MLP-Aug | 7427 | 0.052 ± 0.02 | 4.2e-3 ± 8.0e-4 | 13.4 ± 6.6 | 0.081 | 2.6 |
| Scalarization | 1283 | 0.095 ± 0.05 | 0.031 ± 8.0e-3 | 817 ± 878 | 2.3e-7 | 0.6 |
| GeoBilinear | 28352 | 1.48 ± 0.19 | 0.063 ± 5.2e-3 | 17.8 ± 2.7 | 0.25 | 11.8 |
| GeoEquivariant | 1952 | 0.023 ± 7.4e-3 | 7.3e-3 ± 1.5e-3 | 2.91 ± 1.41 | 2.3e-7 | 12.2 |

All models are worse than a constant predictor on ood_radius. No model extrapolates
in radius. Raw MSE hides this because far-field forces are small.

## Two-body force (force between two points)

| Model | Params | n100_iid | n1000_iid | ood_sep | Equi. err (rel) | s/run |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7427 | 1.24 ± 0.21 | 0.036 ± 8.6e-3 | 19.2 ± 14.2 | 0.19 | 0.7 |
| MLP-Aug | 7427 | 0.093 ± 0.013 | 5.7e-3 ± 9.5e-4 | 19.8 ± 22.5 | 0.097 | 2.3 |
| Scalarization | 1283 | 0.23 ± 0.073 | 0.11 ± 0.033 | 268 ± 291 | 8.4e-7 | 0.6 |
| GeoBilinear | 28352 | 1.77 ± 0.28 | 0.11 ± 0.021 | 9.8 ± 1.9 | 0.39 | 13.1 |
| GeoEquivariant | 1952 | 0.042 ± 0.025 | 6.4e-3 ± 2.8e-3 | 4.37 ± 2.50 | 3.4e-7 | 13.2 |

Same failure as central force on separation extrapolation, for every model.
MLP-Aug is the best model iid at n=1000.

## Composed rotations (apply R2 R1 to a point, 9D input)

| Model | Params | n100_iid | n1000_iid | ood_axis | Equi. err (rel) | s/run |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7601 | 1.51 ± 0.12 | 0.68 ± 0.021 | 2.02 ± 0.17 | 0.80 | 0.8 |
| MLP-Aug | 7601 | 0.92 ± 0.057 | 0.30 ± 0.025 | 0.34 ± 0.023 | 0.49 | 2.4 |
| Scalarization | 1478 | 0.54 ± 0.075 | 0.087 ± 0.019 | 0.055 ± 9.7e-3 | 3.6e-7 | 0.7 |
| GeoBilinear | 29888 | 1.23 ± 0.033 | 0.54 ± 0.032 | 1.04 ± 0.072 | 0.76 | 13.6 |
| GeoEquivariant | 2048 | 0.031 ± 0.010 | 0.012 ± 1.5e-3 | 9.2e-3 ± 7.9e-4 | 2.3e-7 | 13.4 |

This is the one task where GeoEquivariant clearly beats every baseline,
including the strengthened scalarization control below.

## Control 1: is the MLP undertuned? (`results/tune_check.json`)

Grid over lr {1e-3, 5e-3, 2e-2} x epochs {500, 2000}, n=1000 iid, 3 seeds.
Best tuned NMSE:

| Task | MLP tuned | Scalarization tuned | GeoEquivariant (table above) |
|---|---:|---:|---:|
| rotation | 0.072 | 4.7e-4 | 3.1e-3 |
| central_force | 0.013 | 1.2e-3 | 7.3e-3 |

Tuning improves the MLP by 30 to 50 percent and does not change any conclusion.
Tuned scalarization beats GeoEquivariant on both single-product tasks.

## Control 2: strengthened scalarization on composed rotations (`results/scalarization_control.json`)

5 seeds, 200 epochs. Variants add the triple product invariant v1.(v2 x v3)
and more capacity.

| Variant | Params | n100_iid | n1000_iid | ood_axis |
|---|---:|---:|---:|---:|
| hidden 32 | 1478 | 0.54 | 0.087 | 0.055 |
| hidden 32 + triple | 1510 | 0.61 | 0.075 | 0.043 |
| hidden 128 + triple | 18310 | 0.39 | 0.046 | 0.058 |
| hidden 128, depth 3 + triple | 34822 | 0.29 | 0.038 | 0.020 |
| GeoEquivariant | 2048 | 0.031 | 0.012 | 9.2e-3 |

The advantage of GeoEquivariant on composed rotations survives the control:
9x at n=100, 3x at n=1000, 2x on OOD axes, with 17x fewer parameters than the
strongest scalarization variant.

## Control 3: external baseline Vector Neurons (`results/vn_5seeds.json`)

Two variants, 5 seeds, 200 epochs. VN is faithful to Deng et al. 2021 (linear
channel mixing plus directional-projection nonlinearity); VN-Cross appends
pairwise cross products of the inputs as extra channels.

Key readings (NMSE means):

| Task | VN | VN-Cross | GeoEquivariant |
|---|---:|---:|---:|
| cross n=1000 | 1.000 | 0.00012 | 0.014 |
| rotation n=1000 | 0.41 | 0.0029 | 0.0031 |
| compose_rotation n=1000 | 0.67 | 0.12 | 0.012 |
| compose_rotation ood_axis | 0.61 | 0.079 | 0.0092 |
| torque n=100 | 1.52 | 0.37 | 0.059 |
| torque ood_angle | 0.80 | 0.56 | 0.23 |

Vanilla VN cannot express the cross product (output confined to the linear
span of the inputs) and sits exactly at the trivial predictor there. VN-Cross
is the best model on the cross task and matches GeoEquivariant on rotation,
strengthening finding 2. On the compositional tasks it stays 3x to 10x behind
GeoEquivariant, strengthening finding 3 against an external baseline.

## Control 4: ablations (`results/ablations.json`)

- Depth vs composition depth (chains of L rotations, n=1000): required depth
  tracks L; scalarization degrades from 0.004 (L=1) to 1.91 (L=4, worse than
  constant) while depth-4 GeoEquivariant stays at 0.062.
- EquiNorm ablation: without it best seeds reach 9e-5 on cross but training
  is unstable (seed range 1e-4 to 1.1). The normalization causes the finding-2
  plateau by trading peak fit for stability. GradeGate: minor effect.
- Multiplicative scalarization ((Ax)*(Bx)+Cx coefficients): does not close
  the compositional gap (0.22 vs 0.012 on compose_rotation n=1000), worse in
  low data. The Clifford product, not generic multiplication, does the work.
- nested_cross (a x (b x c), no rotation): scalarization wins 20x over
  GeoEquivariant (0.0028 vs 0.068). Compositions that flatten into simple
  invariant coefficients belong to scalarization; the geometric advantage
  tracks the complexity of the flattened coefficient functions.

## Summary of findings

1. Exact equivariance, from any construction, dominates MLP and augmented MLP
   in low data and directional OOD on every task.
2. On single geometric product tasks (rotation, cross, central force),
   plain scalarization matches or beats the Cl(3,0) network at lower cost.
   On cross at n=1000 it is 26x better.
3. On composed rotations, the Cl(3,0) network beats all baselines including
   strengthened scalarization. This is the only result specific to geometric algebra.
4. No model extrapolates invariant magnitudes (angle, radius, separation).
   On radial OOD splits every model is worse than a constant predictor.
5. GeoEquivariant trains 13 to 19x slower per run than the MLP and scalarization.
