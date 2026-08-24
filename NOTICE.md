# Notice / Credits

DCOL++ is a C++ library combining two lineages of work. Please cite/credit
both if you use it.

## `dcolpp::socp` — ported from DifferentiableCollisions.jl

`dcolpp::socp` (headers under `include/dcolpp/socp/`) is a C++ port of
[**DifferentiableCollisions.jl**](https://github.com/kevin-tracy/DifferentiableCollisions.jl)
by **Kevin Tracy**, MIT licensed:

```
MIT License
Copyright (c) 2022 Kevin Tracy
```

This includes the geometric primitives (Polytope, Capsule, Cylinder, Cone,
Sphere, Polygon, Ellipsoid), the primal-dual interior-point second-order-cone
(SOCP) solver with Nesterov-Todd scaling, and the implicit-function-theorem
gradients/Jacobians that make proximity queries differentiable. The
algorithms and structure are Kevin Tracy's; this is a from-scratch C++/Eigen
re-implementation reparameterized on relative SE(3) poses instead of
quaternion/MRP world poses. Every ported source file's header comment names
the original `.jl` file it corresponds to. All credit for the underlying
method belongs to the original author — please cite the original repository
and its accompanying work if you use this engine.

## `dcolpp::implicit` — vendored from iDCOL

`dcolpp::implicit` (headers/sources under `include/dcolpp/implicit/` and
`src/implicit/`) is vendored from **iDCOL**, a research codebase by
**Anup Teejo Mathew** and collaborators, accompanying:

```bibtex
@misc{mathew2026collisiondetectionanalyticalderivatives,
      title={Collision Detection with Analytical Derivatives of Contact Kinematics},
      author={Anup Teejo Mathew and Anees Peringal and Daniele Caradonna and Frederic Boyer and Federico Renda},
      year={2026},
      eprint={2602.03250},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.03250},
}
```

Source: `D:\Research\Soft Robotics Toolbox\SoRoSim GitHub\iDCOL` (see also
[iDCOL.jl](https://github.com/GVSRobotics/iDCOL.jl) and the
[project website](https://gvsrobotics.github.io/iDCOL)). The `core/`
implicit-shape/Newton-KKT solver is vendored as a snapshot rather than a live
dependency — DCOL++ contributes the analytical relative-pose sensitivity
(`NewtonResult::T`) that was stubbed out in the source snapshot, built on
the SE(3) utilities shared with the `socp` engine.

## Shared infrastructure

`dcolpp::se3` (the SE(3) exponential map, its Jacobians, and analytical
pose-derivative primitives) is original to DCOL++, written to let both
engines differentiate proximity queries with respect to the relative SE(3)
pose `g` without depending on a third-party autodiff library.
