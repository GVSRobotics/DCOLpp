# DCOL++ (DCOLpp)

A C++/Eigen differentiable-collision-detection library combining two
lineages of work:

- **`dcolpp::socp`** — a C++ port of
  [DifferentiableCollisions.jl](https://github.com/kevin-tracy/DifferentiableCollisions.jl)
  by **Kevin Tracy**: exact convex primitives (polytope, capsule, cylinder,
  cone, sphere, polygon, ellipsoid) and a custom primal-dual interior-point
  SOCP solver, differentiable via the implicit function theorem.
- **`dcolpp::implicit`** — vendored from **iDCOL**, a research codebase by
  Anup Teejo Mathew et al. accompanying
  ["Collision Detection with Analytical Derivatives of Contact Kinematics"](https://www.arxiv.org/abs/2602.03250):
  smooth strictly-convex implicit shapes solved via a fixed-size Newton-KKT
  system, warm-started and continuation-robust.

Both engines take the relative pose between two bodies as a single
`Eigen::Matrix4d g` (`g = g1⁻¹g2`) and differentiate proximity queries with
respect to `g`'s 6-dof local twist, using a small shared forward-mode
autodiff scalar (`dcolpp::Dual6`) rather than a third-party autodiff
dependency.

Full credits are in [NOTICE.md](NOTICE.md) — please keep them if you fork or
redistribute this code, and cite the original works above.

**Status: work in progress.** This README and full build/usage docs will be
filled in as each engine lands; see the project's implementation plan for
the current phase.

## Building

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Eigen3 is located via `find_package`; if none is found on your system,
CMake fetches it automatically.
