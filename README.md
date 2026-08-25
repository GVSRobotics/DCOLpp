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
respect to `g`'s 6-dof local twist. `dcolpp::socp` computes that derivative
— the Jacobian, the cheaper alpha-only gradient, and the second derivative
(`d²α/dξ²`, for the contact normal's own sensitivity) — with hand-derived,
non-templated analytical formulas
(`include/dcolpp/socp/analytic_derivatives.hpp`) for every primitive pair.
No autodiff dependency anywhere in the codebase; see
[DEVIATIONS.md](DEVIATIONS.md) for the details, including what this adds
beyond DifferentiableCollisions.jl.

Full credits are in [NOTICE.md](NOTICE.md) — please keep them if you fork or
redistribute this code, and cite the original works above.

**Status: work in progress.** This README and full build/usage docs will be
filled in as each engine lands; see the project's implementation plan for
the current phase.

## Building

**Recommended: clang (LLVM-mingw on Windows).** Measured on the SOCP solve
path (`tools/bench_ergodic.cpp`/`.jl`, 100k-pose ergodic sweep vs. the
original Julia library, §1c of [DEVIATIONS.md](DEVIATIONS.md)): the exact
same source, compiled with clang instead of GCC, is 1.1x-1.5x faster —
enough on its own to flip DCOL++ from slower-than-Julia to faster-than-Julia
on every shape pair tested. This is a codegen-quality gap in GCC's
optimizer for this code, not anything DCOL++-specific to work around; it's
simply the better choice here.

- Windows: `winget install MartinStorsjo.LLVM-MinGW.UCRT` (targets the same
  `x86_64-w64-mingw32`/`windows-gnu` ABI as MinGW-GCC — a drop-in compiler
  swap, not a platform change). Put its `bin/` directory on `PATH` (or pass
  `-DCMAKE_CXX_COMPILER=<path>/clang++.exe` below) *before* running your
  binaries too — they dynamically link `libunwind.dll`/`libc++.dll` from
  there.
- Linux/macOS: install `clang`/`clang++` via your usual package manager.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build
ctest --test-dir build
```

MinGW-GCC also works (just omit `-DCMAKE_CXX_COMPILER`) and is fully
supported as a fallback — CMakeLists.txt has no compiler-specific logic —
but expect the slowdown above.

Eigen3 is located via `find_package`; if none is found on your system,
CMake fetches it automatically.
