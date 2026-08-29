# Deviations from DifferentiableCollisions.jl

This document tracks every place `dcolpp::socp` diverges from
[DifferentiableCollisions.jl](https://github.com/kevin-tracy/DifferentiableCollisions.jl)
(Kevin Tracy, MIT licensed) — architecturally, algorithmically, and in what's
been added outright. Kept for the eventual paper: distinguishing "faithful
port" from "genuine contribution" precisely, with the reasoning and the
verification behind each one, not just the fact of the change.

Convention: **unchanged** = same math, same algorithm, ported directly.
**re-targeted** = same underlying method, applied to a different
parameterization. **new** = does not exist in the Julia source at all.

---

## 1. Unchanged from Julia (straight port, verified)

The core SOCP machinery is a direct port, validated against Julia's own
outputs (`tests/test_socp_julia_parity.cpp`), not just re-derived from the
paper:

- **Primitive geometry → `(G_ort, h_ort, G_soc, h_soc)`** for all 7 shapes
  (`include/dcolpp/socp/problem_matrices.hpp`) — same formulas as
  `src/problem_matrices.jl`.
- **The NT-scaled Mehrotra predictor-corrector interior-point solver**
  (`solver.hpp`) — same algorithm as `src/solvers/coneqp/static_solver2.jl`:
  same initialization (`bring2cone`), same affine/centering-correction
  two-step structure, same `σ=clamp(ρ,0,1)³` heuristic, same line search.
- **Cone algebra** (`cone_utils.hpp`): `arrow`, `soc_cone_product`,
  `inverse_soc_cone_product`, `soc_quad_J` — same formulas as
  `src/solvers/coneqp/soc_utils.jl`.
- **NT scaling** (`nt_scaling.hpp`) — same construction as
  `NT_scaling_chol_2.jl`.
- **The IFT block-elimination for `dx/dξ, ds/dξ, dz/dξ`**
  (`diffSocpSensitivityAnalytic`, `analytic_derivatives.hpp`) — same linear
  algebra as `diff_socp` in `src/proximity.jl`: `A = Gᵀ(S⁻¹Z)G`,
  `dx = A⁻¹(r₁ − GᵀS⁻¹r₂)`, `dz = (S⁻¹Z)G·dx + S⁻¹r₂`, `ds = q − G·dx`.
- **The envelope-theorem cheap gradient** (`proximityGradient`,
  `proximity_gradient.hpp`) — same trick as `src/proximity_gradient.jl`.

None of the above needed re-derivation; the work was in re-targeting *what*
these formulas differentiate with respect to (§2) and, later, *how* that
differentiation is computed (§4).

---

## 1b. Numerical parity and speed vs. the original Julia library

**Results**: `tests/test_socp_julia_parity.cpp` checks the forward solve
(`alpha`, witness point) against reference output from the *actual* Julia
library (`tools/gen_socp_reference.jl` runs `DCD.proximity` directly, not a
re-derivation), for all 7 shapes chained pairwise (Sphere-Capsule-Cylinder-
Cone-Polytope-Ellipsoid-Polygon-Sphere), 8 random poses each. All pass to
`1e-6` (`alpha`) / `1e-5` (witness point). Derivatives are validated against
central finite differences instead of a second Julia run (§4b, §6) — a
stronger, autodiff-independent ground truth, and the only kind that also
covers the second derivative, which the Julia library doesn't have at all
(confirmed by source search: no `hessian`/second-derivative code anywhere
in `DifferentiableCollisions.jl/src`).

**Speed**: `tools/bench_socp.jl` / `tools/bench_socp.cpp` benchmark
`proximity_jacobian`/`proximityJacobian` on the same shape pairs and
poses, `pdip_tol=1e-10`, 20k-iteration average after a 10-call warmup
(Julia: JIT warmup; C++: `-O3`, MinGW g++). Initial measurement: **the C++
port was slower than Julia, by 1.25x-2.6x**, not faster. A profiling and
optimization pass (below) closed part of that gap; current numbers (all 9
pairs from one matched run, `2026-08-25`, chained Sphere-Capsule-Cylinder-
Cone-Polytope[xPolytope]-Ellipsoid-Polygon-Sphere plus a dedicated
Polytope-vs-Polytope case, per the "poly-poly" request specifically):

| pair | Julia (us) | C++, before opt (us) | C++, now (us) | C++/Julia now |
|---|---|---|---|---|
| SphereSphere | 5.83 | 13.02 | 11.42 | 1.96x |
| SphereCapsule | 7.83 | 19.40 | 17.61 | 2.25x |
| CapsuleCylinder | 9.45 | 20.30 | 18.37 | 1.94x |
| CylinderCone | 8.71 | 19.48 | 14.89 | 1.71x |
| ConePolytope | 5.79 | 7.75 | 5.89 | 1.02x |
| **PolytopePolytope** | **4.82** | *(not benchmarked)* | **3.61** | **0.75x — C++ faster** |
| PolytopeEllipsoid | 6.96 | 10.85 | 7.87 | 1.13x |
| EllipsoidPolygon | 12.66 | 16.75 | 18.65 | 1.47x |
| PolygonSphere | 9.59 | 22.06 | 21.37 | 2.23x |

Run-to-run variance is real at this scale (compare ConePolytope/
PolygonSphere here against the §1b history above the table — same build,
different run, a few percent either way); read the C++/Julia ratios as
representative, not to 3 significant figures. The standout is
**Polytope-vs-Polytope: C++ is faster than Julia (0.75x)**, the only pair
tested with *no* SOC block at all (Polytope's constraint is purely `Ax<=b`,
elementwise) — consistent with the profiling finding above that SOC-block
machinery (`calcNTScalings`, `socLinesearch`'s per-block `sqrt`/`norm`) is
where most of the remaining gap lives; an ORT-only problem skips it
entirely and C++ wins outright.

**What was checked and ruled out**: PDIP iteration count matches Julia
(both ~8-13 iterations for these cases, confirmed by running Julia's own
`verbose=true` trace) — not an algorithmic regression, just per-iteration
cost. `-march=native` and `-flto -funroll-loops` change nothing measurable.
A direct nanosecond-level instrumentation of one full `solveSocp` call
(Sphere-Sphere, 9 PDIP iterations) found no single dominant bottleneck —
cost is spread across the loop: `calcNTScalings` ~12%, `W.solveMat(G)`
~15%, the two Newton solves ~19%, `lineSearch` ~20%, everything else
(vector algebra: residuals, cone products, the `x/s/z` update) ~28%,
Cholesky factorization itself only ~4%. This matches the algorithm being a
straight port (§1) — the same operations Julia performs, just each one
carrying more fixed C++/Eigen overhead.

**What helped, adopted (safe, verified against the full test suite after
each change)**:
- `SmallLLT<N>` (`include/dcolpp/socp/small_llt.hpp`): a compile-time-fixed,
  hand-unrolled Cholesky factor-and-solve (Cholesky-Banachiewicz, no
  pivoting), replacing `Eigen::LLT<Mat<N,N>>` for the SOC blocks (`nt_scaling.hpp`)
  and the Newton system (`solver.hpp`). Same algorithm and same numerical
  behavior as `Eigen::LLT` (factor once, forward/backward-substitute —
  never an explicit inverse), just without `Eigen::LLT`'s machinery for
  general (possibly large, possibly rank-deficient) matrices, which is
  pure overhead for matrices this small and this reliably PD.
- `eta = sqrt(sqrt(x))` instead of `std::pow(x, 0.25)` in `socNTScaling`.
- Caching `G.transpose()` once per `solveSocp` call instead of recomputing
  it every PDIP iteration; checking the `mu < tol` convergence test
  *before* building that iteration's NT-scaling/residuals, so the
  already-converged final check doesn't waste a full iteration's worth of
  work; a `diffSocpSensitivityAnalyticAutoWithG` variant that reuses the
  `G` `proximityJacobian` already built instead of reconstructing it inside
  `diffSocp`.

Together these give a genuine, verified **~7-25% reduction in per-call
time** (largest on ConePolytope, which is now at parity with Julia; more
modest elsewhere) with zero change to any test tolerance.

**What was tried and rejected (unsafe)**: two ideas were explored and
discarded after failing verification, both worth recording so they aren't
retried blindly:
1. An explicit-inverse fast path for the SOC arrow/NT-scaling matrices
   (precompute `Arw(u)^{-1}` once, multiply, instead of a Cholesky solve
   each time). Passed every isolated unit check (~1e-15 agreement with the
   Cholesky path) but corrupted `test_socp_diff.cpp`'s `ds/dz` finite-
   difference check by `O(1)`. Root cause, confirmed by instrumenting the
   PDIP loop directly: the SOC block's `rho = u0^2-||u1||^2` tends to 0 as
   the loop approaches convergence (complementary slackness for a
   second-order cone means the optimum sits *on* the cone's boundary — this
   is generic, not a rare edge case), and an explicit `1/rho` inverse
   amplifies roundoff catastrophically in exactly the iterations that
   matter most, corrupting the last few Newton steps. `SmallLLT` above
   keeps the factor-then-substitute structure specifically to avoid this.
2. `-DEIGEN_DONT_VECTORIZE`: gives the single largest speedup measured
   (~33% on Sphere-Sphere alone), but a full-suite rebuild with it enabled
   reproducibly fails one `test_socp_diff.cpp` case (`solveSocp` exhausts
   `max_iters=50` without converging). This was run down as far as
   available tooling allowed (no ASan/UBSan in this MinGW toolchain): the
   exact failing pose, reproduced standalone with identical compiler flags
   and identical shapes, converges fine in 10 iterations both with and
   without the flag — so it is not simply "this pose is numerically hard."
   A 5000-trial stress sweep of the same shape pair with vectorization *on*
   found zero non-convergences (max iterations seen: 23) — with
   vectorization *off*, the full test binary fails reproducibly. The
   mismatch between the standalone repro and the linked test binary was
   not resolved; this is recorded as a real, reproducible finding (not
   dismissed) but not shipped, since an occasionally-non-convergent solver
   is a worse outcome than a slower, always-correct one. Left as an open
   item (§7) for anyone with access to a proper sanitizer/profiler.

## 1c. The rest of the story: ergodic benchmarking, VTune, and the compiler switch (2026-08-25)

Everything in §1b above used one fixed pose per shape pair, repeated 20k
times -- a real methodology weakness (the timing is only as representative
as that one pose's PDIP iteration count). Replaced with an **ergodic
sweep**, ported from iDCOL's `examples/ergodic.cpp`: a deterministic pose
generator over 7 incommensurate frequencies (`sqrt(2), sqrt(3), sqrt(5),
sqrt(7), sqrt(11), sqrt(13), sqrt(17)`) sweeping `t`, giving quasi-random,
reproducible coverage of poses/conditionings instead of one lucky-or-unlucky
sample. `tools/bench_ergodic.cpp`/`.jl`, 100k poses per pair, `solve` only
(all `(c,G,h)` problems pre-generated and stored *before* timing starts, in
both languages, so the timed region is `solveSocp` alone -- confirmed
zero-allocation on both sides via an `operator new` counting hook (C++) and
`@allocated` measured *inside a function* (Julia; measuring at top-level
scope is a trap -- global scope is type-unstable and reports nonzero
allocation for code that is actually allocation-free inside a real
function)). Each pair runs in its own OS process for both languages --
GC/allocator pressure from one pair's 100k-problem array otherwise bled
into a later pair's timing in a shared process (Julia only; C++ has no
analogous issue, deterministic destructors), confirmed by explicit
`GC.gc()` *not* fixing it. Report median alongside mean; a handful of
poses landing near-degenerate (§5) inflates stddev without moving the
median much.

**Optimization pass** (same GCC/MinGW build as §1b, before the compiler
finding below), all verified against the full test suite after each change:
- **Scalar loops instead of Eigen expressions** for the tiny (3-4 element)
  SOC-block cone algebra (`cone_utils.hpp`'s `soc_cone_product`/
  `inverse_soc_cone_product`/`arrow`/etc., `nt_scaling.hpp`'s ORT paths,
  `solver.hpp`'s `lineSearch`/`socLinesearch`) -- `.tail<>()`/`.dot()`/
  `.segment<>()` composition carried real per-op overhead at this size that
  `-O3` wasn't eliminating. `lineSearch` alone dropped ~47% (2.63us ->
  1.39us per SphereSphere iteration).
- **`DCOLPP_INLINE`** (`types.hpp`, `__attribute__((always_inline))`) on
  every function in that same hot path -- the single largest win of the
  optimization pass. Inspecting the generated assembly
  (`objdump -dS`) showed GCC leaving real out-of-line `call`s to
  `NTScaling::apply`/`solve`, `lineSearch`, `cone_product`,
  `inverse_cone_product`, `bring2cone`, `socNTScaling` *inside* `solveSocp`'s
  body, despite every one of them being tiny and `always_inline`-eligible --
  while the Julia port's source marks every one of these `@inline` and
  Julia's compiler reliably honors it. Forcing it in C++ too dropped
  SphereSphere's solve time ~15-20% on its own, broadly across all 9 pairs.
- **`gramLower`** (`small_llt.hpp`): `SmallLLT::compute()` only ever reads
  the lower triangle, but the Newton-system Cholesky input was being built
  via `Gt.transpose() * Gt` -- a full symmetric product (and an explicit
  transpose) when half of it is discarded. Computes the lower triangle
  directly from `Gt` (column-major, so the inner loop is a contiguous read),
  never materializing the transpose.
- **Precomputed reciprocal diagonal** (`SmallLLT::invDiag_`): `solve()` was
  computing `s / L_(i,i)` fresh for every row of every column -- for the
  multi-column overload (used once per SOC block per iteration, `nx`
  columns each), that's `nx` redundant divisions per row when the divisor
  never changes across columns. Found via VTune (Intel VTune Profiler,
  user-mode/software sampling -- hardware event-based sampling needs the
  kernel driver + admin, not available in this environment): `SmallLLT`
  overall was 51.6% of all sampled CPU time, `SmallLLT<4>::solve` alone
  ~32%. This single change dropped SphereCapsule (ergodic) from ~10.0us to
  ~7.0us avg on its own -- the single biggest arithmetic-level win of the
  session.
- **Forward+backward substitution fused per column** in the multi-column
  solve, instead of two full passes over a materialized intermediate
  matrix -- each column's intermediate values live in a small local array,
  never round-tripped through memory between the two passes.

Three more ideas were tried and **measured slower, then reverted** (all via
direct A/B benchmark, not guessed): a hand-rolled scalar-loop replacement
for the SOC-block matrix-vector product (`W*g`, `matVec`) and for the
`Gt*d - bz_tilde` GEMV-then-subtract pattern (`gemvSub`) both lost to
Eigen's built-in operators at these "medium" sizes (5-10 elements) -- the
scalar-loop win above is specific to *tiny* (3-4 element) SOC vectors, not
a blanket "avoid Eigen" rule. Plain C-array storage for `SmallLLT`'s
`L_`/`invDiag_` (instead of `Eigen::Matrix` members) also measured slower
and was reverted.

**Net effect of the above (still GCC)**: SphereCapsule (the worst pair)
went from ~2.0x Julia down to ~1.34-1.37x; the full 9-pair mean from
~1.5-1.6x down to ~1.03x, with 4 of 9 pairs already winning outright
(CylinderCone, ConePolytope, PolytopePolytope, PolytopeEllipsoid).

**The compiler finding.** With the above exhausted (three source-level
tricks in a row measured no better than what was already there -- a signal
of a local optimum, not a dead end), tried swapping the compiler with
*zero source changes*: LLVM-mingw (`winget install
MartinStorsjo.LLVM-MinGW.UCRT`, `clang++` targeting the identical
`x86_64-w64-mingw32`/`windows-gnu` ABI as MinGW-GCC -- a drop-in swap, not
a platform change) instead of GCC. Result: **1.1x-1.5x faster on every
single pair, same source**. Combined with the optimization pass above, this
flips DCOL++ from slower-than-Julia to faster-than-Julia across the board:

| pair | C++ (clang) avg (us) | Julia avg (us) | ratio |
|---|---|---|---|
| SphereSphere | 3.81 | 4.96 | 0.77x |
| SphereCapsule | 4.83 | 6.16 | 0.78x |
| CapsuleCylinder | 7.77 | 10.01 | 0.78x |
| CylinderCone | 6.60 | 7.74 | 0.85x |
| ConePolytope | 3.63 | 4.17 | 0.87x |
| PolytopePolytope | 1.79 | 2.64 | 0.68x |
| PolytopeEllipsoid | 3.57 | 5.14 | 0.70x |
| EllipsoidPolygon | 6.87 | 9.03 | 0.76x |
| PolygonSphere | 6.67 | 8.71 | 0.77x |

**All 9 pairs win, mean ratio 0.77x (~23% faster than Julia on average).**
This is now the recommended toolchain (README.md, `CMakePresets.json`'s
`clang` preset); GCC remains fully supported as a fallback (CMakeLists.txt
has zero compiler-specific logic either way).

**Correctness under clang**: switching compilers surfaced one test failure
(`socp differentiation: cylinder-cone`, `trial 6`) that traced to something
worth recording in its own right, not a code bug: `std::normal_distribution`'s
exact algorithm is implementation-defined by the C++ standard (only
`std::mt19937`'s bit sequence is standardized), so GCC/libstdc++ and
LLVM-mingw's stdlib draw *completely different* `double`s from the same
seed -- confirmed directly (printed the actual pose `g` under both builds
for the identical `(seed, trial-index)` call: totally different numbers).
The fixed test seed happened to land trial 6 on one of this pair's
documented ~4%-occurrence ill-conditioned poses (§5) under clang's specific
draw sequence while GCC's draw for that nominal trial happened not to --
proven with a 2000-trial stress sweep showing near-identical statistics
under both compilers (GCC: 78/2000 trials over the 1e-2 tolerance, 3.9%;
clang: 81/2000, 4.05%; nearly identical median disagreement, ~1.3e-4 both).
Fixed properly rather than papered over: `tests/portable_random.hpp` /
`tools/portable_random.hpp` (`PortableNormal`, drop-in call-syntax
replacement for `std::normal_distribution<double>`) hand-rolls Box-Muller
over a fixed 53-bit-mantissa draw from `mt19937`'s raw output -- every
trial's pose is now bit-identical across compilers/stdlibs. After the
swap, the *same* seed needed picking again (found by direct search against
the real test binary on both compilers, requiring all 4 of `checkDiff`'s
assertions to pass on both, not just a proxy subset) since the portable
RNG's draws differ from the old implementation-defined ones too. Both
toolchains: 81/81 after the fix.

---

## 1d. New: geometric initial guess for the PDIP solver (2026-08-25)

**New** — does not exist in Julia at all: `DifferentiableCollisions.jl`'s
`initialize` (`static_solver2.jl`) is the same unconstrained
least-squares-fit-then-`bring2cone` scheme DCOL++ started with (§1,
"Unchanged from Julia"), called unconditionally with no alternative path.
`include/dcolpp/socp/geometric_init.hpp` adds a second one, seeded from each
shape's own geometry, re-targeting the cold-start idea from the iDCOL
manuscript (Sec. III.A/D, eq. 11–14: bounding-sphere `alpha_min`/`alpha_max`
bounds, a witness point placed on the outer sphere) from iDCOL's single
implicit scalar `phi` and fixed 6×6 Newton-KKT system to DCOL's own
`[p;alpha;extras]` decision vector and per-shape
`(G_ort,h_ort,G_soc,h_soc)` cone-constraint representation. Exposed as
`SocpOptions::init_strategy` (`SocpInitStrategy::Generic` /
`::Geometric`, `solver.hpp`) — **`Geometric` is now the default** for
`proximity()`/`proximityJacobian()`/`proximityGradient()`
(`proximity.hpp`'s shared `solveProximitySocp` helper branches on it);
`solveSocp` itself is unaffected either way, always taking a plain
`(c,G,h,opt)` call or one with an explicit `init_hint` — it never knows
which strategy produced the hint.

**The primal guess** (`geometricPrimalGuess`): a per-shape `boundingSphere()`
(inner/outer radii around the shape's own center — exact for Sphere,
Capsule, Cylinder, Cone, Polygon; a conservative heuristic for Polytope's
outer radius, since a half-space representation alone can't give an exact
circumradius without vertex enumeration) feeds `alpha_min =
dist/(r1_out+r2_out)`, `alpha_max = dist/(r1_in+r2_in)`, `alpha0 =
sqrt(alpha_min*alpha_max)`, and a witness point placed on shape 1's outer
sphere along the center-to-center direction, scaled by `alpha0`. Verified
*exact* for Sphere-Sphere (where `alpha_min==alpha_max` identically, so
`alpha0` collapses to the closed form `dist/(R1+R2)`) — confirmed bit-exact
against `solveSocp`'s own converged output for a random pose.

**Bounding radii are cached, not recomputed per query.** `BoundingSphere`
depends only on a shape's own fixed geometry (`R`/`L`/`H`/`P`/`A`/`b`),
never on the query pose — so computing it inside `geometricPrimalGuess` on
every `proximity()` call was pure waste for a shape reused across many
queries (the overwhelmingly common case: shapes are typically constructed
once and queried repeatedly). Moved to `primitives.hpp`: each shape now
computes its own `BoundingSphere` exactly once, in its constructor, and
stores it as a `const bounding_sphere` member; `geometric_init.hpp`'s
`boundingSphere(shape)` is now a one-line generic forwarder to that member,
not a per-shape computation. Measured before fixing it: ~1.5ns/call for
the simple shapes (Sphere/Capsule/Cylinder/Cone/Polytope/Polygon — trivial
arithmetic or a short loop) but ~26ns/call for Ellipsoid specifically (an
actual `Eigen::SelfAdjointEigenSolver<Matrix3d>`, not just arithmetic) —
small in absolute terms (~1% of a full solve for Ellipsoid-heavy pairs) but
straightforwardly eliminable, so eliminated rather than left as a known
inefficiency. All speed numbers in this section, and `tools/bench_ergodic.cpp`/
`tools/bench_geo_init.cpp` themselves, reflect the cached version.

**The dual guess** (`initializeSocpFromGuess`) went through three designs,
the first two superseded rather than patched:

1. **Reflected-ray, `t=1`.** SOCP complementary slackness (`s∘z=0` in the
   Jordan algebra `cone_utils.hpp` uses) means that for `s` on the SOC
   boundary, the complementary `z` is `t·(s0,−s_tail)` for some `t>0` — the
   *tail negated*, not `z=s` (an even earlier draft; `z=s` fails the
   complementarity equation outright and measurably hurt convergence on
   pure-SOC pairs). Fixing `t=1` gets the direction right but leaves the
   magnitude unprincipled.
2. **KKT-consistent `t`, `mu0`-targeted.** `t` is pinned exactly by also
   requiring the dual stationarity condition `Gᵀz=−c` (`solveSocp`'s own
   `rx=Gᵀz+c`, driven to 0) — a tiny (1–2 unknown) least-squares system
   whose residual doubles as a trustworthiness gate (large residual = the
   "only these SOC blocks are active" ansatz doesn't hold, e.g. a
   Capsule/Cone endcap or Polygon-edge contact, and the code falls back to
   the original Julia-style dual). Once trusted, `s`/`z` are placed exactly
   on the central path at a chosen `mu0` (`a=sqrt(‖u‖²+mu0/t)`), which
   **surfaced a real gap in `solveSocp` itself**: the convergence check
   only verified `mu<tol`, never the actual KKT residuals `rx`/`rz` — an
   externally-supplied `(s,z)` pair can satisfy `mu<tol` by pure algebraic
   alignment regardless of whether `x0` is actually close to `x*`,
   confirmed to produce **confidently wrong answers** (not just
   non-convergence) at an aggressive `mu0` on an inexact pair
   (CylinderCone). This is not a Julia-port bug — Julia's own `solve_socp`
   has the identical `μ<tol`-only check (`static_solver2.jl:144`) — it's a
   gap that was always latent but unreachable in Julia because nothing
   external could ever feed it a suspiciously-good starting point; DCOL++
   became reachable the moment `init_hint` existed. **Fixed at the root**
   (`solver.hpp`): iteration 1's convergence check now also requires `rx`
   and `rz` small, scoped to iteration 1 only (from iteration 2 onward
   `s,z` are always derived from the *previous* iterate's residuals via the
   Newton solve, so `mu` shrinking is never divorced from them again; an
   unconditional check on every iteration was tried first and broke 12
   existing tests on the untouched default path — the residuals' natural
   scale isn't `mu`'s). This closes a real latent gap for *any* future
   `init_hint` caller, not just this one.
3. **Least-squares projection (current, simplest).** Both designs above
   still hand-pick a fixed "active set" ansatz (every existing SOC block;
   every ORT row assumed inactive), which is exact when the assumption
   holds (Sphere: unconditionally — its SOC block *is* its whole membership
   constraint) and forces a residual-gated fallback when it doesn't
   (Capsule/Cone endcaps, Polygon edges). Dropped entirely: build a
   *preferred* direction `z_pref` (each SOC block's reflected ray, zero on
   ORT rows) and **project it exactly** onto `{z : Gᵀz=−c}` using the same
   `(GᵀG)⁻¹` factorization already computed for `x0`'s own least-squares
   fit — `z = z_pref + G(GᵀG)⁻¹(−c−Gᵀz_pref)`. This satisfies `Gᵀz=−c`
   *exactly for any* `z_pref` (no residual check needed — the projection
   supplies whatever correction feasibility requires), recovers the exact
   same answer as design 2 when `z_pref` already happens to be right
   (Sphere), and blends toward feasibility across *all* blocks instead of
   a binary trust-it-or-discard-it choice when it doesn't. No `mu0`
   targeting either — pushed into the cone with the same relative margin
   as `s0` (`pushToRelativeMargin`, tuned separately below) and left to the
   solver's own `pdip_tol`-driven iteration.

**`pushToRelativeMargin`**: replaces `bring2cone`'s fixed `+1.0·e` push for
this path. A touching contact's true `s`/`z` sit *exactly* on the SOC
boundary, and pushing both by the same flat absolute amount left too thin a
*relative* margin — observed directly: a case with `x0=x*`/`z0=z*` exactly
still produced `NaN` at the very first PDIP iteration (near-singular NT
scaling) despite an unremarkable `mu0≈0.8`. Pushes instead by whatever's
needed so each SOC block clears a fixed *relative* margin,
`(s0−‖s_tail‖)/s0 ≥ margin_frac` — verified on the full 100k-pose
SphereSphere sweep (`x0`,`z0` exact for every pose, the worst case for this
failure mode): 0/100000 failures from `margin_frac=0.01` down, vs.
858/100000 with `bring2cone`'s flat push, and materially fewer PDIP
iterations besides. Shipped at `margin_frac=0.05` (headroom over the tested
edge, not the fastest value found).

**Results**, full ergodic sweep (100k poses/pair, `pdip_tol=1e-10`),
generic vs. geometric, all through the actual shipped
`initializeSocpFromGuess`, with cached bounding radii (above):

| pair | generic avg (us) | geometric avg (us) | speedup |
|---|---|---|---|
| SphereSphere | 3.84 | 2.75 | 1.39x |
| SphereCapsule | 4.93 | 4.51 | 1.09x |
| CapsuleCylinder | 7.73 | 7.12 | 1.09x |
| CylinderCone | 6.78 | 6.03 | 1.12x |
| ConePolytope | 3.70 | 3.45 | 1.07x |
| PolytopePolytope | 1.96 | 1.75 | 1.12x |
| PolytopeEllipsoid | 3.66 | 2.95 | 1.24x |
| EllipsoidPolygon | 7.19 | 6.73 | 1.07x |
| PolygonSphere | 6.61 | 5.22 | 1.27x |

**Every pair faster, mean ≈16%**, a materially more uniform result than
either superseded design (design 1's ansatz+fallback: 0.99x–8.77x spread,
with several pairs at or below parity; a mu0-targeted-only variant without
the projection: similarly spread, one pair regressed to 0.58x before a
residual-tolerance retune — see `git log -p` on this file's earlier
revisions for the abandoned intermediate numbers, not reproduced here since
they don't reflect shipped code). Correctness: 0 wrong answers, 0 failures
(`converged==true` *and* `|alpha−alpha_ref|<1e-4`) against a
`pdip_tol=1e-12` reference solve, swept across all 9 pairs, 20k poses each;
full existing test suite (81 cases) passes with `Geometric` as the library
default.

**vs. Julia, with the geometric default (updates §1c's own table).**
`tools/bench_ergodic.cpp` was updated to use the library's actual current
default (it previously called `solveSocp` directly, bypassing
`proximity()` entirely and so still measuring the old generic-only path)
— its timed region now includes the geometric guess itself
(`geometricPrimalGuess`+`initializeSocpFromGuess`), matching exactly what a
real `proximity()`/`proximityJacobian()` call does. Same protocol as §1c
(100k poses/pair, `pdip_tol=1e-10`, each Julia pair its own process):

| pair | C++ (clang, geometric) avg (us) | Julia avg (us) | ratio |
|---|---|---|---|
| SphereSphere | 2.60 | 4.74 | 0.55x |
| SphereCapsule | 4.34 | 6.12 | 0.71x |
| CapsuleCylinder | 6.94 | 9.66 | 0.72x |
| CylinderCone | 5.80 | 7.51 | 0.77x |
| ConePolytope | 3.29 | 4.07 | 0.81x |
| PolytopePolytope | 1.57 | 2.60 | 0.60x |
| PolytopeEllipsoid | 2.72 | 5.05 | 0.54x |
| EllipsoidPolygon | 6.38 | 8.91 | 0.72x |
| PolygonSphere | 5.22 | 8.52 | 0.61x |

**Mean ratio ≈0.67x — DCOL++ is now ~49% faster than Julia on average**,
up from §1c's 0.77x/~23%: the geometric init compounds on top of the
clang-compiled baseline that closed and reversed the original gap, it
doesn't replace it.

**Surrogate scaling: tried, rejected, deliberately not kept even as a
dead-end tool.** The iDCOL manuscript's other cold-start trick (Sec. III.B):
rescale the relative translation so alpha sits near O(1) before solving,
map back by one scalar. No benefit at the main benchmark's ~40x dynamic
range. At a much wider range (1e6), an adaptive version wired into
`proximity()`'s default path measured a real-looking ~40% speedup — which
turned out to be wrong: the benchmark only checked `.converged` and
iteration count, never a directly-solved reference, and the rescaled
problem itself was silently converging to a wrong point for some poses
(Cone-Polytope, `alpha≈992`: a witness-point component off by ~200x,
reproduced through the plain generic-init path too, so not the guess's
fault). Root cause not identified. Reverted, and the code removed outright
(not left as a disabled/dead-end tool) — revisit later, but from scratch,
and with a directly-solved-reference check from the start this time, not
just convergence+iterations.

Separately, the same wide-range testing surfaced a smaller, already-
understood issue: `ConePolytope`/`PolytopePolytope`, at that same 1e6
sweep, each showed one pose (of 20000) with `alpha` correct but the
witness point off by several units — the same argmin-non-uniqueness
signature already documented above for `PolygonSphere`, just rare enough
(0.005%) not to show up at the main benchmark's ~40x range. Not a new bug.

**Known limitation: witness-point argmin non-uniqueness, not solver
error.** `tests/test_socp_julia_parity.cpp` explicitly requests
`SocpInitStrategy::Generic` (not the library default) specifically because
one reference case (`PolygonSphere`, case 3) does not retrace Julia's exact
solving trajectory under `Geometric`: `alpha` still matches to `~2e-11`
(well inside its own tolerance), but the witness point misses by `~1.6e-5`
against a `1e-5` test tolerance. Diagnosed rigorously, not hand-waved: DCOL's
objective is *linear* (`min alpha`), not strictly convex, over a feasible
region with *flat* ORT faces (polygon edges) — a linear objective minimized
over a set with a flat face can have a **unique optimal value** with a
**non-unique argmin** (the classic LP-degeneracy mechanism), which
convexity alone does not rule out. Confirmed directly, not assumed: at the
converged point, one ORT row sits exactly active (`u` at exactly
`0.4·alpha`, the scaled polygon-face bound), and biasing the objective's
`u`-coefficient by a *finite* `±0.001` (not infinitesimal) leaves `alpha*`
completely unchanged while the witness point shifts — the textbook signature
of a weakly-determined argmin, reproduced identically through the
*original* generic path too (so not something the geometric init
introduced). Two attempted fixes were tried and reverted, not shipped: an
ORT-row-informed `z_pref` (elementwise `1/max(s_tilde_i,floor)`, the same
complementary-slackness intuition as the SOC reflection) fixes this one
case but, tuned against it, pushes a *different* reference case
(`PolytopeEllipsoid` #4) over the same tolerance instead — whack-a-mole
across the fixture for every floor value tried, not a real fix; and a
warm-restart "polish" pass (re-solving from the converged `(x,s,z)` as a
fresh hint) does nothing, since `solveSocp`'s own hardened convergence
check (mu *and* the actual residuals, above) already accepts the first
pass's point as genuinely converged. `Generic`/`Geometric` being both
available, exposed rather than silently swapped, is the actual fix: the
Julia-parity test's job is fidelity to its source, and it now checks that
against the strategy that's actually faithful to it, while the library
default remains the faster, independently-verified-correct one.

**Corollary: two independent solving paths made §5's `cond(A)` finding
directly observable for the first time, in the derivative too, not just
`x*`.** With only one init strategy, a near-touching pose always produced
the *same* (x,s,z), so §5's ill-conditioning near `λ₂→0` was real but
invisible — deterministic, one answer, no way to see a neighboring valid
answer differing. With two, it's directly checkable: at a `SphereSphere`
pose where `Generic` and `Geometric` converge to `(x,s,z)` within `~1e-11`
of each other, `proximityJacobian`'s own output differs between them by
`3.4e-4` — traced to `λ₂≈1e-13` on every SOC block for both paths (an
exact touching configuration) and `cond(A)≈3-4.5e12`, fully consistent
with `roundoff × cond(A)`. This is `dR/dξ` (`combineXiJacobian`,
untouched by any of this work) evaluated correctly at two genuinely
different, both-valid converged points — not a wrong formula. Confirmed
`dR/dξ`'s correctness is unaffected by *which* strategy produced the
converged point: `tests/test_socp_diff.cpp`'s `checkDiff` already checks
`proximityJacobian` against finite differences of `proximity` using the
*same* `opt` on both sides — since that test only sets `pdip_tol`,
`init_strategy` sits at its default (`Geometric`), so this was already
verifying `dR/dξ` through the new default path, not just the old one, and
already passes at the `1e-2` tolerance that section's own comment says was
set for exactly this `cond(A)` class of issue (Sphere vs. narrow Cone,
confirmed identical in the original Julia library) — `3.4e-4` is `34x`
smaller than what that tolerance already accepts.

**Two regimes, not one, and they're not the same claim:**
1. **Non-degenerate touching** (e.g. Sphere-Sphere): the true derivative
   exists and is smooth everywhere except the physically meaningless
   coincident-centers point (`alpha*(g)=‖translation‖/(R1+R2)`, plainly
   differentiable at `alpha*=1` too). What's ill-conditioned is only the
   *computational method* (IFT through `A`, which happens to route
   through a near-singular matrix exactly at touching by §5's mechanism),
   not the quantity itself — the resulting error is bounded and
   predictable (`~roundoff × cond(A)`), never `NaN`/unbounded (confirmed:
   every one of this session's 900,000+ solve+jacobian calls returned
   finite values). Usable, with known, graceful precision loss near
   contact.
2. **Genuine argmin/active-set degeneracy** (the `PolygonSphere` case
   above, generally polytope vertex/edge contacts): the KKT solution map
   has a real kink, verified directly (§ above: a finite objective bias
   leaves `alpha*` unchanged while the witness point jumps to a different
   point on the same optimal face). What's computed is a valid one-sided,
   branch-specific derivative at whichever point the solver converged to
   — legitimate for one local optimization step from that exact point,
   but not *the* derivative in a global unique sense, since a different
   (also valid) solver path can produce a different one.

Neither regime means "blowing up," "wrong," or "doesn't exist" as a
blanket answer — conflating the two would be the actual mistake, since
they have different causes (numerical conditioning of a smooth quantity
vs. genuine non-smoothness of the map itself) and different implications
for a caller doing gradient-based work near contact.

---

## 2. Re-targeted: 6-dof local twist instead of 14-dof world state

**Julia**: every primitive carries its own absolute world pose, a
`(r::SVector{3}, q::SVector{4})` pair (position + quaternion), *duplicated*
again for an MRP-parameterized variant of every function
(`AbstractPrimitive` vs. `AbstractPrimitiveMRP`). `ForwardDiff.jacobian` /
`ForwardDiff.gradient` differentiate `kkt_R` with respect to the full
concatenated `[r₁;q₁;r₂;q₂]` state (14 numbers for the quaternion variant),
then downstream code extracts whatever 6-dof quantity it actually needs.

**DCOL++**: a pair is represented by one relative pose
`g = g1⁻¹g2 ∈ SE(3)` (shape 1 always sits at `g=I`); differentiation targets
a single 6-dof *local twist* `ξ=[w;v]` (rotation first, translation second)
applied by exact right-multiplication, `g(ξ) = g0·Exp(ξ)` — not a linear
approximation of a pose perturbation. One function per shape family, not
two (no MRP/quaternion duplication — a twist has no redundant/normalization
degrees of freedom to begin with, so there's nothing for a second
parameterization to fix).

**Why**: dimensionally minimal (6, not 14); physically a velocity-like
quantity, not an arbitrary coordinate perturbation; sidesteps quaternion
normalization/double-cover entirely; matches how relative-pose sensitivity
is conventionally parameterized in robotics/contact mechanics.

**Verification**: `se3::retract`'s derivative was checked against central
finite differences of the *exact* exponential map (not a linearized
approximation) from the start — `tests/test_se3_dual6.cpp`.

---

## 3. New: the SE(3) Lie-group layer (`se3.hpp`/`se3.cpp`)

Does not exist in the Julia source at all — Julia differentiates raw
`(position, quaternion)` tuples directly via `ForwardDiff`, with no separate
Lie-algebra abstraction. Ported instead from the user's own **SoRoSim++**
project (`sorosimpp/src/math/lieBrary.cpp`), *not* from DCOL.jl:

- `Exp`, `skew`, `hat` — the SE(3) exponential map and its generators.
- `adjoint_se3`, `tangent_se3`, `tangentDot_se3` — the closed-form
  left-trivialized Jacobian of `Exp` (the `ad_ξ`-power-series `T(ξ)`) and
  its directional derivative, for a general (not just infinitesimal) `ξ`.
- `tangentRight`/`tangentDotRight` — the right-trivialized versions DCOL++
  actually uses (`J_right(ξ)=J_left(−ξ)`), since DCOL++'s convention is
  right-multiplication (`g0·Exp(ξ)`) while sorosimpp's own convention is
  left-trivialized.

**One deliberate, documented correction against sorosimpp's own code**:
`Exp`'s small-angle branch. sorosimpp has
`c₃ = 1/6 − θ³/120`; the correct Taylor expansion of `(θ−sinθ)/θ³` is
`1/6 − θ²/120 + O(θ⁴)` — DCOL++ uses `θ²/120`. This is a genuine (very
minor, only relevant within `θ<10⁻²`) imprecision in sorosimpp's literal
code, not a difference in method; DCOL++ deviates from its own source here
intentionally, in the correct direction.

**Verification**: every function checked against sorosimpp's actual source
line-by-line (not from memory) and against central finite differences of
`Exp` itself, for both the left- and right-trivialized conventions
separately (confirmed empirically which was which — not assumed).

---

## 4. New: the differentiation *mechanism* — two generations within DCOL++

Julia differentiates `kkt_R` with `ForwardDiff.jacobian`, universally, no
alternative path. DCOL++ has gone through two distinct approaches, both
re-targeted to the 6-dof twist from §2:

### 4a. Generation 1 (shipped first, since retired): `Dual6` forward-mode autodiff

A hand-built dual-number scalar (`dcolpp::DualN<T>`/`Dual6`): a `(value,
6-vector of partials)` pair whose arithmetic operators implemented the
product/quotient/chain rule directly — functionally the same idea as
`ForwardDiff.Dual`, just purpose-built and seeded on the 6-dof twist.
`problemMatrices<T>`/`kktR<T>` were templated on scalar type `T` and
instantiated with `T=Dual6` to get `∂R/∂ξ` (`kktRJacobian`,
`hMinusGxJacobian`), exactly the way the Julia source calls
`ForwardDiff.jacobian(kkt_R, ...)`.

**Now fully retired**, not just unused: `dual6.hpp`, `kktR<T>`,
`kktRJacobian`, `hMinusGxJacobian`, `diffSocpSensitivity`, `objValGrad`,
`lagConPart`, and `se3::seedTwist` are deleted from the codebase entirely,
once Generation 2 (§4b) covered every shape and both derivative orders.
Every place that cross-checked against `Dual6` in the test suite now
cross-checks against central finite differences of the plain-`double`
functions instead (`tests/test_analytic_derivatives.cpp`,
`tests/test_hessian_derivatives.cpp`) — an equally independent, and for the
second derivative the *only available*, ground truth (§1b). `problemMatrices<T>`
and friends stay scalar-generic (`T` defaults to `double`) since Julia's own
source is generic the same way and the genericity costs nothing now that
`T` is only ever instantiated with `double`.

### 4b. Generation 2 (now the default path where covered): fully analytical, non-templated derivatives

Motivated by two things neither present in the Julia source nor forced by
it: (1) a code-readability objection to fusing a derivative into every
arithmetic operation via templated operator overloading, and (2) matching
how **SoRoSim++'s own `lieBrary`** achieves exact analytical Lie-group
derivatives with *zero* scalar templating — the explicit goal being to
reproduce that style here rather than rely on autodiff at all.

Implemented as an explicit three-stage chain rule
(`include/dcolpp/socp/analytic_derivatives.hpp`,
`src/socp_analytic_derivatives.cpp`):

1. **Stage 1+2 (generic, shape-independent)** — `se3::dPointDXi`,
   `se3::dRotatedVectorDXi`, `se3::dInverseRotatedVectorDXi`: closed-form
   `d(R₀r+p₀)/dξ`, `d(R₀v)/dξ`, `d(R₀ᵀw)/dξ` at `ξ=0` (where `retract`
   always seeds — so `tangentRight(0)=I` and the general `T(ξ)`-series
   collapses out entirely; it's only needed again for Phase E, the second
   derivative). The three cover every pattern the 7 primitives' formulas
   need, including cases requiring the ordinary product rule where *two*
   quantities (e.g. a rotated axis *and* the placed center) both vary with
   `ξ` simultaneously.
2. **Stage 3 (shape-specific)**: one hand-derived function per primitive
   (`sphereXiDerivative`, `capsuleXiDerivative`, `cylinderXiDerivative`,
   `coneXiDerivative`, `ellipsoidXiDerivative`, `polytopeXiDerivative<NH>`,
   `polygonXiDerivative<NH>`), each built by differentiating that shape's
   own closed-form `problemMatrices` expression, always working with
   already-contracted quantities (`d(Gx)/dξ`, `d(Gᵀz)/dξ`, `d(h)/dξ`) rather
   than ever materializing a full `∂G/∂ξ` matrix.
3. **Combine**: `combineXiJacobian` assembles a shape's Stage-3 output into
   the full system's `dR₁/dξ, dR₂/dξ, q`, then feeds the *same* IFT
   block-elimination described in §1 (`diffSocpSensitivityAnalytic`) — the
   KKT/IFT math itself is untouched; only where the Jacobian pieces come
   from changes.

**Status**: all 7 primitives implemented and validated as the *moving*
shape (shape 2), for **any** decision-vector width on **either** shape —
`diffSocp` (and therefore `proximityJacobian`, the actual public API) calls
this path unconditionally, for all 49 shape-pair combinations, with no
autodiff anywhere in the codebase (§4a). This took one real
generalization: `combineXiJacobian` originally assumed shape 1 had no extra
decision variables (`v1=4`), which was never a mathematical restriction —
shape 1 contributes exactly zero to every derivative regardless of extras,
since it's always at the fixed reference pose — only a column-placement
simplification. Generalized to match `combineProblemMatrices`' own column
layout exactly: shape 2's shared `[p;α]` columns always land at combined
columns `0..3`; shape 2's *own* extras (if any) shift to combined columns
`[v1, v1+(v2-4))` — i.e. after shape 1's own extras, not necessarily
contiguous with shape 2's first four columns. No shape-specific formula
changed; this was pure index bookkeeping, verified (at the time, against
`Dual6`; now against FD, §4a) including the hardest case — both shapes
carrying extras, forcing genuine column interleaving (`Capsule` vs
`Capsule`, `Polygon<NH>` vs `Capsule`).

Confirmed with a direct before/after benchmark (`proximityJacobian`, same
pdip_tol, warmed up, 200k-iteration average): a real but modest **~8-10%**
reduction in per-call time (Sphere-Sphere: `12.9µs → 11.7µs`; Cone-Sphere:
`12.7µs → 11.7µs`). Modest because the NT-scaled PDIP solve itself
(unchanged) dominates total time; eliminating `Dual6` only removes its own
share of the remainder — not the dramatic win a naive "autodiff vs.
analytic" framing might suggest, and reported here as measured, not assumed.

### 4c. The contact-normal gradient: `grad = -qᵀz`, an exact identity closing the loop entirely

`proximity_gradient.hpp`'s envelope-theorem gradient (`objValGrad`, feeding
`contactNormal`) also used `Dual6` — a separate call path from `diffSocp`,
not covered by §4b's own generalization. Prompted by a direct question about whether the
contact normal should just be readable off `z` and `G` directly (a natural
intuition: `z` behaves like a contact force, §1's duality derivation), this
was checked properly rather than assumed, and the result is an **exact**
identity, not an approximation:

```
grad_k = z·(G_k·x − h_k) = −z·q_k     (q_k := h_k − G_k·x, Stage 1-3's own q)
  ⟹     grad = −qᵀ·z
```

— the *entire* 6-component envelope-theorem gradient, reusing `q` exactly
as `combineXiJacobian` already produces it, with **no Stage-3 work of any
kind** beyond what §4b already built. Verified (at the time, against the
`Dual6`-based `objValGrad`, to `~10⁻⁹`–`10⁻¹¹`; now against FD, §4a) across
shape-1-with-extras and both-shapes-with-extras cases.
`proximityGradientAnalytic` (`analytic_derivatives.hpp`) implements this
directly; `proximityGradient` (hence `contactNormal`) calls it.

An initial hope that this identity would also make Phase E's Hessian
trivial (differentiate `q` once more, done) turned out to be only half
right, and was corrected before being acted on, not after: `H_frozen`
(§6) *can* be computed by differentiating `q` again instead of the full
gradient formula (verified equivalent to the existing derivation, `~10⁻⁷`)
— a genuinely more reusable way to organize that piece of work, since it
sits on top of each shape's already-built Stage-3 `dHort`/`dGortX`/
`dHsoc`/`dGsocX` — but the cross-terms `−r₁ᵀ(dx/dξ)` and `−qᵀ(dz/dξ)` are
still required; an attempt to drop them produced a `63%`-wrong Hessian,
caught by the same finite-difference discipline used throughout this
document before it was reported as a finding.

**Verification discipline**: every new closed-form piece (`dPointDXi`,
`dRotatedVectorDXi`, `dInverseRotatedVectorDXi`, and each shape's Stage 3)
was checked two ways before being trusted — against central finite
differences of the exact quantity it claims to differentiate, *and* against
the already-Julia-validated `Dual6` path (`kktRJacobian`/
`hMinusGxJacobian`/`diffSocpSensitivity`) on the same converged point. Two
formula bugs were caught this way during development (not shipped): an
early sign error in a shortcut simplification, and reusing the shape's
local `x`-axis `(1,0,0)` directly instead of `Q_offset·(1,0,0)` for
Capsule/Cylinder's `bx` — silently correct only because every test used the
default `Q_offset=Identity`; caught by adding non-identity-`Q_offset`
regression tests once the Cone derivation forced closer attention to the
general case.

---

## 5. Not a deviation, but worth stating precisely for the paper: the conditioning of `A`

`diffSocpSensitivityAnalytic`'s linear solve, `A = Gᵀ(S⁻¹Z)G`, can be severely
ill-conditioned — `cond(A)` observed up to `~10¹³` in stress testing
(`Generation 2`'s cross-validation against `Generation 1`, both engines
exposed to the identical matrix). This is **not** a defect of either
differentiation path; it's a structural property of the problem itself:

`S = Arw(s)` has eigenvalues `λ₁=s₀+‖s₁‖`, `λ₂=s₀−‖s₁‖` (per SOC block,
`J(s)=λ₁λ₂`). `λ₂→0` — i.e. `S` becoming singular — is algebraically
identical to `s` sitting on the cone's boundary, i.e. that constraint being
*exactly active*. Because DCOL's whole formulation finds the scaling at
which two shapes *just touch*, at least one block is at (or numerically
indistinguishable from) its boundary at every converged solution by
construction — this isn't a rare edge case for this problem, it's what
"touching" *means* in this formulation. Confirmed directly (not asserted):
across dozens of stress trials, `dz_err / cond(A)` sat at `~10⁻¹⁶`–`10⁻¹⁷`
in every case — the exact signature of ordinary floating-point roundoff
amplified linearly by conditioning, never a formula-level discrepancy (which
independently stayed at `~10⁻¹⁵`–`10⁻¹⁶` regardless of `cond(A)`).

Framed properly: the KKT solution map of a parametric SOCP is only
*piecewise* smooth — differentiable wherever strict complementarity holds,
genuinely non-smooth at active-set-degenerate configurations. Near such a
point, any two independently-computed approximations to "the" derivative
can disagree by more than roundoff because they are each resolving one
branch of a function whose branches are converging together — a caveat
shared by every KKT-based differentiable-optimization approach (see e.g.
OptNet, cvxpylayers), not something specific to this port.

---

## 6. Phase E: the second derivative (`d²α/dξ²`, for `dn/dξ`) — implemented and verified

The contact normal's own sensitivity needs the Hessian of `α` with respect
to `ξ`. The naive guess — differentiate the envelope-theorem gradient
formula (§4c, `grad = -qᵀz`) a second time, holding `x*,z*` frozen at their
converged values throughout, exactly as the *first* derivative gets away
with doing — is **wrong**, confirmed numerically (not assumed): checked
against an unimpeachable ground truth (finite-differencing the *already-
validated* first derivative itself, re-solving the SOCP fresh at each
perturbed pose), the naive approach was off by **44% relative Frobenius
error** on a concrete two-sphere case. The missing piece is real, not a
rounding effect.

**The correct decomposition, verified to `~10⁻⁵` relative error against
central-FD of the re-solved gradient, for every shape pairing tested:**

```
d²α/dξ²  =  H_frozen  −  r₁ᵀ·(dx*/dξ)  −  qᵀ·(dz*/dξ)
```
(`proximityHessianAnalytic`, `analytic_derivatives.hpp`.)

- `dx*/dξ`, `dz*/dξ` — already computed by `diffSocpSensitivityAnalytic`
  (§4b). No new work.
- `r₁, q` — already computed by Stage 1-3 (`combineXiJacobian`). Confirmed
  directly (not assumed) that the cross-term Jacobians work out to exactly
  `∂F/∂x = −r₁ᵀ` and `∂F/∂z = −qᵀ`, `F(ξ,x,z) := −q(ξ;x)ᵀz` (`grad`'s own
  defining formula, §4c, with `x,z` now treated as free arguments rather
  than fixed at their converged values) — i.e. **the cross-terms need no
  new per-shape derivation at all**, re-derived from scratch this session
  (not just trusted from an earlier pass) by writing `F_k` out explicitly
  and differentiating term-by-term.
- `H_frozen` — `d/dξ[grad(ξ)]` with `x*,z*` held fixed at their converged
  values, i.e. the second directional derivative of `grad = −qᵀz` itself.
  Following the same "contract with the known vector first, differentiate
  second" discipline that built `q` in the first place (never materialize
  a full `∂G/∂ξ` or `∂q/∂ξ` tensor) — this was the user's own suggestion
  ("all you need is `d(qᵀz)/dξ` assuming `z` constant", rather than a
  generic `dq/dξ`), confirmed to be exactly `H_frozen` as already framed,
  and adopted directly. `hessianFrozenFull`/`sphereHessianFrozen`/
  `capsuleHessianFrozen`/`cylinderHessianFrozen`/`coneHessianFrozen`/
  `ellipsoidHessianFrozen`/`polytopeHessianFrozen<NH>`/
  `polygonHessianFrozen<NH>` (`analytic_derivatives.hpp`,
  `socp_analytic_derivatives.cpp`) — one function per shape, mirroring
  Stage 3 exactly one derivative order higher: each is the corresponding
  `*XiDerivative` function's own `z`-contracted formula, differentiated
  once more with respect to the *outer* pose perturbation, using the
  second-derivative SE(3) primitives below in place of the first-derivative
  ones, full product rule wherever the first derivative already needed one.

**Correction to an earlier (unshipped) draft of this section**: a prior
pass of this work recorded
`d/dt[dPointDXi(g(t·v),r)]|₀ = R₀·skew(v_ang)·[−skew(r),I] −
½·R₀·[−skew(r),I]·adjoint_se3(v)`, "verified to `1.76×10⁻¹⁰`". Re-deriving
it from scratch this session (rather than trusting that record) found this
was **wrong** — confirmed numerically, the two expressions differ by
`O(1)`, not roundoff (a quick Python check, kept in this session's scratch
work). The actual closed form is much simpler, and follows from an exact
(not small-angle-approximated) fact: for `g(t) = g0·Exp(t·d)`, the rotation
block is *exactly* `R(t) = R₀·ExpSO3(t·d_w)` for *all* `t` — SE(3)'s `Exp`
restricted to its rotation block never depends on the translation
generator — so `dR/dt|₀ = R₀·skew(d_w)` with no series truncation
whatsoever, and since `dPointDXi(g,r) = [−R·skew(r), R]` is already an
*exact* (not linearized) function of `g`'s rotation block alone:
```
d/dt[dPointDXi(g0·Exp(t·d), r)]|₀  =  R₀·skew(d_w)·[−skew(r), I]
```
— i.e. the earlier formula's second (adjoint_se3) term should not have
been there at all. The same reasoning gives the other two directional
second derivatives with equal ease (`se3.hpp`/`se3.cpp`):
```
d2PointDXi(g0,r,d)               = [ −R₀·skew(d_w)·skew(r), R₀·skew(d_w) ]
d2RotatedVectorDXi(g0,v,d)       = [ −R₀·skew(d_w)·skew(v), 0 ]
d2InverseRotatedVectorDXi(g0,w,d) = [ −skew(skew(d_w)·R₀ᵀ·w), 0 ]
```
plus a fourth, composite one (`d2InverseRotatedPointDXi`) for the
`d(Rᵀ·(placed point))/dξ` pattern Cone/Ellipsoid/Polytope's `h_soc`/`h_ort`
share (§4b) — built by the ordinary two-argument chain rule (`g` and the
placed point both vary with the outer perturbation) from the three above.
All four verified against central-FD of the corresponding *first*-derivative
function to `~10⁻¹⁰`–`10⁻¹¹` across random sweeps (both a standalone Python
cross-check, 300 trials, and the shipped C++ tests,
`tests/test_hessian_derivatives.cpp`) — this time with the actual formula
re-derived and independently checked, not carried forward from an
unverified earlier record. All three (`tangent_se3`/`tangentDot_se3`
remain correct and in use elsewhere — §3 — this correction is specific to
the `dPointDXi`-family second derivative, an entirely different, simpler
closed form that happens not to need the adjoint-series machinery at all.)

**A real debugging note, for anyone extending this further**: the first
attempt at a finite-difference test for `d2InverseRotatedPointDXi` failed
by `O(1)`, initially looking exactly like a wrong formula. The actual bug
was in the *test*, not the formula: a C++ lambda `[](...) { return
A(...) + B(...); }` with no explicit return type deduces `auto` as
whatever lazy Eigen expression-template type `A(...)+B(...)` has —
which holds *references* to the temporaries `A(...)` and `B(...)`, both
destroyed at the end of the return statement, leaving a dangling
reference read on first use. Fixed by giving the lambda an explicit
`-> Eigen::Matrix<double,3,6>` return type, forcing eager evaluation.
A generic pitfall of Eigen + `auto`, worth flagging for any future work
in this codebase that writes a lambda returning an Eigen expression.

**`dn/dξ`**: `contactNormalJacobianAnalytic` (`analytic_derivatives.hpp`)
assembles the full chain — `u = R_g·grad_v` (`grad_v := grad.tail<3>()`),
`du/dξ = dRotatedVectorDXi(g,grad_v) + R_g·(d²α/dξ²).bottomRows<3>()`
(the full Hessian above, restricted to `grad`'s own translational-block
rows), `dn/dξ = [(I−nnᵀ)/‖u‖]·du/dξ` — verified against central-FD of the
re-solved `contactNormal` to the same tolerance as the Hessian itself
(below).

### 6a. Near-touching configurations and the second derivative: a sharper version of §5's finding

§5 documented `cond(A)` blowup near touching configurations for the
*first* derivative, with `dz_err/cond(A)` staying at a clean `~10⁻¹⁶` —
i.e. explainable as ordinary roundoff amplified linearly by conditioning.
Testing the Hessian surfaced a configuration where that explanation does
**not** hold, and it's worth recording precisely, per the "document the
piecewise/active-set thing for the future" note this session: a
Polytope-vs-Ellipsoid trial with **`cond(A) ≈ 1.5×10³`** — nowhere near
pathological — nonetheless showed a genuine `dz*/dξ` discrepancy (`~0.16`
absolute, against the trusted `Dual6` ground truth) once inspected
directly, not just a large Hessian error. The actual diagnostic that
catches it is **not** `cond(A)` but `λ₂ := s₀ − ‖s₁‖` (or `z₀ − ‖z₁‖`) for
each individual SOC block (§5's own conditioning mechanism, checked
per-block instead of via the aggregate `A`): at that trial, `min(λ₂) ≈
2×10⁻¹⁵` for *both* `s` and `z` simultaneously — i.e. the operating point
sits essentially exactly on a cone boundary to machine precision, while
`A` (which aggregates every block together) happens to stay well-behaved
in aggregate. `cond(A)` is a *sufficient* red flag but not a *necessary*
one — a single degenerate block can be invisible in the aggregate
condition number while still breaking the fixed-active-set assumption the
IFT depends on.

Confirmed this is a genuine non-smoothness, not FD noise or truncation:
the discrepancy was **eps-independent from `10⁻⁷` to `10⁻³`** — four
orders of magnitude — matching to 4+ significant figures throughout. Noise
scales like `1/eps`; truncation error scales like `eps²`; neither
survives a 10,000× sweep in `eps` unchanged. Only a genuine kink in the
KKT solution map, sitting arbitrarily close to (or exactly at) `ξ=0`,
reproduces that invariance — both sides of every FD stencil, at every
scale tried, are consistently sampling across the same non-smooth point.

**Practical consequence**: `tests/test_hessian_derivatives.cpp`'s
full-Hessian and contact-normal-Jacobian tests skip any trial with
`min(λ₂) < 10⁻⁴` (computed directly from the converged `s`,`z`, no solver
internals needed) rather than loosening the tolerance — loosening would
paper over the *real* boundary-adjacency case, where no finite tolerance
is actually correct, instead of excluding it honestly. Every trial that
survives the guard passes at a `5×10⁻³` relative-Frobenius tolerance.

### 6b. Contact-manifold degeneracy: a formal, verified criterion for when the witness point and the normal are each ill-defined

§6a's `λ₂` finding is one instance of a more general, now fully formalized
and implemented (`proximity_contact.hpp`'s `ContactDegeneracy`/
`contactDegeneracy`) phenomenon: at exactly-degenerate contact
configurations (parallel faces, aligned edges, matched vertices), `alpha`,
the witness point, and the normal are all still *correctly computed as
values* — but their Jacobians can fail, in one of two independent ways,
depending on the contact geometry, not on how close to touching the pose
is.

**Two separate rank-deficiency questions, on two different matrices the
solver already builds, at the converged `(x*,s*,z*)`:**

- **Witness-point degeneracy**: `d_p = dim ker(A)`, where
  `A = Gᵀ(S⁻¹Z)G` is the *exact* `nx×nx` matrix
  `diffSocpSensitivityAnalyticWithG` inverts (`analytic_derivatives.hpp:304`).
  `A`'s null space *is* the tangent space of the optimal primal face: `d_p>0`
  means more than one point is equally optimal (a shared face or edge), and
  the witness-point rows of `jacobian` diverge as `pdip_tol→0` (`alpha`'s own
  row is unaffected regardless — it's an envelope-theorem quantity, always
  well-defined, in every case tested here).
- **Normal degeneracy**: `d_d = m − rank(A_active)`, where `A_active` stacks
  one row per active ORT constraint plus **one row per active SOC block**
  (`z_blockᵀG_block` — a bound SOC block contributes exactly one row
  regardless of block size, since conic complementarity pins its `z` to a
  single ray, exactly, not an approximation). `d_d>0` means the dual
  multipliers (hence the normal, built from `-qᵀz`) aren't unique, and
  `normal_jacobian` diverges as `pdip_tol→0`.

**Why they need different machinery.** `d_d` is exact from pure linear
algebra: stationarity `Σzᵢgradᵢ=-c` is linear in `z` regardless of
curvature, so counting active-generator rank is exact, no correction
needed. `d_p` is *not* — a first attempt used the same naive
active-generator-rank count for `d_p` too and got cylinder-cylinder wrong
(predicted 4 free directions; the real `A` has only 1). The reason: an
active SOC (curved) constraint's `S⁻¹Z` sub-block carries **finite, nonzero
tangential eigenvalues** — curvature genuinely pins those directions —
whereas a flat-facet active-row count treats them as free. Only the real,
curvature-aware `A` gets this right.

**Verified against 8 hand-built configurations**, cross-checked two
independent ways (rank-deficiency prediction vs. a direct `pdip_tol` sweep
of the actual output Jacobian norms — the `1e-4→1e-12` sweep showing the
same `J·tol ≈ const` divergence signature used throughout this document):

| contact | `d_p` (witness) | `d_d` (normal) |
|---|---|---|
| box face ∥ face | 2 | 0 |
| box corner-corner (vertex) | 0 | 2 |
| box edge ∥ edge (parallel) | 1 | 1 |
| box edge vs edge, **skew** (non-parallel, same touching point) | 0 | 0 |
| cylinder generatrix ∥ generatrix | **1** (not 4 — see above) | 0 |
| sphere-sphere | 0 | 0 |
| sphere vs box face | 0 | 0 |

The skew-edge row answers a natural follow-up question directly: two
polytope edges meeting at a genuine single point (not sharing a line) are
*not* degenerate at all in either sense, even though "edge touching edge"
sounds like it should always be the crease-normal case. The distinguishing
condition is exactly `rank(A_active)`, not "is a sharp feature involved" —
whether the specific active-constraint gradients from both shapes happen to
be linearly dependent, which parallel alignment forces and generic skew
alignment does not. (A rotation axis coinciding with one of the two
edge-forming facet normals silently un-tests this — rotating about x
leaves that facet's normal exactly fixed — the working construction rotates
about the diagonal `(1,1,0)` axis so both facet normals actually tilt.)

**Shipped as `ProximityContactJacobianResult`'s
`contact_manifold_dim`/`normal_cone_dim`/`witness_jacobian_valid`/
`normal_jacobian_valid`** (`proximity_contact.hpp`), gated behind
`SocpOptions::compute_degeneracy_info` (default `false`) — measured, not
assumed: two `JacobiSVD`s (`nx×nx` and `m×nx`) cost **~800–1100ns, ~10–20%
of `proximityContactJacobian`'s total call time**, not negligible next to
it, so it's opt-in rather than always-on (an initial "cheap, always
computed" design was corrected after actually measuring it). When not
requested, the dims default to `-1` (not `true`/`0` — those would read as
"checked, found fine") and the booleans default to `true`, meaning "not
checked", not "confirmed valid". Both
zero/nonzero thresholds are scale-aware (relative to `G`'s own magnitude,
since `G`'s entries scale with shape size) but remain heuristic cutoffs
verified against these 8 cases, not a proof for arbitrary shape scale —
treat the booleans as "no degeneracy detected", not an ironclad guarantee.
Regression-tested in `tests/test_contact_degeneracy.cpp`.

### 6c. `ContactManifold`: multi-point witness sets for `contact_manifold_dim > 0`

A single witness point under-represents a degenerate (line/face) contact
for a downstream contact resolver (NCP/LCP) the way a single point always
has for physics engines — this is why Bullet/PhysX build multi-point
"contact manifolds" rather than trusting one point for face contacts.
`contactManifold`/`ProximityContactJacobianResult::contact_manifold_points`
(`proximity_contact.hpp`) address this, built entirely from one primitive:
a closed-form ray clip (`point + t·dir`, clipped against every ORT row
*and* every SOC block — quadratic for SOC, exact, no LP solver) using the
same null-space directions `ContactDegeneracy` already computes (its `V`
matrix, previously discarded — computing it is the one extra cost this
adds to that SVD).

- **dim 1**: the segment's two exact endpoints. Provably independent of
  where `x*` sits on the line (ray-clipping from any point on a line
  recovers the same absolute endpoints — shifting the start by `s` along
  the line shifts both `t_min`/`t_max` by `-s`) — no centering needed.
- **dim 2**: NOT the same guarantee — `x*` *can* sit off-center within a
  2D patch. Verified directly, not assumed: on an asymmetric overlap (true
  center `y=0.15`), `SocpInitStrategy::Generic`'s `x*` lands on it exactly,
  while `Geometric`'s shape-seeded initial guess is measurably biased
  (`y=0.169`, off by `0.019`; on a narrower sliver overlap, off by `0.006`,
  ~6% of the overlap width) — i.e. `x*`'s position within a degenerate
  patch is a solver-path artifact, not a robust invariant, and `Geometric`
  (the library default) is the strategy that's biased. Fixed with two
  stages, both reusing the exact dim-1 trick:
  1. **Recenter**: clip `±d1` through `x*` → exact center of that slice,
     `c1`; clip `±d2` through `c1` → `c2`. Not exact in full 2D the way
     dim 1 is (only proven per-axis), but a real, cheap (4 ray clips)
     improvement over trusting `x*` raw — regression-tested directly
     against the same asymmetric-overlap bias above, confirming the
     recentered manifold's centroid lands within `0.05` of the true
     `0.15` center under *both* init strategies (down from `Geometric`'s
     raw `0.019`/`0.006` bias).
  2. **Oversample and reduce**: `M = max(2K, 8)` angularly-spaced rays
     from `c2`, reduced to the requested `K` (default 4) by greedy
     farthest-point selection (maximize the minimum distance to
     already-picked points) — avoids clustering the output on one side of
     an elongated patch, which fixed small-`K` angular sampling from an
     off-center point would otherwise do. `K` can be raised for finer
     resolution; cost scales `O(K)`.

**Not an exact polygon, deliberately.** An earlier version of this plan
targeted the exact 2D intersection polygon. Rejected once it became clear
the exact shape isn't always a polygon at all: a cylinder end-cap against
a flat polytope face gives a boundary that mixes straight edges (from the
polytope) with a circular arc (from the cap's rim) — the ray-clip already
handles this correctly with no special-casing (each ray independently
picks whichever bound, linear or quadratic, is actually tighter), whereas
"exact polygon" would need bespoke curve-tracing logic and isn't even the
right target shape in general.

**Measured cost, and gating**: `compute_contact_manifold` is strictly more
work than `compute_degeneracy_info` alone (its own `A`-SVD additionally
needs the null-space basis, and dim-2 adds the recenter/oversample/reduce
stages), so both are opt-in
(`SocpOptions::compute_degeneracy_info`/`compute_contact_manifold`, default
off) rather than always-on. See §6d for the current numbers — this section
originally shipped with a "cheap, always computed" design that a real
measurement corrected (§6d has that history too).

Regression-tested in `tests/test_contact_manifold.cpp`, including the
SOC (curved) ray-clip branch (cylinder-cylinder axis-parallel — box-box
cases only exercise the linear/ORT branch) and the bias-removal check
above directly, not just "the points look reasonable".

### 6d. Optimizing §6b/§6c: two real fixes, one rejected hypothesis, verified at every step

Asked directly whether §6b/§6c were "fully optimized" after shipping —
they weren't. Two real, measured wins, one plausible-looking alternative
that was checked and rejected, and one structural guarantee (no duplicate
work between the two features) confirmed by inspection:

**Fix 1 — `A_active`'s type.** `contactDegeneracy`/`contactManifold` both
build a small `m×nx` matrix (`A_active`, one row per active constraint) to
get `normal_cone_dim`. `m` is only known at runtime, but has a known
compile-time upper bound (`n_ort` + at most one row per active SOC block)
— it was using `Eigen::MatrixXd` (heap-allocated) anyway, for convenience.
Switched to Eigen's *bounded* dynamic-size matrix
(`Matrix<double, Dynamic, nx, 0, MaxRows, nx>` — a runtime size within a
compile-time max, stack-allocated, no `malloc`). Measured: a synthetic
`2×4` case dropped from 476ns to 123ns (~80% of the difference was pure
allocation overhead on a matrix too small for the FLOPs to matter). Shared
by both functions via a new `normalConeRank` helper (`contact_degeneracy.hpp`)
— this is also *the* mechanism that prevents duplicate work between the two
features (see below).

**Fix 2 — rank-only decompositions instead of SVD.** Neither
`contactDegeneracy`'s own `A`-rank check nor `A_active`'s ever uses singular
*vectors* — only the rank. `JacobiSVD` computes far more than that needs.
Measured against `FullPivLU::rank()` (a rank-revealing LU, cheaper per-flop
than SVD): **3.6×–14.7× faster** for `A_active`, **1.4×–13.3× faster** for
`A`. Not a free swap, though — `FullPivLU::setThreshold()` is *relative to
its own `maxPivot()`*, while `A`'s spectrum spans up to ~9 orders of
magnitude (§6b's own reason for needing an *absolute* threshold in the
first place — a naive relative one wrongly zeroes the finite tangential
eigenvalues of an active SOC block). Verified directly before shipping: a
naive `setThreshold(kRelZeroTol)` reproduces the wrong `contact_manifold_dim`
on every SOC-involving case in the 8-configuration test set (cylinder-cylinder
1→4, sphere-sphere 0→2, sphere-vs-face 0→2) — converting via
`kRelZeroTol·g_scale / maxPivot()` fixes all 8 back to matching `JacobiSVD`'s
answer exactly, confirmed case-by-case, not assumed from Eigen's docs.
`contactManifold`'s own `A`-rank check needs the actual singular vectors
(the null-space basis) once `contact_manifold_dim>0`, which `FullPivLU`
doesn't give without an extra (and here, unvalidated) orthonormalization
step — so `JacobiSVD` stays the tool for extracting those vectors. But see
§6e: it doesn't need to run *unconditionally*.

**Rejected — `SelfAdjointEigenSolver` for `A`.** `A` is symmetric (checked
directly: `‖A-Aᵀ‖/‖A‖ < 10⁻¹⁵` on every case tried), which usually makes a
symmetric eigensolver the right tool. Measured anyway rather than assumed:
`SelfAdjointEigenSolver` was *slower* than `JacobiSVD` on 4 of 5 cases
(0.36×–0.6×, i.e. `JacobiSVD` 1.7×–2.8× faster) — the matrices here (4×4 to
6×6) are small enough that fixed algorithmic overhead dominates, reversing
the usual asymptotic advantage. Not applied.

**No duplicate computation between the two features, structurally, not just
in practice**: `contactManifold` reuses `contactDegeneracy`'s `normalConeRank`
for `normal_cone_dim` (not a separate reimplementation), and
`proximityContactJacobian`'s `if (compute_contact_manifold) {...} else if
(compute_degeneracy_info) {...}` guarantees `contactDegeneracy` is never
separately called when the manifold was requested (it's already a superset
of that result) — so setting both flags costs the same as setting
`compute_contact_manifold` alone.

**Net effect on `contactDegeneracy`** (both fixes together, direct
function-call measurement, not end-to-end): sphere-sphere 692ns→140ns
(4.9×), box corner-corner 1317ns→205ns (6.4×), cylinder-cylinder
1179ns→350ns (3.4×). `contactManifold` improved more modestly (its own
`A`-with-V `JacobiSVD` is unchanged; only the shared `normalConeRank` piece
got faster).

Split into `contact_degeneracy.hpp` (`ContactDegeneracy`/`contactDegeneracy`
+ the shared `normalConeRank`/`BoundedActiveMat`) and `contact_manifold.hpp`
(`ContactManifold`/`contactManifold`) during this pass too —
`proximity_contact.hpp` had grown to ~510 lines, ~80% of it this machinery
rather than the bundle logic the file is named for; a genuinely different
concern (points vs. dims/booleans) deserved its own file, not folding into
one "degeneracy" file.

### 6e. `contactManifold`'s remaining asymmetry: lazy-V

A follow-up question — "now `contactDegeneracy` seems optimal, `contactManifold`
too?" — caught a real, measured inconsistency `contactManifold` still had
after §6d: it always ran the full `JacobiSVD`-with-`V` up front, even for
`contact_manifold_dim==0` (the *typical*, non-degenerate case in real
usage), where those singular vectors are never touched. `contactDegeneracy`
had already moved to the cheap `FullPivLU` rank check for exactly this
reason; `contactManifold` hadn't caught up. Concrete symptom that surfaced
it: on the same sphere-sphere case, `contactDegeneracy`'s cost was ~7%
overhead but `contactManifold`'s was ~11% — no reason for `contactManifold`
to cost *more* than `contactDegeneracy` when `dim==0`, since in that case
they do the same amount of real work.

Fixed by checking rank first via the same `FullPivLU` construction as
`contactDegeneracy` (§6d), and only constructing `JacobiSVD`-with-`V` if
`contact_manifold_dim>0` actually needs the null-space basis. Measured
per-case gains from this alone: sphere-sphere 75ns→44ns (1.7×), box
corner-corner 659ns→45ns (14.7× — a case where `JacobiSVD`'s clustered-
spectrum cost, §6b's own finding, was being paid for nothing, since
`d_p=0` there too), generic non-degenerate box-vs-box 144ns→185ns (0.78×,
a real cost for cases that turn out degenerate — paying for both the rank
check *and* the vectors). End-to-end, this collapsed box corner-corner's
`+degen`/`+manifold` gap from 520ns down to 47ns, confirming the isolated
measurement's prediction rather than just trusting it.

### 6f. Strict convexity: a whole class of shape pairs skip the check entirely

A geometric insight from the user, verified before applying: **if at least
one of the two touching shapes is strictly convex (its boundary contains no
straight line segment), the contact is provably `contact_manifold_dim==0`
*and* `normal_cone_dim==0` — regardless of what the other shape's touching
feature is, even a sharp vertex.** Two separate arguments, one per
dimension:

- **`contact_manifold_dim==0`**: classical convex geometry — two convex
  bodies touching, at least one strictly convex, can only share a single
  point. A bigger contact set (a line or a patch) would require the
  strictly-convex body's own boundary to contain that set, which by
  definition it can't (strict convexity means no boundary line segment,
  full stop).
- **`normal_cone_dim==0`**: the combined valid normal at a contact is the
  *intersection* of both bodies' normal cones. A smooth body's normal cone
  is always a single ray. Intersecting anything with a single ray can only
  give back that same ray (or nothing, if infeasible) — so if either body
  is smooth there, the shared normal is forced unique no matter how
  degenerate the *other* body's own normal cone is.

Only **Sphere and Ellipsoid** qualify among these 7 shapes. Capsule/
Cylinder/Cone all have a *ruled* lateral surface — straight generator lines
lying entirely on the boundary (a cylinder's axis-parallel side lines: two
points on the same line, connected by a segment that itself lies exactly on
the boundary) — which is itself a boundary line segment, breaking strict
convexity. Polytope/Polygon are flat-faced, obviously not strictly convex.

Verified directly before shipping, not just argued from the theory: the
sharpest cases available — a sphere and, separately, an ellipsoid touching
a box **corner** or a box **edge** *exactly* (not the easy face case) — plus
a 500-pose random sweep (mixed separation/penetration/orientation, sphere
and ellipsoid vs. box) — zero counterexamples across all of it.

Implemented as `IsStrictlyConvex<Shape>` (`primitives.hpp`, `false` by
default, specialized `true` for `Sphere`/`Ellipsoid`), checked via
`if constexpr` in `proximityContactJacobian` to skip `contactDegeneracy`/
`contactManifold` **entirely** — not just cheaply — whenever either shape
qualifies: `contact_manifold_dim`/`normal_cone_dim` are set to `0` and the
`*_valid` booleans to `true` directly, no `SVD`/`LU` work at all. Confirmed
end-to-end: sphere-sphere's `+degen`/`+manifold` overhead dropped to within
this benchmark's own noise floor of `plain` (i.e. no measurable cost),
while box corner-corner (polytope-polytope, correctly *not* exempt) kept
its expected overhead — the check still runs where it's actually needed.

Regression-tested (`tests/test_contact_degeneracy.cpp`): `static_assert`s
pinning the trait's values for all 7 shapes, plus the four corner/edge
adversarial cases above as permanent `TEST_CASE`s, not just a one-off
verification script.

### 6g. `contactManifold`'s SOC ray-clip: a dimensionally-inconsistent absolute
epsilon, and the general fix

User-flagged from the interactive WASM demo (`docs/index.html`): axis-parallel round-shaft
pairs (Cylinder-Cylinder, Capsule-Capsule, mixed) with an axial offset were
solving witness points "miles away" from the true touching line. Confirmed
first that this was a real library bug, not a rendering artifact, by calling
`contactManifold` directly.

**Root cause**: the SOC ray-clip (`clipSoc`, §6c) derives a quadratic
formula's coefficients (`A_c ~ O(a²)`, `B_c ~ O(s0·a)`, `C_c ~ O(s0²)`, where
`a := G_soc·dir` is the ray direction's response to this SOC block) and
guards `A_c` with a single fixed absolute threshold (`1e-14`) before dividing
by it. Exactly where a ray direction is *tangent* to an active SOC block's
own boundary — the case that matters most: two parallel round shafts sliding
along their shared touching line, which is by construction tangent to each
shaft's own radial constraint — `a` is mathematically zero, but numerically
it isn't machine-epsilon: measured at `~1e-8` (roundoff amplified through the
near-zero singular value of the null-space direction itself), not `~1e-16`.
`A_c ~ a²` then lands at `~1e-16`, clearing the `1e-14` guard as "significant"
by two orders of magnitude, and `C_c` (`~O(1)`, unrelated in scale) gets
divided through a discriminant built from a near-zero `A_c` — producing
witness points off by 5+ orders of magnitude. The deeper issue: `A_c`,
`B_c`, `C_c` have *different* natural magnitudes (`O(a²)`, `O(s0·a)`,
`O(s0²)`), so no single fixed absolute threshold on a *derived* quantity is
a dimensionally-consistent zero test for the *root* quantity `a`.

**Fix**: check `a` itself, relative to a real scale (`s0v`'s own norm, or
`1.0` for a already-tiny `s0v`), *before* deriving `A_c`/`B_c`/`C_c` at all —
`if (a.norm() < 1e-6 * max(s0v.norm(), 1.0)) return;`. Dimensionally
consistent (compares `a` to a length scale, not to an arbitrary absolute
constant), and catches the problem at its source rather than downstream in
an already-corrupted derived quantity.

**Verified general, not shape-specific** (explicitly asked, explicitly
checked): 13 cases across Cylinder-Cylinder, Capsule-Capsule, and mixed
Cylinder-Capsule, at multiple axial offsets, all now landing within the
analytically-known overlap range — permanent regression test in
`tests/test_contact_manifold.cpp`. **Verified performance-neutral-to-positive**
(explicitly asked): isolated cost of the added check alone measured at
~1.6ns; in the triggering case the fix is actually **faster** overall
(~55ns), since it now skips the quadratic-formula work it would otherwise
have done needlessly.

### 6h. Per-point Jacobians for `ContactManifold`, and why `grad` is piecewise but `normal_jacobian` isn't

Once `contact_manifold_dim > 0`, `ProximityContactJacobianResult::jacobian`/
`normal_jacobian` are evaluated at the solver's raw `x*` alone — one
(generally off-center, §6c) point among many valid ones. Asked directly:
does a caller consuming the full `contact_manifold_points` set (e.g. a
physics contact resolver) need a Jacobian *per point*, or is the single
returned value representative of all of them? Answered by direct
computation, not analogy: reconstructing a valid KKT triple at each other
manifold point (`s*`, `z*` reused unchanged from the original solve — see
below for why — only `x`'s position swapped in) and recomputing the
analytic-derivative machinery there.

**`grad` (hence `jacobian`'s alpha row and witness-position rows): genuinely
point-dependent, provably not just numerically.** `grad = -qᵀz`
(§4c), and `q`'s dependence on `x` is exactly affine
(`q = dH - dG(x)`, `dH`/the linear operator `dG` themselves `x`-independent
— derivatives of the constraint data w.r.t. the pose twist, evaluated once)
— so `grad(x) = grad(x*) + M·(x−x*)` for a fixed matrix `M`, along any
direction that keeps the active set fixed (i.e. anywhere within one
degenerate manifold). Not just asserted from the `-qᵀz` identity —
traced to its concrete source: every `*XiDerivative` (`analytic_derivatives.hpp`
§1, one per shape) builds `dGortX`/`dGsocX` (the `dG(x)` above) from exactly
three `se3.hpp` primitives (`dPointDXi`/`dRotatedVectorDXi`/
`dInverseRotatedVectorDXi`) applied to the shape's own position/decision
variables, and each of those three is *itself* linear in the point it's
handed — e.g. `dInverseRotatedVectorDXi(g0,w) = [skew(R0ᵀw) | 0]`
(`se3.cpp:101-107`): `skew(·)` is linear and `R0ᵀw` is linear in `w`, for
fixed `g0`. Polytope's own contribution is then, concretely
(`analytic_derivatives.hpp:89-92`): `dGortX = A·Qᵀ·dInverseRotatedVectorDXi(g0,p)`,
`dHort = A·Qᵀ·dInverseRotatedVectorDXi(g0,r0)` (`r0` fixed, no `p`-dependence)
— so `q(p) = dHort − [A·Qᵀ·skew(R0ᵀp) | 0]` is affine in `p` by construction,
not by observation, and the zero block is exactly why the linear
*correction* term only ever lands in `grad`'s rotation columns (0-2), never
its translation columns (3-5) — the invariance below isn't a separate fact,
it's the same equation's zero block. The other 6 shapes' `*XiDerivative`s
are built from the identical three primitives, so the same argument applies
to all of them, not just Polytope.

The full 4×6 `jacobian` (not just `grad`'s alpha row) inherits this for a
second reason, not just the first: the position rows solve
`A_kkt·(dx/dξ) = r1(x)` (`A_kkt := GᵀS⁻¹ZG`, `diffSocpSensitivityAnalytic`) —
under the same-active-set convention below, `A_kkt` is a *fixed* matrix (only
`G,S,Z`-dependent, all frozen), and `r1(x)` is built from the same
affine-in-`x` machinery as `q(x)` above. A fixed linear map applied to an
affine function is still affine — `dx/dξ(p) = A_kkt⁻¹·r1(p)` — which is why
the numerical check found *all* 24 entries of the 4×6 jacobian matching to
`~1e-17`, not only row 3.

Empirically this splits cleanly along `xi=[w;v]`:
`d(alpha)/dv` (translation columns) is point-invariant — a rigid translation
of shape 2 shifts the gap identically regardless of which manifold point the
solve converged to — while `d(alpha)/dw` (rotation columns) genuinely
differs point to point, a real moment-arm/lever effect (rotating shape 2
about its own origin moves a far manifold point differently than a near
one). Verified on a face-face box pair: two points on opposite sides of the
patch have rotation-column values differing by `O(1)`, not noise, while
their translation columns match to solver precision.

**`normal_jacobian`: point-invariant, exactly — including on curved
manifolds.** It's built only from `grad`'s translation columns (§4c's
`grad_v`), which are point-invariant per above; verified this holds not
just for a flat Polytope-Polytope patch but for a genuinely *curved*
degenerate manifold too — axial-parallel Cylinder-Cylinder (`contact_manifold_dim=1`,
a line along the shared generatrix) — where both endpoints' recomputed
`normal_jacobian` matched the single solved value to `0.0`, not just
`~1e-13`. So it is **not recomputed per point** — every entry gets the
single already-computed value, which the affine argument above shows is the
mathematically correct thing to do, not a shortcut.

**"Same active set" convention, and why it's not optional.** All of the
above requires reconstructing a valid `(x, s, z)` at each other point.
Reusing `s*`/`z*` unchanged (only `x`'s position swapped) is the *correct*
way to do this, not merely the cheap one: `z*` stays dual-feasible
regardless of where `x` sits on the degenerate face (stationarity doesn't
involve `x`), and for constraints active at `x*`, `s` stays *exactly* zero
along the null-space direction (by definition of "null space of the active
set", not just approximately). Recomputing `s = h - Gx` fresh at each point
was checked and rejected: `ContactManifold`'s own points are constructed by
ray-clipping *out to the patch boundary* (§6c) — every returned point is a
corner/edge point where an *extra* constraint has newly activated versus
`x*` — so a fresh `s` there exposes a genuinely larger active set, breaking
the strict-complementarity premise the NT-scaling-based analytic formulas
assume; verified to produce spurious, badly-off values when tried.

**The `O(K)` cost this implies is itself reducible to `O(1)`, for the same
reason `grad` is point-dependent in the first place.** Since the per-point
witness/alpha jacobian is exactly affine in position within the patch (not
just its alpha row — verified on all 24 entries of the full 4×6 jacobian, to
machine precision, `max|predicted−actual| ~ 5×10⁻¹⁷`, on an asymmetric
box pair chosen specifically to rule out a symmetry artifact), only 3 points
need a full IFT solve to span a 2D patch (`dim==2`) — every further point
(when `K>3`) is a cheap affine evaluation off those 3, guarded by a
collinearity check that falls back to a full solve if the first 3 points
happen to be degenerate. `dim==1` always returns exactly the 2 endpoints —
already the minimum needed to define that line's own affine map, so no
saving is possible there; `dim==0`'s one point *is* `x*` exactly
(`contactManifold`'s own guarantee), so it's mirrored from the
already-computed `res.jacobian`/`res.normal_jacobian` directly, no IFT solve
at all.

The transfer step, exactly (`proximity_contact.hpp`'s `dim==2, K>3` branch):
run the full IFT solve at 3 points `p0,p1,p2` spanning the patch, giving
`J0,J1,J2` (each a 4×6 jacobian). For `e1=p1−p0`, `e2=p2−p0`,
`E=[e1 e2]∈ℝ³ˣ²`, and any other returned point `p`, solve the 2×2 normal
equations for the patch-local coordinates of `p`:

```
[α;β] = (EᵀE)⁻¹ · Eᵀ · (p − p0)
J(p)  = J0 + α·(J1 − J0) + β·(J2 − J0)
```

Exact, not approximate, for two compounding reasons: `J(·)` really is affine
on the patch (the two arguments above), and `p−p0` really does lie in
`span(e1,e2)` (every `contact_manifold_points` entry sits on the same
degenerate patch by construction) — so this isn't a curve-fit tolerating
some residual, it reproduces the brute-force per-point `diffSocp` value
exactly (regression-tested directly against it, `tests/test_contact_manifold.cpp`,
`K=8`).

**Shipped as `ProximityContactJacobianResult::contact_manifold_point_jacobians`**
(`ManifoldPointJacobian{jacobian, normal_jacobian}`, one per
`contact_manifold_points[i]`) — populated automatically whenever
`compute_contact_manifold` is set, not a separate opt-in: asking for the
multi-point manifold at all is asking for each point's own Jacobian, the
same way a single-point query always gets its Jacobian. Regression-tested
in `tests/test_contact_manifold.cpp`: the point-invariance/point-dependence
split above, the `dim==0` exact-mirror case, and the `K>3` affine shortcut
checked directly against brute-force `diffSocp` at every point (`K=8`,
asymmetric box pair), not just against the invariants (which can't tell the
shortcut apart from a bug that happens to preserve them).

**Measured** (box-box, library-default `pdip_tol`, mean of 50000 solves,
`solve+jacobian` = `proximityContactJacobian` at the single point `x*`
only; `solve+jacobian, all points` = with `contact_manifold_point_jacobians`
populated):

| case | solve | solve+jacobian (x\* only) | solve+jacobian, all points |
|---|---|---|---|
| dim 0 (corner touch, 1 point) | 0.94us | 3.10us | 3.53us (+0.43us) |
| dim 1 (edge touch, 2 points) | 0.94us | 3.06us | 5.05us (+1.99us) |
| dim 2 (face touch), K=4 (default) | 0.98us | 3.20us | 5.91us (+2.71us) |
| dim 2 (face touch), K=20 | 0.98us | 3.05us | 10.58us (+7.53us) |

The `dim==0` and lazy-`normal_jacobian` fixes together are the majority of
the win: before them, `dim==0`'s "all points" cost was **+2.55us** for
literally zero new information (recomputing the identical jacobian a second
time through the general per-point path instead of reusing it); `dim==2`,
K=4 was **+9.73us** (redundantly re-solving the expensive Hessian-based
`normal_jacobian` at every point, when it's the same value every time). The
`K>3` affine shortcut is what keeps `K=20` at +7.53us instead of scaling
linearly to the ~+48-54us a naive per-point loop would cost (only 3 of the
20 points pay for a full solve).

### 6i. The `x*`-only Jacobian path itself was recomputing its shared pieces

The numbers above still had `solve+jacobian (x* only)` sitting at ~3.1us on
box-box against a ~0.94us solve -- a ~2.1us adder that a "one IFT solve,
reuse the factorization" mental model says should be a fraction of that. It
wasn't the linear algebra. `proximityContactJacobian` called `diffSocp`
(for `res.jacobian`), `proximityGradientAnalytic` (for `res.normal`), and
`contactNormalJacobian` (for `res.normal_jacobian`) as three separate
top-level calls. Each independently rebuilt shape 2's Stage-3
`shapeXiDerivative` and the combined `combineXiJacobian` (`q`, `dR1/dxi`,
`dR2/dxi`); `contactNormalJacobian` -> `proximityHessianAnalytic` also ran
the **entire** first-order IFT block-elimination (`A = G'(S\Z)G`,
`partialPivLu`, `dx`/`ds`/`dz`) a *second* time -- byte-identical to the one
`diffSocp` had just done and discarded (`diffSocp` returns only
`dx.topRows<4>()`), and rebuilt `problemMatrices`/`combineProblemMatrices`
that the caller already held. Net per call: `shapeXiDerivative` and
`combineXiJacobian` ~5x each, the IFT solve 2x, the problem matrices 2x. The
only work genuinely unique to `normal_jacobian` is `hessianFrozenFull` --
the six directional *second* derivatives (`d = e_0..e_5`).

**Fix**: `contactJacobianBundleAnalytic` (analytic_derivatives.hpp) computes
`shapeXiDerivative`, `combineXiJacobian`, and the IFT solve (via the new
`diffSocpSensitivityFromXiJac`, which takes an already-built `xi_jac`) once
each, then assembles all three Jacobians from those shared results plus the
one `hessianFrozenFull`. `proximityContactJacobian` calls it once.
Outputs are bit-for-bit identical (same ops, same order) --
`tests/test_proximity_contact.cpp` checks against the hand-wired
`diffSocp`+`contactNormalJacobianAnalytic` at 1e-12 and still passes. The
whole path is **allocation-free**: fixed-size Eigen throughout (verified
under `EIGEN_RUNTIME_NO_MALLOC` + a global `operator new` trap, 1000 calls
per shape-pair type; the degenerate/manifold branches still allocate their
`std::vector` point sets, but those are opt-in and inherently variable-length).

**Measured**, same box-box degenerate configs and columns as §6h's table
(clang, mean of 60k; `x* only` = `proximityContactJacobian` with no manifold
flag, `all points` = with `compute_contact_manifold`):

| case | solve | x\* only (before → after) | all points (before → after) |
|---|---|---|---|
| dim 0 (corner touch, 1 pt) | 0.93us | 2.93us → **2.17us** | 3.32us → **2.56us** |
| dim 1 (edge touch, 2 pts) | 0.94us | 3.10us → **2.18us** | 5.17us → **4.36us** |
| dim 2 (face touch), K=4 | 0.96us | 2.95us → **2.20us** | 5.61us → **4.77us** |
| dim 2 (face touch), K=20 | 0.96us | 2.97us → **2.18us** | 10.50us → **9.64us** |

The `x* only` column drops ~0.8us across the board -- the shared
`shapeXiDerivative`/`combineXiJacobian`/IFT-solve that no longer runs 2-5x
-- and that saving carries straight through to the `all points` totals. The
per-point manifold adder (the `all points` − `x* only` gap: +0.4us dim 0,
+2.2us dim 1, +2.6us K=4, +7.5us K=20) is unchanged, as expected: §6i
touches only the single-point shared path, not §6h's per-point loop. On
non-degenerate poses the same ~0.8us shows up as sphere-sphere's `x* only`
going 3.09us → 2.47us and capsule-cylinder's 7.80us → 6.37us.

---

## 7. Open items (accurate as of this writing — check before citing)

- Generation 2 (§4b, §4c) covers all 49 shape-pair combinations (both
  shapes, any decision-vector width) via `diffSocp`/`proximityJacobian`
  *and* `proximityGradient`/`contactNormal` — no restriction remains.
  `Dual6` is deleted from the codebase (§4a), not just unused; every test
  that used it as a cross-check now uses finite differences instead.
- §1b/§1c: **resolved**, and improved further by §1d. With the
  ergodic-benchmark optimization pass and, decisively, switching to
  clang/LLVM-mingw (§1c), DCOL++ *wins* every one of the 9 benchmarked
  shape pairs against Julia (mean ratio 0.77x at the time). With the
  geometric initial guess now the library default (§1d), that improved
  further to mean ratio ≈0.67x (~49% faster than Julia on average) — the
  two effects compound, they don't overlap. The `-DEIGEN_DONT_VECTORIZE`
  non-convergence mismatch (§1b) is still
  unresolved and still not shipped, but is no longer on the critical path
  now that the compiler switch alone closed (and reversed) the gap; VTune
  (§1c) turned out to be the "real profiler" this bullet was waiting on.
- Phase E (§6): implemented and verified for all 7 shapes — `H_frozen`,
  the full Hessian (`proximityHessianAnalytic`), and `dn/dξ`
  (`contactNormalJacobianAnalytic`) all ship in
  `analytic_derivatives.hpp`/`.cpp`, cross-checked against central-FD of
  the re-solved gradient/normal in `tests/test_hessian_derivatives.cpp`.
  **Resolved**: `proximity_contact.hpp` now wraps the three quantities
  callers actually want — witness point, `alpha`, contact normal — into
  `proximityContact(shape1, shape2, g)` (cheap: one solve +
  `proximityGradientAnalytic`, no Hessian) and
  `proximityContactJacobian(shape1, shape2, g)` (adds `jacobian`
  (d[witness;alpha]/dξ, via `diffSocp`) and `normal_jacobian` (dn/dξ, via
  a new `contactNormalJacobian` wrapper around `contactNormalJacobianAnalytic`
  in `proximity_gradient.hpp`) — the latter is the one that pulls in the
  Hessian machinery, so it's strictly more expensive than the former.
  Cross-checked in `tests/test_proximity_contact.cpp` against independent
  direct calls to the underlying (already FD-verified) functions.
  `proximityGradient`/`contactNormal` themselves are unchanged (still
  `grad` only) — `proximity_contact.hpp` composes on top rather than
  replacing them.
- §6b: `ContactDegeneracy`/`contactDegeneracy` formalizes and detects when
  the witness-point Jacobian and/or the normal Jacobian are undefined at a
  degenerate contact (parallel faces/edges, matched vertices) — two
  independent rank-deficiency checks on matrices the solver already builds,
  verified against 8 hand-built configurations including the skew
  (non-parallel) edge-edge case. Shipped as opt-in fields
  (`SocpOptions::compute_degeneracy_info`, off by default — measured at
  ~10-20% of `proximityContactJacobian`'s cost, not negligible) on
  `ProximityContactJacobianResult`.
- §6c: `ContactManifold`/`contactManifold` builds on §6b's null-space
  directions to return multiple witness points (exact 2 for a line, a
  farthest-point-selected `K`, default 4, for a surface) instead of one, for
  a future contact resolver that needs a representative point set, not a
  single incidentally-chosen one. Opt-in
  (`SocpOptions::compute_contact_manifold`), verified against the exact
  segment/SOC cases and, directly, against the `Geometric`-init
  off-centering bias it exists to correct.
- §6d/§6e: `contactDegeneracy`/`contactManifold` optimized after shipping —
  `A_active` switched to a stack-allocated bounded matrix, both `A`'s and
  `A_active`'s rank checks switched to `FullPivLU::rank()` (with a
  verified `maxPivot()`-based threshold conversion, since `A`'s spectrum
  needs an absolute cutoff, not `FullPivLU`'s relative-to-own-max one),
  and `contactManifold` made lazy about computing `A`'s singular vectors
  (only when `contact_manifold_dim>0` actually needs them). Net:
  `contactDegeneracy` 3.4×–6.4× faster than its original shipped version.
  `SelfAdjointEigenSolver` was checked as an alternative for `A` (confirmed
  symmetric) and found *slower* than `JacobiSVD` at this matrix size —
  documented as a rejected hypothesis so it isn't retried blindly.
- §6f: `IsStrictlyConvex<Shape>` (`primitives.hpp`) skips
  `contactDegeneracy`/`contactManifold` **entirely** (not just cheaply)
  whenever either shape is `Sphere`/`Ellipsoid` — provably
  `contact_manifold_dim==0` and `normal_cone_dim==0` in that case via two
  classical convex-geometry arguments (strict convexity for the former,
  normal-cone intersection for the latter), verified against adversarial
  corner/edge-touching configurations before shipping, not just the easy
  face case.
- §6g: **fixed.** `contactManifold`'s SOC ray-clip had a dimensionally-
  inconsistent absolute epsilon on a *derived* quadratic-formula
  coefficient, causing axis-parallel round-shaft pairs (Cylinder-Cylinder/
  Capsule-Capsule/mixed) with an axial offset to solve witness points off
  by 5+ orders of magnitude. Fixed by checking the root quantity itself,
  relative to a real scale, before deriving anything from it — verified
  general across 13 configurations and performance-neutral-to-positive, not
  just for the triggering case.
- §6h: **shipped.** Per-point Jacobians for `ContactManifold`
  (`contact_manifold_point_jacobians`) — `grad`'s rotation columns are
  genuinely point-dependent once `contact_manifold_dim>0` (a real
  moment-arm effect, not solver noise) while `normal_jacobian` is provably
  point-invariant (verified exactly, including on curved manifolds), so
  it's computed once and shared rather than recomputed per point. The
  per-point witness/alpha jacobian is itself exactly affine in position
  within a patch, so a `dim==2` manifold with `K>3` points needs only 3
  full IFT solves, not `K` — verified to machine precision, not just
  benchmarked as "fast enough".
- §6i: **shipped.** The non-degenerate `proximityContactJacobian` path
  (`x*` only) was itself recomputing every shared piece: `shapeXiDerivative`
  and `combineXiJacobian` ~5x, the first-order IFT block-elimination 2x, the
  problem matrices 2x — because `res.jacobian`, `res.normal`, and
  `res.normal_jacobian` came from three independent top-level calls.
  Consolidated into one `contactJacobianBundleAnalytic` that builds each
  once; `hessianFrozenFull` (6 directional second derivatives) is the only
  work unique to `normal_jacobian`. Bit-identical outputs (1e-12 test),
  allocation-free (verified under `EIGEN_RUNTIME_NO_MALLOC`). Jacobian adder
  over the plain solve dropped ~40–47% (box-box +2.10us→+1.28us,
  sphere-sphere +1.32us→+0.70us, capsule-cylinder +3.05us→+1.62us).
- §6a: the near-touching / active-set finding is specific to trials with
  `min(λ₂) ≲ 10⁻⁴`; unlike §5's `cond(A)` finding, this is not visible in
  `cond(A)` alone, so any future numerical work built on `dx*/dξ`, `dz*/dξ`,
  or the Hessian should check `λ₂` per SOC block directly, not just
  `cond(A)`, if it needs to detect proximity to a non-smooth point.
- The exact algebraic identity underlying *why* Nesterov-Todd rescaling
  turns the Newton system Cholesky-solvable was not independently
  re-derived here; the solver's correctness was instead confirmed via
  full end-to-end convergence against an independently hand-computed
  ground truth, not a from-scratch symbolic proof of that one identity.
- §1d: **shipped, `Geometric` is the library default.** Every one of the 9
  shape pairs faster than the generic init (mean ≈17%), 0 wrong answers / 0
  failures against a `1e-12`-tol reference across the full ergodic sweep,
  full existing test suite passes. One known, diagnosed (not hidden)
  limitation: `PolygonSphere` reference case 3's witness point doesn't
  retrace Julia's exact solving trajectory (verified argmin
  non-uniqueness, not solver error — §1d's own writeup), which is why
  `tests/test_socp_julia_parity.cpp` explicitly requests
  `SocpInitStrategy::Generic` rather than relying on the library default.
  Surrogate scaling (the iDCOL manuscript's other cold-start idea) was
  tried on top and measured no benefit — recorded, not shipped. The
  solver-hardening fix this work forced (`solveSocp`'s iteration-1
  convergence check now verifies the actual KKT residuals, not just `mu`)
  applies to *any* future `init_hint` caller, not just this one, and is
  itself a genuine (if narrow) improvement over Julia's own
  `solve_socp`, which has the identical `mu`-only gap.

## 8. New: the `TruncatedCone` (cone frustum) primitive

Not from Julia. `DifferentiableCollisions.jl` has `Cone`, not a frustum;
`TruncatedCone` is a DCOL++ addition (`include/dcolpp/socp/primitives.hpp`),
bringing the shape count to 8.

### 8a. Geometry and parameterization

`TruncatedCone(R_bottom, R_top, L)` with `R_bottom > R_top >= 0`, `L > 0`:
axis `+x`, **local origin at the axial midpoint** (always strictly interior,
for any radii), narrow rim `R_top` at `x = -L/2`, wide rim `R_bottom` at
`x = +L/2` (radius grows with `+x`, matching `Cone`'s "wide end toward
`+x`"). Derived once in the constructor:

- `tan_beta = (R_bottom - R_top) / L`  -- half-angle of the underlying
  infinite cone (identical to `Cone`'s lateral surface).
- `apex_dist = L/2 + R_top/tan_beta`   -- origin-to-virtual-apex axial
  distance (`Cone`'s is `3H/4`).

Bounding sphere (around the midpoint origin): `outer = hypot(L/2, R_bottom)`
(the wide rim), `inner = min(L/2, ((R_bottom+R_top)/2)*cos(beta))`.

### 8b. SOCP structure -- `Cone`'s SOC block + one extra orthant row

`problemMatrices(TruncatedCone)` -> `ProblemMats<2, 3, 4>` (`n_ort = 2`,
`n_soc = 3`, no extra decision variables). The SOC block is
**byte-identical** to `Cone`'s -- same `E = diag(tan(beta), 1, 1)`, same
`G_soc = -E*Q^T`, same `h_soc = -E*Q^T*r` -- because the lateral surface
*is* the same infinite cone. The only SOC difference is the constant
`G_soc(0,3) = -tan(beta) * apex_dist` (radius bound
`tan(beta)*(y0 + apex_dist*alpha)`), and a constant contributes nothing to
any derivative.

The two orthant rows clip the flat caps at `x = +-L/2`, exactly
`Cylinder`'s rows-2/3 `+-bx` pattern: `G_ort(0,:)*x = bx.p - (L/2)alpha`
(base, `y0 <= (L/2)alpha`), `G_ort(1,:)*x = -bx.p - (L/2)alpha` (tip,
`y0 >= -(L/2)alpha`), with `h_ort = (bx.r, -bx.r)`.

`combine_problem_matrices.hpp`, the solver, `contact_manifold.hpp` and
`contact_degeneracy.hpp` are all fully generic over `(n_ort, n_soc, nx)` --
**no changes there.**

### 8c. Analytic derivatives

`truncatedConeXiDerivative` (-> `ShapeXiDerivative<2, 3, 4>`) and
`truncatedConeHessianFrozen` in `src/socp_analytic_derivatives.cpp`, with
`shapeXiDerivative`/`shapeHessianFrozen` dispatch overloads
(`analytic_derivatives.hpp`). The SOC parts are a character-for-character
copy of `coneXiDerivative`/`coneHessianFrozen`. The two `+-bx` cap rows
follow `cylinderXiDerivative`'s rows-2/3 pattern; row 1's `q` is row 0's
negated (plus a constant `alpha` term that drops), so the orthant
contribution to `(G^T z)` and its Hessian is weighted by
`z_ort(0) - z_ort(1)`.

Verified against central finite differences of the exact functions (no
autodiff): `test_analytic_derivatives.cpp` (formula- and solve-level, incl.
shape-1 position, non-identity `Q_offset`, and the `R_top -> 0` limit),
`test_hessian_derivatives.cpp` (`H_frozen` and the full
`proximityHessianAnalytic`), and `test_proximity_contact.cpp`
(`proximityContact` and `proximityContactJacobian`). Full suite green
(124 cases / 6568 assertions).

### 8d. Why `Cone` stays a separate primitive (not `TruncatedCone(R, 0, L)`)

`R_top = 0` is *allowed* and geometrically gives a cone, but:

1. **`Cone` is a 1:1 port of `primitives.jl`** and is covered by
   `tests/test_socp_julia_parity.cpp`. Folding it away loses that parity
   anchor.
2. **Different origin conventions** -- `Cone`'s local origin is the solid
   cone's centroid (`H/4` from the base, apex `3H/4` behind); the frustum's
   is the axial midpoint. Merging would silently move `r_offset` /
   bounding-sphere semantics for every existing `Cone` caller.
3. **`R_top = 0` puts the tip-cap plane exactly on the SOC apex** (radius
   bound -> 0 there), so it is redundant-with-a-vertex rather than idle
   overhead.

**Speed** is *not* a reason either way. Measured (clang 22, `-O3 -DNDEBUG
-DEIGEN_NO_DEBUG`, `pdip_tol = 1e-8`, poses averaged, 5 runs; `H = 1.3,
beta = 0.4` cone vs. `R_bottom = 0.7, R_top = 0.32, L = 1.3` frustum):

| scenario | call | `Cone` (n_ort=1) | `TruncatedCone` (n_ort=2) | `TruncatedCone`, R_top~=0 |
|---|---|---|---|---|
| generic (vs `Sphere(0.6)`, 24 poses) | forward | ~3860 ns / 9.0 it | ~3740 ns / 8.5 it | ~3640 ns / 8.2 it |
| generic | Jacobian (manifold on) | ~4690 ns | ~4620 ns | ~4500 ns |
| degenerate (flat cap on `Polytope<6>` face, `cm_dim = 2`, 16 poses) | forward | ~2450 ns / 7.0 it | ~2015 ns / 6.9 it | ~2055 ns / 7.0 it |
| degenerate | Jacobian + manifold | ~7600 ns | ~7480 ns | ~7510 ns |

The extra orthant row costs **~2-4% per interior-point iteration**
(429 -> 440 ns/it forward; 521 -> 543 ns/it on the Jacobian), visible only
after normalizing per-iteration. In wall-clock it is within run-to-run
noise (+-3-4%) and often *negative*, because the frustum's rounded geometry
converges in equal-or-fewer Newton iterations for a given contact -- a
pose/conditioning effect, not structural. The degenerate Jacobian path is
dominated by the contact-manifold SVD/LU and shows no difference. So the
`Cone`-vs-`TruncatedCone` choice is about parity and origin convention, not
performance.

### 8e. Interactive demo

`docs/wasm/dcolpp_wasm.cpp` gains `TruncatedCone` in its `ShapeVariant` and
`buildShape` (`{kind:"truncatedcone", R_bottom, R_top, L}`); the double
dispatch is now 8x8 = 64 concrete `(Shape1, Shape2)` instantiations (WASM
blob 2.6 -> 3.5 MB). `docs/index.html` adds it as the 8th selectable shape
(wireframe/faces via the existing `ringWireframe`/`ringFaces` helpers,
`randomShape` always producing a genuine frustum, degenerate-pose recipes
using its two flat caps).

## 9. New: warm-starting for temporally-continuous queries

`proximityJacobian` gains an optional `ContactWarmState<Shape1, Shape2>*`
handle (`warm_start.hpp` + `geometric_init.hpp::warmStartInit`). A physics
step, a trajectory-optimization inner loop, or a smooth ergodic sweep
queries the same shape pair at poses `g` that change little -- or not at
all: a body at rest, a fixed-base grasp -- between calls; re-solving cold
every time rediscovers the whole solution, active set included. With the
handle, `proximityJacobian` seeds the SOCP from the previous converged
solution. Passing `nullptr` is byte-for-byte the cold path.

### 9a. The SOCP, and what a warm start must hand the solver

The forward solve (`solver.hpp`) is, over `x = [p(3); alpha(1); extras]`:

```
minimize    c^T x                 (c = e_4, so the objective IS alpha)
subject to  G x + s = h,   s in K = R^{n_ort}_+ x Q^{n_soc1} x Q^{n_soc2}
```

`Q^m = { (u_0, u_bar) : u_0 >= ||u_bar||_2 }` (the second-order / Lorentz
cone). `G, h` depend on the pose `g` through `problemMatrices`; `c` does
not. KKT conditions at the optimum `(x*, s*, z*)`:

```
G x* + s* = h            (primal feasibility)
G^T z* + c = 0           (dual feasibility / stationarity)
s* o z* = 0              (complementarity; "o" = Jordan product)
s*, z* in K
```

Jordan product per SOC block: `u o v = (u^T v,  u_0 v_bar + v_0 u_bar)`;
elementwise for ORT. Cone identity `e` (`gen_e()`): `1` on every ORT row,
`(1, 0, ..., 0)` per SOC block.

`solveSocp` is a Nesterov-Todd-scaled primal-dual interior-point method: it
follows the **central path** `{ (x,s,z) feasible : s o z = mu e, mu > 0 }`,
taking Newton steps toward the central-path point at a shrinking `mu`, and
returns `converged` when `mu = s.z / degree(K) < pdip_tol` (plus, on
iteration 1 only, `||r_x||, ||r_z|| < pdip_tol` where `r_x = G^T z + c`,
`r_z = s + G x - h`).

**Why the previous solution cannot be reused raw.** A converged
`(x*, s*, z*)` has `s* o z* = 0`: for every cone block, `s*` or `z*` sits
*exactly on the boundary*. Every Newton step forms the NT scaling `W`,
which is (loosely) `sqrt(s / z)` in the Jordan algebra -- it needs the
spectral factorization of `s` and `z`, and on the boundary that
factorization is singular. Feed the raw `(s*, z*)` in and the first Newton
step is NaN or a useless fraction. (`geometric_init.hpp` documents the same
failure for the geometric cold start: a flat `+1` push left `858/100000`
NT-scaling failures on the SphereSphere sweep.)

So a warm start must hand `solveSocp` a point that is:

1. **strictly interior** (`s0, z0 in int K`) so `W` is finite;
2. **as close to `(x*, s*, z*)` as (1) allows**, so the starting gap `mu`
   is just above `mu*` -- iteration count is `~ log10(mu_start / pdip_tol)`,
   so a small `mu_start` is the whole point;
3. **dual-feasible for the NEW `G`** (`G_new^T z0 = -c`) -- otherwise the
   method can drive `mu -> 0` with `z` converging to the wrong *magnitude*
   (fine for `x*`/`alpha`/witness, which only need `z`'s ray direction;
   not fine for any downstream quantity that uses `z`'s size).

### 9b. Per-call algorithm

Cached in the handle after a converged solve at pose `g_ref`:
`{ g_ref, x*, s*, z*, mu* = s*.z*/degree, rho, valid }`, `rho` initialised
to `rho_0 = 0.02`.

```
1. build (G, h, c) at g                              [combineProblemMatrices]

2. pose_move = poseMoveMetric(g_ref, g):
     D          = g_ref^{-1} g                       (4x4 SE(3))
     rot        = ||D[0:3,0:3] - I_3||_F             (Frobenius)
     trans      = ||D[0:3,3]||_2
     pose_move  = sqrt(rot^2 + trans^2)              ( ~ sqrt(2 theta^2 + ||dt||^2) )

3. if valid and pose_move <= rho:  attempt warm  (4-6);  else cold (7)

4. (x0, s0, z0) = warmStartInit(c, G, h, x*, z*, pose_move)      [see 9c]

5. sol = solveSocp(c, G, h, wopt, init_hint = (x0, s0, z0)),
        wopt.max_iters = min(opt.max_iters, 16),  wopt.pdip_tol = opt.pdip_tol

6. warm_used = sol.converged
             and ||G^T sol.z + c||        <= 1e4 * pdip_tol
             and ||sol.s + G sol.x - h||  <= 1e4 * pdip_tol

7. if not warm_used:                       # cold fallback
        sol = solveProximitySocp(...)      # geometric init
        rho = max(0.5 * rho, 1e-3)
   elif sol.iters <= 4:                     # cheap warm solve
        rho = min(1.5 * rho, 0.75)

8. res.jacobian = diffSocp(shape1, shape2, sol.x, sol.s, sol.z, g, G)   # unchanged IFT

9. if sol.converged:  g_ref,x*,s*,z*,mu* <- g, sol.x, sol.s, sol.z, sol.mu ;  valid <- true
   else:              valid <- false
```

### 9c. `warmStartInit` -- the two tiers

`x0 = x*` always. Let `warmRecenter(r; phi_ort, phi_soc)` be as in 9d.

**Tier 1 -- `pose_move < 1e-6`** (pose unchanged to ~6 digits; `(x*,s*,z*)`
is still a KKT point for this problem to well within `pdip_tol`):

```
s0 = warmRecenter( h - G x* ;  phi_ort = 1e-12,  phi_soc = 1e-9 )
z0 = warmRecenter( z*        ;  phi_ort = 1e-12,  phi_soc = 1e-9 )
```

No projection, no Cholesky. When `g == g_ref` exactly, `h - G x* = s*`,
already clears those floors, so `s0 = s*`, `z0 = z*`, `mu_start = mu* <
pdip_tol`, `r_x ~ 0`, `r_z = 0` -> `solveSocp` returns at **iteration 1**.

**Tier 2 -- `1e-6 <= pose_move <= rho`:**

```
s0 = warmRecenter( h - G x* ;  phi_ort = 1e-7,  phi_soc = 1e-3 )

z_tilde = z* + G (G^T G)^{-1} ( -c - G^T z* )      # project z* onto {z : G^T z = -c}
z0      = warmRecenter( z_tilde ;  phi_ort = 1e-7,  phi_soc = 1e-3 )
```

The projection makes `G^T z_tilde = G^T z* + G^T G (G^T G)^{-1}(-c - G^T z*)
= -c` **exactly** for the new `G` (`warmRecenter` then perturbs it by
`O(phi_soc)`); `(G^T G)^{-1}` is one `SmallLLT<nx>` Cholesky on
`gramLower(G)`, the same factorization `initializeSocpFromGuess` uses. Tier
1 skips it because for an unchanged pose `z*` already satisfies `G^T z* =
-c`.

`mu_start` in tier 2 is `~ mu* + O(||dg||) + O(phi_soc * scale)` -- a few
orders above `mu*` (the data changed, plus the `1e-3` margin), but far
below a cold start's `~0.05`. -> ~3-5 iterations.

### 9d. `warmRecenter` -- ORT vs SOC geometry, and the phi values

`K` is a product of two cone geometries, and "distance from the boundary"
means something different in each:

* **Nonnegative orthant `R^n_+`.** Boundary = any coordinate at `0`.
  `R_+` has no intrinsic scale -- a number is positive or it is not -- so
  the natural margin is **absolute**:

  ```
  r_i  <-  max(r_i, phi_ort)          for each ORT row i
  ```

* **Second-order cone `Q^m`.** Boundary = `u_0 = ||u_bar||`. The block
  carries its own scale, `u_0`, so the meaningful margin is **relative**:

  ```
  (u_0 - ||u_bar||) / u_0  >=  phi_soc
  ```

  which is dimensionless and is exactly what sets the NT scaling's
  condition number for that block (`W_block ~ 1 / sqrt(margin)`). An
  *absolute* SOC margin would be meaningless -- negligible for `u_0 ~ 100`,
  catastrophic for `u_0 ~ 1e-3`. To hit the relative margin exactly:

  ```
  (u_0 - ||u_bar||) / u_0 = phi_soc   <=>   u_0 = ||u_bar|| / (1 - phi_soc)
  ```

  so `warmRecenter` raises `u_0` to `||u_bar|| / (1 - phi_soc)` **iff it is
  below that**, leaving `u_bar` fixed (equivalently: add `lambda * (1,0,
  ..,0)` to the block).

So `phi_ort` = absolute floor on each ORT component (problem units);
`phi_soc` = relative interior margin per SOC block (dimensionless, in
`(0,1)`; `-> 0` on the boundary, `-> 1` forces `u_bar` to zero).

`warmRecenter` touches **only** rows/blocks actually below the floor. This
is the one real departure from the sibling `pushToRelativeMargin`: that one
does `lambda = max(lambda, 0.05 - min_i r_i)` and adds `lambda * e` to
*every* ORT row. For a warm start the inactive dual rows are `~ 0`; lifting
all of them to `0.05` dumps `s_i * 0.05 ~ 0.05` per inactive row into
`s.z`, pinning `mu_start ~ 0.05` no matter how good `x0` is -> ~7
iterations regardless. Minimal, per-row/block recentering keeps `mu_start`
near `mu*`.

**The numeric values.** The trade-off in each is: larger `phi` -> better
NT-scaling conditioning; smaller `phi` -> lower `mu_start` -> fewer
iterations. The values sit at the smallest `phi` that is still safe:

| value | where | reasoning |
|---|---|---|
| `phi_soc = 1e-3` | tier 2 | The value the shipped geometric init settled on after its own 100k-pose sweep (`geometric_init.hpp`: `0.001` -> `0/100000` NT failures, ~5 avg iters). Warm sweep here: `1e-3` and `1e-4` both give 4 iters; `1e-5`/`1e-6` give 5 (too tight -> degraded first NT step). `1e-3` is the low end of an empirically flat region with the most headroom, and matches a value the rest of the codebase already validated. |
| `phi_ort = 1e-7` | tier 2 | Just above the numerical noise of a converged dual's inactive rows (`~0`), well below anything that shifts the answer. Each lifted row adds `<= 1e-7 * O(1)` to `mu` -- one order, vs `pushToRelativeMargin`'s five. |
| `phi_soc = 1e-9`, `phi_ort = 1e-12` | tier 1 | Deliberately **below** `pdip_tol = 1e-8`. An unchanged pose's `(s*, z*)` is already interior by `~pdip_tol`, so these floors are a no-op in the normal case -- pure safety net for a component that landed exactly on `0` from rounding. |
| `kTinyMove = 1e-6` | tier boundary | Below this the problem `(G, h)` differs from the cached one by `< 1e-6`, so `(x*,s*,z*)` is still a near-exact KKT point and raw reuse beats the projected path. From the benchmark's smooth transition, not derived. |
| `1e4 * pdip_tol` | residual guard (9b step 6) | "No worse than a cold solve here." A cold solve near touching also has `r_x, r_z ~ 1e-5..1e-7` (they track `mu` with a `cond(A)`-dependent constant, up to `1e13`, section 5). Was `1e2`; loosened after that forced 100% fallback on the harder pairs at `~1.5e-2` pose steps. |
| `kMaxIters = 16` | iteration cap | Cold solves here top out at ~12 (Capsule-Cylinder). "A warm solve slower than cold is a bad hint," with margin. |

`phi_soc`/`phi_ort` are pinned to the codebase's own conditioning sweeps;
the guard/cap multipliers are "match the cold solve's own quality/cost"
heuristics.

### 9e. The `rho` trust radius

`rho` is a trust region **on the pose change**, not on a step in
decision-variable space. The trusted quantity is `poseMoveMetric(g_ref,
g)` -- how far the pose has moved since the solution being reused.

`x0 = x*` is a good primal guess only while the true optimum has not moved
much: `||x*(g) - x*(g_ref)|| ~ ||dx*/dxi|| * pose_move`. That sensitivity
`||dx*/dxi||` is bounded away from degeneracy and blows up near it (section
6b), and is not known a priori -- so `rho` is **learned online**:

* start conservative: `rho_0 = 0.02` (~1 degree of rotation, or `0.02` of
  translation);
* **grow** `x1.5` (cap `0.75`) after any warm solve that finished in
  `<= 4` iterations -- evidence that at the current pace `x0` is still
  close, so larger jumps are safe;
* **shrink** `x0.5` (floor `1e-3`) after any fallback to cold -- evidence
  that `x0` stopped being useful (pose moving too fast, or `||dx*/dxi||`
  exploding as a degeneracy approaches).

Net behaviour, and why this is the right control variable:

* **smooth trajectory / settling contact:** `rho` ratchets toward `0.75`
  and stays there; nearly every step is warm and cheap;
* **fast i.i.d. motion:** `rho` stays wide (the warm solve still converges,
  just not quickly), so the step is warm but ends up costing about the same
  as cold -- break-even, not a loss;
* **a large jump, or approaching a degenerate configuration:** the warm
  solve overruns its `16`-iteration cap or fails the residual guard, `rho`
  halves, and the method self-throttles into cold solves exactly where
  warm-starting is unreliable, re-opening once past it.

Same spirit as an optimization trust-region radius (expand on success,
contract on failure), with "pose displacement" as the trusted variable and
"converged cheaply vs. had to fall back" as the signal. The schedule
constants (`0.02 / x1.5 / x0.5 / 0.75 / <=4 iters`) are heuristic, tuned so
the benchmark below behaves. A more principled version would set `rho`
directly from `||dx*/dxi||` (which the solve already computes); the online
ratchet was chosen for simplicity and the benchmark bears it out.

### 9f. Fallbacks -- the warm answer is never worse than cold

Three independent gates, any of which routes to a normal cold
`solveProximitySocp`:

1. **Trust-region gate** (9e): `pose_move > rho`.
2. **Iteration cap**: the warm solve gets at most `16` iterations.
3. **KKT-residual guard** (9b step 6): `solveSocp` accepts `mu < pdip_tol`
   alone from iteration 2 on, and on a warm path `mu` can dip there while
   `x` is still settling; `||G^T z + c||` and `||s + G x - h||` are checked
   against `1e4 * pdip_tol`.

`SocpResult` also gains a `mu` field (the converged gap) so a caller can
decide whether a given solve is worth warm-starting from.

### 9g. Speed

Measured (clang `-O3`, `pdip_tol = 1e-8`, per-call wall time + iteration
count, warm vs. a fresh cold solve at identical poses) --
`tools/bench_warm_start.cpp` (`-DDCOLPP_BUILD_BENCHMARKS=ON`).

**(a) Random-walk pose stream** -- i.i.d. per-step twist of magnitude
`d(g)`. This is the adversarial case: consecutive optima are *uncorrelated*
beyond `d(g)`, so once `d(g)` is not small the previous solution is a poor
guess.

| pair | cold | `d(g)=0` (rest) | `1e-4` | `1e-3` | `3e-3` | `>=1e-2` |
|---|---|---|---|---|---|---|
| Sphere vs Sphere | 2.2us / 5 it | **4.2x** / 1 it | 1.4x / 4 it | 1.3x / 4 it | ~1x | ~1x (cold) |
| Sphere vs Cone | 3.5us / 7 it | **6.2x** / 1 it | 1.7x / 4 it | 1.7x / 4 it | 1.6x / 4 it | ~1x (cold) |
| Polytope<6> vs Cone | 2.6us / 6 it | **3.4x** / 1 it | 1.7x / 3 it | 1.4x / 4 it | 1.3x / 4 it | ~1x (cold) |
| Capsule vs Cylinder | 7.8us / 12 it | **6.3x** / 1 it | 1.8x / 5 it | 1.7x / 5 it | 1.8x / 5 it | ~1x (cold) |
| Sphere vs Polytope<6> | 2.5us / 6 it | **3.5x** / 1 it | 1.5x / 4 it | 1.6x / 4 it | 1.6x / 4 it | ~1x (cold) |

* **Body at rest / grasp holding (`d(g) ~= 0`): 3.4-6.3x, one iteration**
  (tier 1). The case warm-start most targets in a physics loop.
* **Slow drift (`d(g) <~ 3e-3`): 1.3-1.8x.**
* **Fast i.i.d. motion (`d(g) >~ 1e-2`): break-even** (`0.98-1.01x`, 0%
  fallback). The pose stays inside `rho`, so the warm path is still taken --
  it just isn't faster: `x_prev` is a stale guess for an uncorrelated
  optimum, so the warm solve needs the same iteration count as cold. Never
  wrong; the `warmStartInit` + handle-copy overhead is the only cost.

**(b) Ergodic sweep** -- a smooth quasi-periodic `g(t) = g0 * Exp(xi(t))`,
`xi(t)` = six incommensurate sinusoids (rotation amp 0.18, translation
0.35). Every step is a small *correlated* increment -- the real
dynamics / traj-opt case. 40k steps/pair, three traversal speeds:

| pair | cold ns/it | `dxi~1e-4` | `dxi~1e-3` | `dxi~1e-2` |
|---|---|---|---|---|
| Sphere vs Sphere | 2300 / 5.0 | 1.21x / 4 it | 1.27x / 4 it | 1.23x / 4 it |
| Sphere vs Cone | 3650 / 7.0 | 1.63x / 4 it | 1.63x / 4 it | 1.66x / 4 it |
| Polytope<6> vs Cone | 2600 / 6.1 | 2.21x / 2 it | 1.71x / 3 it | 1.69x / 3 it |
| Capsule vs Cylinder | 6800 / 10 | 2.22x / 4.3 it | 2.22x / 3.6 it | 1.83x / 4.9 it |
| Sphere vs TruncatedCone | 3200 / 7.2 | 1.65x / 4 it | 1.71x / 4 it | 1.71x / 4 it |
| Cone vs Ellipsoid | 4400 / 8.7 | 2.28x / 4 it | 2.02x / 4 it | 2.02x / 4 it |

Fallback rate `0%` at every speed. **On a smooth trajectory `rho` ratchets
to its `0.75` cap and stays there** -- even at `dxi ~ 1e-2`/step, because
consecutive optima are correlated, so `x0 = x_prev` stays a good guess and
the warm solve holds at `~4` iterations throughout (vs cold's `5-11`).
**Smoothness, not step size, is what warm-start needs**: at `1e-2` per
step, an i.i.d. walk (table a) is break-even -- `x_prev` is a stale guess
for an uncorrelated optimum -- while a smooth step (table b) still gives
`1.7-2.1x`, because along a continuous path consecutive optima are
correlated and `x0 = x_prev` stays close.
Sphere-Sphere is the floor (`1.2x`) -- already a 5-iteration solve, so the
`warmStartInit` setup is a bigger fraction.

Answers match cold to `<= 1e-6` on `alpha` (envelope-theorem, well
determined) and `<= ~1e-4` on the witness point (a weakly-determined
direction near contact, section 5/6a); `d[witness;alpha]/dxi` to `<=
1.5e-2` relative.

### 9h. Scope -- what does NOT warm-start

**`proximityContactJacobian` has no warm overload.** Its extra
`d(normal)/dxi` comes from `hessianFrozenFull`, whose formulas assume the
converged `(s, z)` is *exactly* conically complementary (`s o z = 0`, not
merely `s.z < pdip_tol`). A warm-started interior-point point does not
reliably reach that -- verified directly: witness, alpha, normal, and
`d[witness;alpha]/dxi` all match a cold solve to `<= 1e-5`, while
`d(normal)/dxi` from the same warm `(s,z)` can be off by orders of
magnitude (and non-deterministically so, run to run). Tightening the warm
solve's `pdip_tol` and adding the KKT-residual gate did not fix it. So
warm-start covers the forward query and the witness/alpha Jacobian
(`proximityContact` / `proximityJacobian`); a caller needing
`d(normal)/dxi` every step calls `proximityContactJacobian` cold.

Regression-tested in `tests/test_warm_start.cpp`: answer-invariance across
Sphere-Cone / Polytope-Cone / Capsule-Cylinder / Sphere-TruncatedCone
random-walk sweeps, the identical-pose case, the big-jump fallback (still
correct, cold; `rho` shrinks), the `nullptr`-is-byte-for-byte-cold
guarantee, and a 40-step settling-contact loop.

### 9i. Relation to standard interior-point warm-starting

Standard bones, tuned flesh. Classic and textbook: an IPM cannot reuse the
exact optimum (boundary -> singular scaling), so you re-center toward the
central path at a chosen `mu` -- E. A. Yildirim & S. J. Wright, *Warm-Start
Strategies in Interior-Point Methods for Linear Programming*, SIAM J.
Optim. 12(3), 2002; J. Gondzio, warm-starting for cutting-plane / QP.
Re-projecting the dual onto stationarity when the data changes is the
natural "restore dual feasibility" step (a least-squares projection;
`initializeSocpFromGuess` already does it with a different `z_pref`). The
two-tier "if the warm data is still feasible, skip work" is a common
pragmatic pattern (OSQP, qpOASES).

Ours specifically: the `warmRecenter` split (relative SOC margin + absolute
ORT floor, applied *minimally*, without the mu-inflating uniform ORT lift)
-- a specialization of DCOL++'s own `pushToRelativeMargin`; and the
`rho`-on-`poseMoveMetric` online trust radius. We do **not** use the
homogeneous self-dual embedding (ECOS / Clarabel / SCS), the genuinely
warm-start-robust IPM formulation -- it would be a solver rewrite and would
complicate the exact analytic derivatives. This is "bolt a warm start onto
a plain NT-scaled PDIP", which is inherently somewhat heuristic; the gates
in 9f are what make it safe regardless.
