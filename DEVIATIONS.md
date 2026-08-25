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
