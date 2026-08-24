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
`proximity_jacobian`/`proximityJacobian` on the same 8 shape pairs and
poses, `pdip_tol=1e-10`, 20k-iteration average after a 10-call warmup
(Julia: JIT warmup; C++: `-O3`, MinGW g++). Initial measurement: **the C++
port was slower than Julia, by 1.25x-2.6x**, not faster. A profiling and
optimization pass (below) closed part of that gap; current numbers:

| pair | Julia (us) | C++, before (us) | C++, after (us) | C++/Julia now |
|---|---|---|---|---|
| SphereSphere | 5.94 | 13.02 | 11.53 | 1.94x |
| SphereCapsule | 7.68 | 19.40 | 17.62 | 2.29x |
| CapsuleCylinder | 9.28 | 20.30 | 18.77 | 2.02x |
| CylinderCone | 8.61 | 19.48 | 15.08 | 1.75x |
| ConePolytope | 5.71 | 7.75 | 5.78 | 1.01x |
| PolytopeEllipsoid | 8.72 | 10.85 | 10.03 | 1.15x |
| EllipsoidPolygon | 10.78 | 16.75 | 15.33 | 1.42x |
| PolygonSphere | 11.34 | 22.06 | 19.66 | 1.73x |

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
- §1b: after `SmallLLT` and the other safe optimizations, the C++ port is
  1.01x-2.3x slower than Julia on `proximityJacobian` (down from 1.25x-2.6x;
  ConePolytope is now at parity). Closing the rest of the gap likely needs
  either a real profiler (perf/VTune-class, not available in this MinGW
  environment) or resolving the `-DEIGEN_DONT_VECTORIZE` non-convergence
  mismatch documented in §1b, which showed the largest single lever (~33%)
  but was not safe to ship. Worth revisiting before citing a performance
  claim in the paper.
- Phase E (§6): implemented and verified for all 7 shapes — `H_frozen`,
  the full Hessian (`proximityHessianAnalytic`), and `dn/dξ`
  (`contactNormalJacobianAnalytic`) all ship in
  `analytic_derivatives.hpp`/`.cpp`, cross-checked against central-FD of
  the re-solved gradient/normal in `tests/test_hessian_derivatives.cpp`.
  Not yet wired into `proximityGradient`/`contactNormal` as the default
  public API the way §4b/§4c's first derivatives are (`proximityGradient`
  still only returns `grad`, not the Hessian or `dn/dξ`) — these are
  available as their own functions, callable directly, but nothing in
  `proximity_gradient.hpp` calls them yet.
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
