# Deviations from DifferentiableCollisions.jl

`dcolpp::socp` started from
[DifferentiableCollisions.jl](https://github.com/kevin-tracy/DifferentiableCollisions.jl)
(Kevin Tracy, MIT): the SOCP proximity formulation, the interior-point solver,
the cone algebra, the primitive geometry, and the implicit-function-theorem
Jacobian. On top of that core it re-targets *what* is differentiated and adds a
second-derivative layer, a contact layer, warm-starting, and a new primitive.
This document separates "faithful port" from "new contribution", and gives the
derivation of every formula that is original here, so a paper can cite each
piece precisely.

**Legend.** *unchanged* = same math, ported directly. *re-targeted* = same
method, different parameterization. *new* = not in the Julia source.

**Notation.** `g = (R, p) ∈ SE(3)` is the relative pose of shape 2 in shape 1's
frame (shape 1 fixed at `g = I`). `ξ = [w; v] ∈ ℝ⁶` is a body twist,
rotation-first, applied by exact right-multiplication `g(ξ) = g₀ · Exp(ξ)`. A
hat `x̂ = skew(x)` is the 3×3 cross-product matrix; `x̂ y = x × y = −ŷ x`. `eⱼ`
is the j-th standard basis vector; `êⱼ` the corresponding basis skew. All
`d…/dξ` Jacobians are evaluated at `ξ = 0` (i.e. at `g₀`).

---

## 1. Unchanged from Julia (straight port)

**The SOCP**, over `x = [p(3); α(1); extras]`, `c = e₄` (objective *is* `α`),
`K = ℝⁿ_ort₊ × Qⁿ_soc¹ × Qⁿ_soc²` (`Qᵐ = {(u₀, u_bar) : u₀ ≥ ‖u_bar‖}`):

```
minimize   cᵀx    s.t.   G x + s = h,   s ∈ K
```

`G, h` depend on the pose `g` (via `problemMatrices`); `c` does not. KKT at the
optimum `(x*, s*, z*)`: `Gx* + s* = h`, `Gᵀz* + c = 0`, `s* ∘ z* = 0`,
`s*, z* ∈ K` (`∘` = the Jordan product; `s ∘ z = 0` ⟹ each block sits on a
cone boundary).

Validated against the actual Julia library's output
(`tools/gen_socp_reference.jl` runs `DCD.proximity` directly;
`tests/test_socp_julia_parity.cpp` checks the 7 shared shapes chained pairwise,
8 poses each, to `1e-6` on `alpha` / `1e-5` on the witness point).

- **Primitive geometry → `(G_ort, h_ort, G_soc, h_soc)`** for the 7 shared
  shapes (`problem_matrices.hpp`) — `src/problem_matrices.jl`.
- **NT-scaled Mehrotra predictor–corrector interior-point solver**
  (`solver.hpp`) — `src/solvers/coneqp/static_solver2.jl`: same
  initialization structure, affine/centering split, `σ = clamp(ρ,0,1)³`,
  line search.
- **Cone algebra** (`cone_utils.hpp`): `arrow`, `soc_cone_product`,
  `inverse_soc_cone_product`, `soc_quad_J` — `soc_utils.jl`.
- **NT scaling** (`nt_scaling.hpp`) — `NT_scaling_chol_2.jl`.
- **IFT block-elimination** for `dx/dξ, ds/dξ, dz/dξ`
  (`computeSocpSensitivity`, `analytic_derivatives.hpp`) — `diff_socp` in
  `src/proximity.jl`.
- **Envelope-theorem gradient** (`computeProximityGradient`) —
  `src/proximity_gradient.jl`.

The KKT/IFT linear algebra is untouched. The work was in re-targeting *what* is
differentiated (§2) and *how* the Jacobian pieces are produced (§3–4).

### 1a. Derivation of the IFT block-elimination (for reference)

The port keeps this verbatim; the derivation is recorded because every new
formula in §4 plugs into it.

Treat the KKT residual as a function of the parameter `ξ` (through `G(ξ), h(ξ)`;
`c` is constant) and of the primal/dual unknowns:

```
Φ(ξ, x, s, z) = [ G x + s − h ;  Gᵀ z + c ;  s ∘ z ] = 0.
```

On a fixed solution branch, `dΦ/dξ = 0`. With `S := arw(s)`, `Z := arw(z)` (so
`s ∘ z = S z = Z s`, `d(s∘z) = Z ds + S dz`):

```
G dx + ds            =  q,          q  := ∂h/∂ξ − (∂G/∂ξ) x          (primal)
Gᵀ dz                =  r₁,         r₁ := −(∂Gᵀ/∂ξ) z                (dual)
Z ds + S dz          =  0.                                          (complementarity)
```

Eliminate: from the primal row `ds = q − G dx`; substitute into the
complementarity row,

```
dz = −S⁻¹ Z (q − G dx) = (S⁻¹Z) G dx − S⁻¹ Z q = (S⁻¹Z) G dx + S⁻¹ r₂,
     r₂ := −Z q ;
```

substitute that into the dual row,

```
Gᵀ (S⁻¹Z) G dx + Gᵀ S⁻¹ r₂ = r₁
  ⟹  A dx = r₁ − Gᵀ S⁻¹ r₂ ,   A := Gᵀ (S⁻¹Z) G .
```

`S⁻¹Z` is diagonal (`zᵢ/sᵢ`) on the ORT part and `arw(s)⁻¹arw(z)` per SOC block
(`inverse_soc_cone_product`). `A` is symmetric only on the exact central path
(where `arw(s), arw(z)` commute); tests converge as loose as `pdip_tol = 1e-2`,
where it is materially non-symmetric, so `computeSocpSensitivity` solves the
`A dx = …` system with a pivoted LU, not a Cholesky (see §11).

---

## 2. Re-targeted: 6-DOF local twist instead of 14-DOF world state

**Julia.** Each primitive carries an absolute world pose `(r, q)` (position +
quaternion), duplicated for an MRP variant of every function. `ForwardDiff`
differentiates `kkt_R` w.r.t. the concatenated `[r₁;q₁;r₂;q₂]` (14 numbers),
and downstream code extracts the 6-DOF quantity it needs.

**DCOL++.** A pair is one relative pose `g = g₁⁻¹g₂ ∈ SE(3)` (shape 1 fixed at
`g = I`). Differentiation targets a single 6-DOF **body twist** `ξ = [w; v]`
(rotation-first), applied by exact right-multiplication `g(ξ) = g₀·Exp(ξ)` — not
a linearized perturbation. One function per shape, not two (a twist has no
normalization DOF for a second parameterization to fix).

Dimensionally minimal, velocity-like, no quaternion double-cover, and the
convention robotics/contact mechanics already uses for relative-pose
sensitivity. Verified against central FD of the exact `Exp` from the start.

**Chain rule to robot generalized coordinates.** For any query output `Y`,

```
dY/dq = (dY/dξ) · J_rel(q),
```

where `J_rel(q)` (6×n_dof, rows `[angular; linear]`) is shape 2's body Jacobian
minus `Ad_{g⁻¹}` times shape 1's (just shape 2's if shape 1 is world-fixed).
This is why the library differentiates w.r.t. `ξ` only — the mapping to any
particular robot parameterization is one matrix multiply the caller supplies.

### 2a. Dropped the per-shape mounting offset (`r_offset` / `R_offset`)

Julia composes a local mount with the pose (`R = Rg·R_offset`,
`r = rg + Rg·r_offset`). DCOL++ carried the fields but never exposed them, so
they were speculative generality threaded through every `problemMatrices`,
`*XiDerivative`, and `*HessianFrozen`. Removed: a shape is defined in its own
frame, `(R, p)` read straight off `g`. A caller needing a mount folds it into
`g` (`g = (g₁·g₁_offset)⁻¹·(g₂·g₂_offset)`) and, for the Jacobian,
post-multiplies by the corresponding constant `Ad`.

---

## 3. New: SE(3) Lie-group layer (`se3.hpp` / `se3.cpp`)

Not in Julia (which differentiates raw `(position, quaternion)` tuples). Ported
from the author's **SoRoSim++** `lieBrary`, not from DCOL: `Exp`, `skew`, `hat`,
and the three first-derivative primitives the analytic path is built from.

### 3a. Derivation of the first-derivative primitives

Right-perturb `g₀ = (R₀, p₀)` by a twist and expand to first order. With
`Exp(ξ) = ( exp(ŵ),  V(w) v )`, `exp(ŵ) = I + ŵ + O(‖w‖²)`, `V(w) = I + O(‖w‖)`:

```
R(ξ) = R₀ exp(ŵ) = R₀ (I + ŵ) + O(‖ξ‖²)
p(ξ) = p₀ + R₀ V(w) v = p₀ + R₀ v + O(‖ξ‖²).
```

So `∂R/∂w_j = R₀ êⱼ`, `∂R/∂v = 0`, `∂p/∂w = 0`, `∂p/∂v = R₀`. Apply to the
three quantities the analytic derivatives need (`c` a fixed local vector, `u` a
fixed shape-1-frame vector), using `x̂ y = −ŷ x`:

**Rotated vector** `R c` (a local *direction*, e.g. a shape axis):

```
d(R c)/dw_j = R₀ êⱼ c = R₀ (eⱼ × c) = −R₀ ĉ eⱼ   ⟹   dRotatedVectorDXi(g₀,c) = [ −R₀ ĉ | 0 ].
```

**Placed point** `R c + p` (a local point carried by the pose):

```
d(R c + p)/dξ = [ −R₀ ĉ | R₀ ]   =:   dPointDXi(g₀, c).
```

**Inverse-rotated vector** `Rᵀ u` (a fixed shape-1 vector pulled into the local
frame). From `Rᵀ(ξ) = (R₀ + R₀ŵ)ᵀ = R₀ᵀ − ŵ R₀ᵀ`:

```
d(Rᵀ u)/dw_j = −êⱼ R₀ᵀ u = −(eⱼ × R₀ᵀu) = \widehat{R₀ᵀ u} eⱼ
  ⟹  dInverseRotatedVectorDXi(g₀, u) = [ \widehat{R₀ᵀ u} | 0 ].
```

Written out (`c̃ = skew(c)`):

```
dRotatedVectorDXi(g₀, c)         = d(R c)/dξ          = [ −R₀·c̃        | 0  ]
dPointDXi(g₀, c)                 = d(R c + p)/dξ      = [ −R₀·c̃        | R₀ ]
dInverseRotatedVectorDXi(g₀, u)  = d(Rᵀ u)/dξ, u fixed = [ (R₀ᵀu)̃      | 0  ]
```

Because these are taken at `ξ = 0`, the left-Jacobian series in `V(w)` collapses
to `I` and no `V`-derivative term survives.

**One deliberate correction against SoRoSim++'s own code:** `Exp`'s small-angle
branch. The correct Taylor expansion of `(θ − sinθ)/θ³` is `1/6 − θ²/120 + …`;
SoRoSim++ has `θ³/120`. DCOL++ uses `θ²/120` (matters only for `θ < 10⁻²`).

---

## 4. New: fully analytical derivatives (no autodiff)

Julia differentiates `kkt_R` with `ForwardDiff` universally. DCOL++ has no
autodiff anywhere — a hand-built dual-number path (`Dual6`) was the first
generation and is now deleted, once the analytical path covered every shape and
both derivative orders. Every test that cross-checked against `Dual6` now uses
central FD of the plain-`double` functions.

### 4a. Three-stage chain rule (`analytic_derivatives.{hpp,cpp}`)

1. **Generic primitives** (`se3::dPointDXi` / `dRotatedVectorDXi` /
   `dInverseRotatedVectorDXi`, §3) — closed-form `d/dξ` at `ξ = 0`.
2. **Per-shape** (`sphereXiDerivative` … `polygonXiDerivative<NH>`) — one
   hand-derived function per primitive, obtained by differentiating that
   shape's own `problemMatrices` expression, always in *already-contracted*
   form (`d(Gx)/dξ`, `d(Gᵀz)/dξ`, `d(h)/dξ`); a full `∂G/∂ξ` tensor is never
   materialized.
3. **Combine** — `combineXiJacobian` places a shape's stage-2 output into the
   combined system's `dR₁/dξ, dR₂/dξ, q`, feeding the unchanged IFT solve (§1).

All 8 shapes are implemented as the moving body, for any decision-vector width
on either shape (49→64 pairings). `proximityJacobian` uses this path
unconditionally.

**Worked stage-2 example — the Cone SOC block.** `problemMatrices(Cone)` puts
the moving point through `E Rᵀ(p_dec − p)` where `p_dec` is the decision
variable, `p` the pose translation, and `E = diag(tanβ, 1, 1)`. The contracted
pieces:

- `G_soc·x_local = −E Rᵀ p_dec + (const)·α`. Only `Rᵀ p_dec` moves with `ξ`
  (`p_dec` frozen), so `d(G_soc x)/dξ = −E · dInverseRotatedVectorDXi(g₀, p_dec)`.
- `h_soc = −E Rᵀ p`. Here `p = p(ξ)` moves *too*, so the product rule gives
  `d(Rᵀ p)/dξ = dInverseRotatedVectorDXi(g₀, p₀) + R₀ᵀ · dPointDXi(g₀, 0)`
  — one term with `p` frozen, one with `Rᵀ` frozen.
- `(G_socᵀ z_soc)` first three entries `= −R (E z_soc)`; `E z_soc` is
  ξ-constant, so `= −dRotatedVectorDXi(g₀, E z_soc)`.

Every other shape's stage-2 function is the same exercise on its own geometry
(`src/analytic_derivatives.cpp`).

### 4b. The gradient identity `dα/dξ = −qᵀz`

`α` is the SOCP objective (`cᵀx = α`), so `dα/dξ = cᵀ dx*/dξ`, which naively
needs the full re-solve sensitivity. The envelope theorem collapses it.

**Derivation.** Adjoin only the affine constraint (the cone is handled by the
interior-point barrier):

```
L(x, s, z; ξ) = cᵀx + zᵀ ( G(ξ) x + s − h(ξ) ).
```

At a KKT point `∂L/∂x = c + Gᵀz = 0` and the multiplier is stationary, so the
total derivative of the optimal value equals the *partial* in the explicit
parameter (envelope theorem):

```
dα*/dξ = ∂L/∂ξ |_(x*,s*,z*) = z*ᵀ ( (∂G/∂ξ) x* − (∂h/∂ξ) ).
```

With `q := ∂h/∂ξ − (∂G/∂ξ) x*` — exactly the per-block `qₖ = hₖ − Gₖ x`
differentiated, which `combineXiJacobian` already forms — this is

```
dα*/dξ = −z*ᵀ q = −qᵀ z*      (1×6).
```

**Exact**, not a linearization, and O(1) after the solve (no IFT system). Julia
has this trick (`proximity_gradient.jl`); the contribution here is the analytic,
autodiff-free realization. The contact normal builds on it:
`n̂ ∝ Rg · (dα/dξ)_v` (translation half of the gradient; Le Cleac'h et al.,
RAL 2023, Eq. 14).

### 4c. The second derivative `d²α/dξ²` — not in Julia at all

Julia has no Hessian/second-derivative code (confirmed by source search). The
contact normal's Jacobian needs it.

**Why the naive version is wrong.** Differentiating `dα/dξ = −qᵀz` a second time
with `x*, z*` held fixed drops the terms where the optimizer itself moves;
`x*(ξ), z*(ξ)` do move. Measured ~44% error on a two-sphere case.

**Correct decomposition.** `dα/dξ = F(ξ, x*(ξ), z*(ξ))` with
`F(ξ, x, z) := −q(ξ; x)ᵀ z`, whose dependence on `ξ` is both explicit (through
`G(ξ), h(ξ)`) and implicit (through `x*, z*`). Total derivative:

```
d²α/dξ² = ∂F/∂ξ |_(x,z frozen)  +  (∂F/∂x)(dx*/dξ)  +  (∂F/∂z)(dz*/dξ).
```

Each `ξ`-component of the gradient is `Fⱼ = −qⱼᵀ z`, `qⱼ = ∂h/∂ξⱼ − (∂G/∂ξⱼ) x`
— **linear in `x`**, so `∂qⱼ/∂x = −∂G/∂ξⱼ` and

```
∂Fⱼ/∂x = −(∂qⱼ/∂x)ᵀ z = (∂G/∂ξⱼ)ᵀ z = −r₁[:,j] ,
∂Fⱼ/∂z = −qⱼ ,
```

using `r₁ := −(∂Gᵀ/∂ξ) z` (column `j` is `−(∂G/∂ξⱼ)ᵀ z`), the same `r₁` the IFT
uses. Stacking `j = 1…6`:

```
d²α/dξ²  =  H_frozen  −  r₁ᵀ (dx*/dξ)  −  qᵀ (dz*/dξ) ,
```

- `H_frozen := ∂F/∂ξ` with `x*, z*` frozen — the closed-form 6×6 below.
- `dx*/dξ, dz*/dξ` — from `computeSocpSensitivity` (§1). No new work.
- `r₁, q` — from `combineXiJacobian`. **No per-shape derivation** for the
  cross-terms: they are structurally `−r₁ᵀ` and `−qᵀ`.

Verified `< 5×10⁻³` relative Frobenius against central FD of the *re-solved*
gradient, for every pairing.

#### `H_frozen` closed form

`H_frozen = d/dξ[ −qᵀz ]` with `x, z` frozen `= d/dξ[ Σₖ zₖ (Gₖ x − hₖ) ]`. For
every shape, each pose-dependent factor of `Gₖ x − hₖ` is a frozen covector `a`
contracted with **one of the §3 field-Jacobian patterns** `J(g)` — `d(Rc)/dξ`,
`d(Rc+p)/dξ`, `d(Rᵀu)/dξ`, or the combination `d(Rᵀ(Rc+p))/dξ`. So
`grad = aᵀ J(g)`, and we need `d/dξ[ J(g)ᵀ a ]`.

**The two-step pattern.** (i) Contract the §3 primitive with `a` and simplify
using `û a = −â u`; the result is `[ (something) · (R or Rᵀ)(fixed) ; … ]`.
(ii) Differentiate once more — only the remaining `R(·)` / `Rᵀ(·)` / `p` carries
`ξ`, so substitute the matching §3 primitive again. Everything else is constant,
giving a fixed 6×6 whose column `j` is `d(Jᵀa)/dξⱼ`.

**Worked: `d2Point(a, c) = d/dξ[ (dPointDXi(g,c))ᵀ a ]`.** Step (i):

```
(dPointDXi)ᵀ a = [ (−R ĉ)ᵀ a ; Rᵀ a ] = [ ĉ (Rᵀ a) ; Rᵀ a ]      (used ĉᵀ = −ĉ).
```

Step (ii): only `Rᵀ a` depends on `ξ`; by §3,
`d(Rᵀ a)/dξ = [ \widehat{R₀ᵀ a} | 0 ]`. With `a_r := R₀ᵀ a`:

```
d2Point(a, c) = [ ĉ · ã_r  | 0 ;   ã_r | 0 ].
```

The other three by the identical pattern:

```
d2Rot(a,c)      d/dξ[(d(R c)/dξ)ᵀ a]           = [ c̃·a_r̃    | 0 ;  0 | 0 ]
d2InvRot(a,u)   d/dξ[(d(Rᵀ u)/dξ)ᵀ a], u fixed  = [ ã·(R₀ᵀu)̃ | 0 ;  0 | 0 ]
d2InvPoint(a)   d/dξ[(d(Rᵀ(R c + p))/dξ)ᵀ a]   = [ ã·(R₀ᵀp₀)̃ | ã ;  0 | 0 ]
```

- `d2InvRot`: step (i) gives `[ â (Rᵀ u) ; 0 ]`; step (ii),
  `d(Rᵀu)/dξ = [ \widehat{R₀ᵀu} | 0 ]`.
- `d2InvPoint`: `Rᵀ(R c + p) = c + Rᵀ p`, and `c` is `ξ`-constant, so only
  `Rᵀ p` matters — a genuine product (`Rᵀ` *and* `p` move):
  `d(Rᵀ p)/dξ = [ \widehat{R₀ᵀ p₀} | I ]` (first term from `dInverseRotated`,
  second from `R₀ᵀ · ∂p/∂v = R₀ᵀ R₀ = I`). Step (i) gives `[ â (Rᵀp) ; a ]`;
  step (ii) differentiates the top block. The result is **independent of the
  local point `c`** — the structural reason a placed-point offset never reaches
  this derivative.

Each `*HessianFrozen` is a short linear combination of these. Example
(`sphereHessianFrozen`): the only ξ-dependent factor of `grad` is
`z_vecᵀ d(center)/dξ`, `center` = pose origin, so `H_frozen = d2Point(z_vec, 0)`.
`capsuleHessianFrozen` adds `t · d2Rot(z_vec, e₁)` for the axial term.

**The product-rule outer products (Cylinder / Cone / TruncatedCone caps).** The
end-cap orthant `q`-row is a *product of two* pose-dependent factors —
`bx(ξ) := R e₁` (cap normal) and `r(ξ) := p` (pose origin) — e.g.
`q_cap = ± bx·(r − p_dec) + (const)·α`. Its first derivative already needed the
product rule (`d(bx·r)/dξ = rᵀ dbx/dξ + bxᵀ dr/dξ`), so its second derivative
picks up, beyond the `d2*` terms, the two **constant** first-Jacobian outer
products

```
(∂bx/∂ξ)ᵀ (∂r/∂ξ)  +  (∂r/∂ξ)ᵀ (∂bx/∂ξ)
```

(`coneOrtQSecondDeriv` / the `BR` block in `cylinderHessianFrozen`). The ORT
contribution is weighted by the difference of the two cap multipliers
(`z_ort(2) − z_ort(3)` for Cylinder, `z_ort` for Cone, `z_ort(0) − z_ort(1)`
for TruncatedCone — the top cap's `q` is the base cap's negated).

#### `dn̂/dξ` (`computeContactNormalJacobian`)

`n̂ = u / ‖u‖`, `u := Rg · (dα/dξ)_v` (rotate the gradient's translation half
into shape 1's frame). Product rule on `u`:

```
du/dξ = dRotatedVectorDXi(g, (dα/dξ)_v)  +  Rg · (d²α/dξ²)_[v-rows] ,
```

first term for `Rg` moving, second for the gradient's translation rows moving
(this is exactly where the §4c second derivative is consumed). Then the
normalize-map Jacobian:

```
dn̂/dξ = (1/‖u‖) (I − n̂ n̂ᵀ) · du/dξ .
```

Verified against re-solved central FD. `computeContactJacobianBundle` produces
`jacobian`, `grad`, and `normal_jacobian` from one shared `xi_jac` + one IFT
solve; `hessianFrozenFull` is the only work unique to `normal_jacobian`.

---

## 5. New: contact layer (`contact.hpp`, `contact_degeneracy.hpp`, `contact_manifold.hpp`)

Julia returns one witness point and its Jacobian. This layer adds the contact
normal and, for degenerate contacts, degeneracy diagnostics and a multi-point
witness set with per-point Jacobians.

### 5a. `contactDegeneracy` — when a Jacobian is undefined

At a degenerate contact (parallel faces, aligned edges, matched vertices), `α`,
the witness point, and the normal are still correct as *values*, but their
Jacobians can fail. Two independent rank tests on matrices the solver already
holds at the converged `(x*, s*, z*)` — no re-solve, no shape geometry.

#### `contact_manifold_dim = nx − rank(A)`,  `A = Gᵀ(S⁻¹Z)G`

**Derivation.** `A` is precisely the matrix the IFT inverts for `dx*/dξ`
(§1: `A dx = r₁ − GᵀS⁻¹r₂`). Consider a primal displacement `δx` with `G δx`
lying in the currently-active constraint set:

- on an **active ORT row** `i` (`sᵢ = 0, zᵢ > 0`): `S⁻¹Z` has entry
  `zᵢ/sᵢ → ∞`, so `δxᵀ A δx` stays finite only if `(G δx)ᵢ = 0`;
- on an **active SOC block** (`s` on the cone boundary): `arw(s)⁻¹arw(z)` has
  one direction blowing up (the block's `λ₂ = s₀ − ‖s_tail‖ → 0`) and finite
  eigenvalues on the tangent directions.

So `ker(A) = { δx : G δx keeps every active constraint exactly active, in the
directions that carry infinite curvature }` — this is the **tangent space of
the optimal primal face** (the set of equally-optimal witness points). Hence

```
contact_manifold_dim = dim ker(A) = nx − rank(A) ,
```

`0` = the witness point is locally unique, `> 0` = a shared face/edge and the
witness rows of `jacobian` diverge as `pdip_tol → 0`. `α`'s row is the envelope
quantity and stays finite regardless.

Must use the **real `A`**, not an active-gradient count: an active SOC block's
`S⁻¹Z` carries finite tangential eigenvalues that a flat "count the active
gradients" rule would wrongly call free directions (naive count gives
cylinder–cylinder line contact 4 free directions; `A` gives 1).

#### `normal_cone_dim = m − rank(A_active)`

**Setup.** Stationarity `Σₖ zₖ Gₖ = −cᵀ` is the statement that `−c` lies in the
cone generated by the active constraint gradients; the contact normal is read
off `z*`. The normal (dual solution) is unique iff that generating set is
linearly independent on the active blocks.

`A_active` collects one row per active block:

- one row `Gᵢ` per active **ORT** constraint (`zᵢ > kActiveTol`);
- one row `z_blockᵀ G_block` per active **SOC** block — *one* row regardless of
  the block's size, because complementarity pins a bound SOC block's `z` to a
  single ray (its direction is fixed; only its magnitude is free).

`m` := number of active blocks (that row count). Then

```
normal_cone_dim = m − rank(A_active) ,
```

`> 0` ⟺ the active generators are dependent ⟺ the multiplier `z*` (hence the
normal) is not unique — a kink: a polytope edge or vertex. **Exact from a plain
rank count**: stationarity is linear in `z` regardless of surface curvature, so
no curvature-weighted matrix is needed here (unlike `A` above).

Verified on 8 hand-built configurations (`test_contact_degeneracy.cpp`),
including skew (non-parallel) edge–edge, which is *not* degenerate — the
criterion is `rank(A_active)`, not "a sharp feature is present". Rank via
`FullPivLU`; `A`'s spectrum spans ~9 orders of magnitude, so its threshold is
absolute (`kRelZeroTol·‖G‖²`, converted to LU's relative convention via
`maxPivot()`). Opt-in (`SocpOptions::compute_degeneracy_info`); when off, dims
are `-1` and `*_valid` is `true`.

**Strict-convexity fast path.** If either shape is strictly convex
(`IsStrictlyConvex<Shape>` — only `Sphere`/`Ellipsoid`), the contact is provably
`contact_manifold_dim == 0` and `normal_cone_dim == 0` (a strictly convex
boundary contains no segment; the intersection of a smooth body's single-ray
normal cone with anything is that ray). `proximityContactJacobian` skips the
`SVD`/`LU` entirely via `if constexpr`.

### 5b. `contactManifold` — multi-point witness sets

When `contact_manifold_dim > 0`, one point under-represents the contact for a
downstream resolver (the reason physics engines build contact manifolds). Built
from one primitive: a **closed-form ray clip** of `point + t·dir` against every
cone, along the null-space directions of `A` (the `V` columns from its SVD, the
one extra cost over `contactDegeneracy`).

**Derivation of the clip.** The slack at `point + t·dir` is affine in `t`:
`s(t) = s_here − t·(G dir)` (since `G(point + t dir) + s(t) = h`). Require every
block feasible:

- **ORT row `i`**: `sᵢ(t) ≥ 0` ⟺ `t ≤ s_here(i)/(Gᵢ·dir)` if `Gᵢ·dir > 0`,
  `t ≥ …` if `< 0` — a linear bound on the `[t_min, t_max]` interval.
- **SOC block**: `s₀(t) ≥ ‖s_tail(t)‖` ⟺ `s₀(t)² − ‖s_tail(t)²‖ ≥ 0` (with
  `s₀(t) ≥ 0`). Write `a := G_block·dir`, split `s_here`, `a` into head/tail.
  `s₀(t) = s₀ − t a₀`, `s_tail(t) = s_tail − t a_tail`. Expanding the quadratic
  in `t`:

  ```
  A_c t² + B_c t + C_c ≥ 0 ,
  A_c = a₀² − ‖a_tail‖² ,
  B_c = −2( s₀ a₀ − s_tail·a_tail ) ,
  C_c = s₀² − ‖s_tail‖² .
  ```

  For a **bounded** shape `A_c < 0` (the leading coefficient of the cone
  cross-section is negative), so the feasible set is the closed interval
  *between* the two roots `(−B_c ∓ √(B_c²−4A_cC_c)) / (2A_c)`; intersect it with
  `[t_min, t_max]`. `‖a‖ < 10⁻⁶·max(‖s_block‖, 1)` ⟹ skip this block: the
  **tangent-slide** direction (parallel cylinders/capsules sliding along their
  shared line), where `a` is roundoff, not zero, and a threshold on the
  *derived* `A_c` (natural magnitude `O(a²)`) would be dimensionally wrong.

**Using the clip.**

- **dim 1** — clip along `d₁` (smallest singular vector); the two endpoints
  `x + t_min d₁`, `x + t_max d₁` are exact and independent of where `x*` sits on
  the degenerate line (clipping ±d around any point on the line recovers the
  same absolute endpoints).
- **dim 2** — `x*` can sit off-center (the geometric init biases it). Recenter
  by two sequential 1D centerings: clip along `d₁`, move to the midpoint `c₁`;
  clip `c₁` along `d₂`, move to `c₂`. Then sample `M = max(2K, 8)` rays
  `cos θ·d₁ + sin θ·d₂` from `c₂`, take each forward hit, and reduce to `K`
  (default 4) by greedy farthest-point selection (maximize the minimum distance
  to already-picked points) so the output spans the patch instead of
  clustering.

Deliberately not the exact 2D intersection polygon: a cylinder cap on a
polytope face has a mixed straight/arc boundary that the ray clip handles with
no special-casing. Opt-in (`SocpOptions::compute_contact_manifold`).

### 5c. Per-point Jacobians — the affine transfer

`contact_manifold_point_jacobians` gives `{jacobian, normal_jacobian}` per
manifold point, under the **"same active set as `x*`"** convention: reuse
`s*, z*` unchanged, swap only the position. `z*` stays dual-feasible regardless
of position; `s` stays exactly zero on the active set along a null-space
direction. Recomputing `s = h − Gx` fresh is wrong — the manifold's boundary
points activate an extra constraint.

**Derivation.** Under this convention `A = Gᵀ(S⁻¹Z)G` is **frozen** (it depends
only on `s*, z*, G`, none of which change with the witness position), and the
IFT right-hand side `r₁ − GᵀS⁻¹r₂` is **affine in the position `P`** (`q`, hence
`r₁` and `r₂`, are affine in `x` — §4c — and `A⁻¹` is constant). A linear map of
an affine argument is affine, so the witness/`α` `jacobian` `J(P)` is an **exact
affine function of `P` on the patch.**

For a dim-2 patch, pick 3 non-collinear points `P₀, P₁, P₂`, do the full IFT
solve at each (`J₀, J₁, J₂`), set `E := [P₁−P₀ | P₂−P₀]` (3×2). Any further
point `P` has barycentric-style coordinates from the least-squares fit

```
[c₁; c₂] = (EᵀE)⁻¹ Eᵀ (P − P₀) ,
```

and, by affineness,

```
J(P) = J₀ + c₁ (J₁ − J₀) + c₂ (J₂ − J₀) .
```

So `dim 2` with `K > 3` needs only **3** full IFT solves; `dim 1` needs 2 (its
endpoints); `dim 0`'s point *is* `x*`, mirrored. Reproduces per-point
`diffSocp` exactly (regression-tested at `K = 8`); falls back to a full solve
per point if `P₀, P₁, P₂` are near-collinear.

`normal_jacobian` is **point-invariant** under this convention (it uses only the
gradient's translation columns, which are verified position-independent even on
a curved cylinder–cylinder line). Computed once, copied.

---

## 6. New: geometric initial guess (`socp_init.hpp`)

Julia's `initialize` is unconstrained least-squares + `bring2cone`,
unconditionally. DCOL++ keeps that as `SocpInitStrategy::Generic` and adds
`Geometric` (**the library default**), re-targeting the iDCOL manuscript's
cold-start (Sec. III.A/D) to DCOL's `[p; α; extras]` decision vector.

### 6a. Derivation of the primal seed

Each shape caches an inner and outer bounding-sphere radius about its local
origin (`r_in ≤ shape ≤ r_out`, `primitives.hpp`). Let `r = center₂ − center₁`,
`d = ‖r‖`, `r̂ = r/d`. Grow both shapes by a common factor `α` about a shared
point on the center line:

- The two **outer** spheres, scaled by `α`, first touch when
  `α(r₁_out + r₂_out) = d`, i.e. `α_min = d / (r₁_out + r₂_out)` — no contact is
  possible below this, so it lower-bounds `α*`.
- The two **inner** spheres touch at `α_max = d / (r₁_in + r₂_in)` — the shapes
  certainly overlap above this, so it upper-bounds `α*`.
- `α*` lies in `[α_min, α_max]`; take the **geometric mean**
  `α₀ = √(α_min α_max)` (scale-free midpoint — the bracket often spans a
  multiplicative factor, so the arithmetic mean would bias toward the larger
  end).
- The witness point sits on the center line at the scaled outer radius of
  shape 1: `p₀ = center₁ + α₀ r₁_out r̂`.

Exact for Sphere–Sphere (`α_min = α_max`, so `α₀ = d/(R₁+R₂)`). Degenerate
guard: if `α_max < α_min` (near-coincident centers), widen to
`α_max ← 2α_min + ε`.

### 6b. Derivation of the dual seed

`z` must satisfy dual feasibility `Gᵀz = −c` and prefer the cone interior.
Start from a preferred direction `z_pref` — each SOC block's **reflected slack
ray** `(0, −s_tail)` from `s = h − G x₀` (points into the cone), zero on ORT
(assumed inactive) — then project it onto the affine set `{z : Gᵀz = −c}`
orthogonally:

```
z₀ = z_pref + G (GᵀG)⁻¹ (−c − Gᵀ z_pref) .
```

Check: `Gᵀ z₀ = Gᵀz_pref + (GᵀG)(GᵀG)⁻¹(−c − Gᵀz_pref) = −c`. ✓ The `(GᵀG)⁻¹`
factorization is the same one the primal least-squares fit uses.

### 6c. Interior push (`pushToRelativeMargin`)

A touching contact's `s`/`z` sit exactly on the SOC boundary; a flat *absolute*
push leaves too thin a *relative* margin (NaN at the first NT step). Instead
push each SOC block to a fixed relative margin: require
`(u₀ − ‖u_tail‖)/u₀ ≥ margin_frac`, i.e. raise `u₀ ← ‖u_tail‖/(1 − margin_frac)`
when it is below (shipped at `margin_frac = 0.05`). Only under-margin blocks are
touched.

Every shape pair faster than Generic (~16% mean, full 100k-pose ergodic sweep),
0 wrong answers against a `1e-12` reference.

**Solver-hardening it forced — an improvement over Julia.** An externally
supplied `(s, z)` can satisfy `μ < pdip_tol` by algebraic alignment while `x` is
nowhere near `x*`, producing a *confidently wrong* answer. Julia's `solve_socp`
has the identical `μ`-only check but nothing external ever feeds it a hint.
`solver.hpp`'s iteration-1 convergence check now also requires `‖Gᵀz + c‖` and
`‖s + Gx − h‖` small (iteration 1 only — from iteration 2 on, `s, z` derive from
the previous residuals so `μ` shrinking is never divorced from them).

---

## 7. New: `TruncatedCone` primitive

Julia has `Cone`, not a frustum. `TruncatedCone(R_bottom, R_top, L)` brings the
shape count to 8. Local origin at the axial midpoint, axis `+x`, requires
`R_bottom > R_top ≥ 0`, `L > 0`.

### 7a. Derivation of `tan_beta` and `apex_dist`

The lateral surface is a right circular cone. Its radius as a function of the
axial coordinate `x ∈ [−L/2, +L/2]` drops linearly from `R_bottom` (at `−L/2`)
to `R_top` (at `+L/2`):

```
tan_beta = (R_bottom − R_top) / L        (radius change per unit length).
```

The **virtual apex** is where that radius hits zero. Continuing past `+L/2` at
slope `−tan_beta`, radius `R_top` reaches `0` after a further `R_top / tan_beta`,
so the apex is at axial coordinate `+L/2 + R_top/tan_beta` — its distance from
the (midpoint) origin is

```
apex_dist = L/2 + R_top / tan_beta = L/2 + R_top·L / (R_bottom − R_top) .
```

(`Cone`'s equivalent is `3H/4`: apex-to-centroid of a solid cone.)

### 7b. The SOC block and why the frustum reuses Cone's derivatives

The infinite-cone membership test for a point at local axial coord `x` and
radial part `ρ` is `ρ ≤ tan_beta·(x + apex_dist)`. In DCOL's scaled/posed SOC
form (`E = diag(tan_beta, 1, 1)`, moving point `p_dec` pulled back by
`Rᵀ(p_dec − p)`):

```
‖E Rᵀ(p_dec − p)‖  ≤  tan_beta·apex_dist·α  +  tan_beta·(Rᵀ(p_dec − p))_x ,
```

so `G_soc` is **byte-identical to `Cone`'s** except the constant α-coefficient
`G_soc(0, 3) = −tan_beta·apex_dist` (`Cone`: `−tan_beta·(3H/4)`). Being a
**constant**, it drops from every derivative — hence
`coneXiDerivative` / `coneHessianFrozen`'s SOC code is reused verbatim
(`coneSocSecondDeriv`, `coneOrtQSecondDeriv`).

The two flat caps at `x = ±L/2` are clipped by two ORT rows in Cylinder's `±bx`
pattern, `h_ort = (bx·p, −bx·p)`. Analytic derivatives reuse the Cylinder cap
formulas with the orthant contribution weighted by `z_ort(0) − z_ort(1)` (top
cap's `q` is the base cap's negated).

`Cone` stays separate rather than `TruncatedCone(R, 0, L)`: it is the
Julia-parity anchor, has a different origin convention (solid-cone centroid vs.
axial midpoint), and `R_top = 0` would put the tip cap exactly on the SOC apex.
Cost of the extra orthant row is ~2–4% per interior-point iteration.

---

## 8. New: warm-starting for temporally-continuous queries

A physics step, traj-opt inner loop, or smooth sweep queries the same pair at
poses that change little. Each proximity query (`proximity`, `alphaGradient`,
`proximityJacobian`, `proximityContactJacobian`) takes an optional
`ContactWarmState<S1,S2>*`; `nullptr` is byte-for-byte the cold path.
`solveForQuery` (`proximity.hpp`) owns the warm/cold logic for all four.

**Why the previous solution can't be reused raw.** A converged `(x*, s*, z*)`
has `s* ∘ z* = 0` — every block on a cone boundary, where the NT scaling's
spectral factorization is singular. A warm seed must be (1) strictly interior,
(2) as close to `(x*, s*, z*)` as (1) allows (iteration count ≈
`log₁₀(μ_start/pdip_tol)`), (3) dual-feasible for the *new* `G`
(`Gᵀz₀ = −c`, or `z`'s magnitude converges wrong).

### 8a. Derivation of the pose-move metric

Trust-region admission needs a scalar "how far did the pose move" on `SE(3)`.
With `D := g_ref⁻¹ g` (the relative motion since the reference solve):

```
poseMoveMetric = √( ‖D[0:3,0:3] − I₃‖_F²  +  ‖D[0:3,3]‖² ).
```

For small motion `D ≈ Exp([θ; Δt])`, `D[0:3,0:3] − I₃ ≈ θ̂`, and
`‖θ̂‖_F² = 2‖θ‖²`, so `poseMoveMetric ≈ √(2‖θ‖² + ‖Δt‖²)` — a smooth,
rotation-and-translation-aware distance that is `0` iff `g = g_ref`. `ρ` is a
trust radius on it: `ρ₀ = 0.02`, grow ×1.5 (cap 0.75) after a cheap warm solve,
shrink ×0.5 (floor 1e-3) after a fallback — a smooth trajectory ratchets `ρ` to
its cap while fast i.i.d. motion or an approaching degeneracy self-throttles
into cold solves.

**Per call:** if the handle is valid and `poseMoveMetric ≤ ρ`, seed and solve
(iteration-capped); keep the result if its KKT residuals pass
`resid_tol_mul · pdip_tol`, else redo cold.

### 8b. The two seeds (`socp_init.hpp`), `x₀ = x*` in both

- `warmStartInit` (`WarmSeed::Standard`, for `proximity` / `alphaGradient` /
  `proximityJacobian`): carry `(x*, z*)` forward with a per-block lift into the
  interior (`s ∘ z` non-uniform — fine for the value and the first-order
  Jacobian). Two tiers. `pose_move < 1e-6`: `s₀ = h − Gx*`, `z₀ = z*`, lifted
  only by a `~pdip_tol` safety floor — 1 iteration. Otherwise:
  `s₀ = h − Gx*`, `z₀ = z* + G(GᵀG)⁻¹(−c − Gᵀz*)` (**re-project** `z*` onto
  `{Gᵀz = −c}` for the *new* `G`, same projection as §6b), both lifted to a
  `1e-3` relative SOC margin — ~3–5 iterations. Gate: `1e4 · pdip_tol`.
- `warmStartInitCentral` (`WarmSeed::Central`, for `proximityContactJacobian`):
  `normal_jacobian`'s frozen Hessian needs `(s, z)` genuinely central
  (`s ∘ z ∝ e` uniformly). `s₀ = h_new − G_new·x*`,
  `z₀ = z* + G(GᵀG)⁻¹(−c − Gᵀz*)`; if either pokes outside the cone, a
  **uniform** `λe` lift (same `λ` on every block, so `s ∘ z` stays `∝ e`). Same
  stopping rule as cold (`μ` **and** residuals to `pdip_tol`).

### 8c. `warmRecenter` — the per-block lift

The ORT boundary is any coordinate at 0 (no intrinsic scale) → an **absolute**
floor `rᵢ ← max(rᵢ, φ_ort)`. The SOC boundary is `u₀ = ‖u_tail‖` and the block
*carries its own scale*, so use a **relative** margin: when
`(u₀ − ‖u_tail‖)/u₀ < φ_soc`, raise `u₀ ← ‖u_tail‖/(1 − φ_soc)`. Only
rows/blocks actually below the floor are touched — a uniform ORT lift like the
cold path's would pin `μ_start` and cost the whole warm-start win.

**Speed** (`bench_warm_start.cpp`): body at rest → 3.4–6.3×, one iteration; a
smooth correlated trajectory → 1.7–2.3× at every step size; fast i.i.d. motion →
break-even (never wrong). Smoothness, not step size, is what warm-start needs.

---

## 9. Performance vs Julia

Same protocol as `test_socp_julia_parity` (100k-pose ergodic sweep,
`pdip_tol = 1e-10`, each Julia pair its own OS process to isolate GC pressure),
clang/LLVM-mingw C++ with the geometric-init default vs. Julia:

| pair | C++ (µs) | Julia (µs) | ratio |
|---|---|---|---|
| SphereSphere | 2.60 | 4.74 | 0.55× |
| SphereCapsule | 4.34 | 6.12 | 0.71× |
| CapsuleCylinder | 6.94 | 9.66 | 0.72× |
| CylinderCone | 5.80 | 7.51 | 0.77× |
| ConePolytope | 3.29 | 4.07 | 0.81× |
| PolytopePolytope | 1.57 | 2.60 | 0.60× |
| PolytopeEllipsoid | 2.72 | 5.05 | 0.54× |
| EllipsoidPolygon | 6.38 | 8.91 | 0.72× |
| PolygonSphere | 5.22 | 8.52 | 0.61× |

**Mean ≈ 0.67× — DCOL++ ~50% faster on average.** The port started *slower* than
Julia; the gap closed and reversed via (a) forced inlining and scalar loops for
the tiny SOC-block algebra, (b) a hand-rolled fixed-size Cholesky (`SmallLLT`)
for the Newton system and SOC blocks, (c) switching GCC → clang/LLVM-mingw
(~1.1–1.5× on its own, zero source change), and (d) the geometric init. clang is
the recommended toolchain; GCC is a supported fallback (no compiler-specific
code).

Derivatives are validated against central FD rather than a second Julia run — a
stronger, autodiff-independent ground truth, and the only one that covers the
second derivative (which Julia lacks).

Test-RNG portability: `std::normal_distribution`'s algorithm is
implementation-defined, so GCC and clang draw different poses from the same seed.
`tests/portable_random.hpp` (`PortableNormal`, hand-rolled Box–Muller over
`mt19937`'s raw output) makes every trial's pose bit-identical across compilers.

---

## 10. Known limitations

- **Conditioning of `A` near touching.** `A = Gᵀ(S⁻¹Z)G` has `cond(A)` up to
  ~10¹³ when a SOC block sits on its boundary (`λ₂ = s₀ − ‖s_tail‖ → 0`) —
  which, for a formulation that finds the scaling at which shapes *just touch*,
  is every converged solution by construction. This is a structural property of
  any KKT-based differentiable optimization (cf. OptNet, cvxpylayers), not a
  port defect. The witness-point rows of the first-order Jacobian lose precision
  gracefully (`~roundoff·cond(A)`, always finite); `α`'s row (envelope) is
  unaffected. The Hessian and `dn̂/dξ` tests skip trials with per-block
  `min(λ₂) < 10⁻⁴` rather than loosen the tolerance.
- **Piecewise-smooth KKT map.** At genuine active-set degeneracy (polytope
  vertex/edge contacts) the solution map has a real kink: a finite bias to the
  objective leaves `α*` unchanged while the witness point jumps to another
  optimal point. What's computed there is a valid one-sided, branch-specific
  derivative — legitimate for one local optimization step, not a global unique
  derivative. `test_socp_julia_parity.cpp` requests `Generic` for one
  `PolygonSphere` case whose witness point (argmin non-unique, not solver error)
  doesn't retrace Julia's trajectory under `Geometric`.
- **Heuristic thresholds.** `contactDegeneracy`'s zero cutoffs are scale-aware
  but verified against a fixed set of configurations, not proven for arbitrary
  shape scale. Treat `*_valid == true` as "no degeneracy detected".
- The NT-rescaling identity that makes the Newton system Cholesky-solvable was
  not independently re-derived; correctness is confirmed end-to-end against a
  hand-computed ground truth.

---

## 11. Approaches tried and rejected

Recorded only so they aren't retried blindly; see git history for detail.

- **Surrogate scaling** (iDCOL Sec. III.B — rescale the translation so `α ~ 1`
  before solving). No benefit at normal dynamic range; at wide range a measured
  "speedup" was an artifact of a benchmark that checked only convergence, and
  the rescaled problem converged to a wrong point for some poses. Removed.
- **Explicit-inverse SOC arrow/NT matrices** instead of a Cholesky solve.
  Passed isolated checks (~1e-15) but `1/ρ` with `ρ = s₀² − ‖s_tail‖² → 0` at
  convergence corrupted `ds/dz` by `O(1)`. `SmallLLT` keeps factor-then-solve
  specifically to avoid this.
- **`-DEIGEN_DONT_VECTORIZE`** — largest single speedup measured, but a full
  build with it reproducibly fails one `test_socp_diff.cpp` case; a standalone
  repro of the same pose converges fine. Not shipped; off the critical path
  since the compiler switch.
- **Cholesky for the IFT solve.** `A = Gᵀ(S⁻¹Z)G` is symmetric only at the exact
  central path (`arw(s), arw(z)` commute there); away from it — and tests
  converge as loose as `pdip_tol = 1e-2` — it is materially non-symmetric, so
  `computeSocpSensitivity` uses a pivoted LU. Substituting the true NT scaling
  to force symmetry gives a wrong sensitivity (the choice `diff_socp` makes too).
- **`SelfAdjointEigenSolver` for `contactDegeneracy`'s `A`** — slower than
  `JacobiSVD` at 4×4–6×6 (fixed overhead dominates).
