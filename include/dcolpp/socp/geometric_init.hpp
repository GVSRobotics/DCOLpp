#pragma once
// Geometric initial guess for dcolpp::socp's PDIP solver -- an experiment,
// not (yet) the default. Replaces solver.hpp's `initializeSocp` (an
// unconstrained least-squares fit of x, then a generic `bring2cone` push)
// with one seeded from each shape's own inner/outer bounding-sphere radii,
// re-targeting the cold-start scheme from the iDCOL manuscript (Sec. III.A/D,
// eq. 11/14: `alpha_min = ||r||/(r1_out+r2_out)`, `alpha_max =
// ||r||/(r1_in+r2_in)`, `x0` placed on the outer sphere along `r_hat`) from
// iDCOL's single implicit scalar `phi` to DCOL's own `[p;alpha;extras]`
// decision vector and per-shape (G_ort,h_ort,G_soc,h_soc) representation.
// Only the primal `x` gets the geometric treatment; `z` (a Lagrange
// multiplier/contact-force-like quantity, not a spatial point) keeps the
// existing least-squares-then-`bring2cone` construction -- there is no
// obvious geometric analogue for it.
//
// This compares DCOL++ against itself (generic init vs. geometric init),
// not against Julia (DEVIATIONS.md §1c already settled that comparison).

#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

// BoundingSphere itself now lives in primitives.hpp, computed ONCE per
// shape in its own constructor and cached as a `const` member
// (shape.bounding_sphere) -- not recomputed per query the way this used to
// (measured cost before caching: ~1.5ns/call for the simple shapes, ~26ns
// for Ellipsoid's actual eigendecomposition; small, but real waste for a
// shape reused across many proximity() calls, which is the common case).
// This is a thin forwarder, kept so existing call sites (geometricPrimalGuess
// below, tools/bench_surrogate.cpp) don't need their own per-shape overload
// set -- one generic template covers every shape kind now that the
// computation itself lives with the shape, not here.
template <typename Shape>
inline const BoundingSphere& boundingSphere(const Shape& s) {
    return s.bounding_sphere;
}

// Per-shape extra decision-variable count, mirroring
// combineProblemMatrices' V1/V2 (V==4 <=> no extras). Specialized for the
// three shapes that carry one (Capsule/Cylinder: axial parameter) or two
// (Polygon: in-plane 2D coordinate).
template <typename Shape>
struct ExtraDim {
    static constexpr int value = 0;
};
template <>
struct ExtraDim<Capsule> {
    static constexpr int value = 1;
};
template <>
struct ExtraDim<Cylinder> {
    static constexpr int value = 1;
};
template <int NH>
struct ExtraDim<Polygon<NH>> {
    static constexpr int value = 2;
};

inline Eigen::Matrix<double, 1, 1> extrasGuess(const Capsule& c, const Eigen::Matrix4d& g, const Eigen::Vector3d& p0) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.Q * Eigen::Vector3d(1, 0, 0);
    const double half_l = c.L / 2.0;
    const double t = std::max(-half_l, std::min(half_l, bx.dot(p0 - pf.r)));
    return Eigen::Matrix<double, 1, 1>(t);
}

inline Eigen::Matrix<double, 1, 1> extrasGuess(const Cylinder& c, const Eigen::Matrix4d& g, const Eigen::Vector3d& p0) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.Q * Eigen::Vector3d(1, 0, 0);
    const double half_l = c.L / 2.0;
    const double t = std::max(-half_l, std::min(half_l, bx.dot(p0 - pf.r)));
    return Eigen::Matrix<double, 1, 1>(t);
}

template <int NH>
Eigen::Matrix<double, 2, 1> extrasGuess(const Polygon<NH>& poly, const Eigen::Matrix4d& g, const Eigen::Vector3d& p0) {
    const auto pf = placeShape(poly, g);
    const Eigen::Matrix<double, 3, 2> Qtilde = pf.Q.template block<3, 2>(0, 0);
    return Qtilde.transpose() * (p0 - pf.r);
}

// Builds the geometric [p;alpha;extras1;extras2] primal guess for the pair
// (shape1 @ identity, shape2 @ g), in the same column layout
// combineProblemMatrices uses.
template <typename Shape1, typename Shape2>
auto geometricPrimalGuess(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g) {
    constexpr int e1 = ExtraDim<Shape1>::value;
    constexpr int e2 = ExtraDim<Shape2>::value;
    constexpr int NX = 4 + e1 + e2;
    using XVec = Eigen::Matrix<double, NX, 1>;

    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto pf1 = placeShape(shape1, I4);
    const auto pf2 = placeShape(shape2, g);
    const BoundingSphere b1 = boundingSphere(shape1);
    const BoundingSphere b2 = boundingSphere(shape2);

    const Eigen::Vector3d rvec = pf2.r - pf1.r;
    const double dist = rvec.norm();
    const Eigen::Vector3d rhat = (dist > 1e-9) ? Eigen::Vector3d(rvec / dist) : Eigen::Vector3d(1, 0, 0);

    const double eps = 1e-9;
    const double alpha_min = dist / std::max(b1.outer + b2.outer, eps);
    double alpha_max = dist / std::max(b1.inner + b2.inner, eps);
    if (alpha_max < alpha_min) alpha_max = 2.0 * alpha_min + eps; // degenerate/near-coincident centers
    const double alpha0 = std::sqrt(std::max(alpha_min, eps) * std::max(alpha_max, eps));

    const Eigen::Vector3d p0 = pf1.r + (alpha0 * b1.outer) * rhat;

    XVec x0;
    x0.template head<3>() = p0;
    x0(3) = alpha0;
    if constexpr (e1 > 0) x0.template segment<e1>(4) = extrasGuess(shape1, I4, p0);
    if constexpr (e2 > 0) x0.template segment<e2>(4 + e1) = extrasGuess(shape2, g, p0);
    return x0;
}

// Assembles a full SocpInit (primal geometric; slack/dual both derived from
// it) for solveSocp's optional init_hint parameter. `x0` must already be in
// the combined problem's column layout (geometricPrimalGuess above).
//
// s0: the solver's own residual is `rz = s + Gx - h` (solver.hpp), so the
// candidate slack for a given x is `h - Gx`, not `Gx - h` -- getting this
// sign backwards (as an earlier version of this function did) silently
// negates the SOC blocks' vector components, discarding a good x0's
// informativeness entirely. bring2cone still applies: a touching contact's
// true s sits exactly ON the cone boundary, and the interior-point method
// needs a strictly-interior start.
//
// z0, take 2 -- KKT-consistent, not just sign-corrected.
//
// SOCP complementary slackness is `s o z = 0` in the Jordan algebra this
// codebase uses (cone_utils.hpp: `u o v = (u.v, u0*v_tail + v0*u_tail)`,
// dot including the u0*v0 term). For s on the SOC boundary (s0 =
// ||s_tail||), solving `s o z = 0` for z gives `z_tail = -(z0/s0)*s_tail`
// -- i.e. z lies along s's own boundary ray with its TAIL NEGATED
// (reflected through the cone axis): z = t*(s0,-s_tail) for some t>0, one
// free scalar per active SOC block. That gets the *direction* right but
// leaves the magnitude(s) `t` free -- and picking t=1 (an earlier version
// of this function) ignores a second, independent equation the true dual
// must also satisfy exactly: KKT stationarity, `G^T z* = -c`
// (solver.hpp's `rx = G^T z + c`, driven to 0). With one active SOC block
// per shape as the ansatz (ORT rows assumed inactive, z=0 there -- correct
// complementary-slackness value *if* that assumption holds), G^T z0 = -c
// becomes a tiny (nx x (1 or 2)) least-squares system in the t's: solving
// it pins the magnitude too, and -- since it's overdetermined (nx=4..6
// equations, 1-2 unknowns) -- its residual is a built-in trustworthiness
// check. Near-zero residual: the "each SOC-bearing shape's own membership
// constraint is the active one" assumption holds and z0 is essentially
// z*. Large residual: it doesn't (e.g. a capsule/cone endcap contact, not
// its lateral surface -- an ORT row is active instead, which this ansatz
// can't represent), and the caller should not trust this z0.
//
// Shapes with no SOC block at all anywhere in the pair (Polytope-Polytope)
// have no ray to build an ansatz from; reported untrustworthy immediately.
//
// Interior-margin push, take 2: bring2cone's fixed "+1.0*e" is calibrated
// for the *generic* least-squares init, which is never close to exact, so
// a flat absolute push safely lands it well inside the cone. The geometric
// guess can be exact in BOTH s0 and z0 simultaneously (a touching
// contact's true solution sits exactly on the SOC boundary in both), and
// pushing each by the same flat +1.0 leaves too thin a *relative* margin
// -- observed directly: a case with x0=x* and z0=z* exactly still
// produced NaN at the very first PDIP iteration (near-singular NT
// scaling), despite mu0 being unremarkable (~0.8, smaller than plenty of
// converging cases). Root cause isn't "not enough progress possible", it's
// the starting NT-scaling itself breaking down. Fix: push by whatever's
// needed so each SOC block clears a fixed *relative* margin,
// (s0-||s_tail||)/s0 >= margin_frac, rather than a fixed absolute amount
// -- verified on the full 100k-pose SphereSphere sweep (where x0,z0 are
// exact for every pose, the worst case for this failure mode): 0/100000
// failures from margin_frac=0.01 all the way down, vs. 858/100000 with
// bring2cone's flat push, and materially fewer PDIP iterations besides
// (avg ~5.0 at margin_frac=0.001 vs. bring2cone's ~9.6 with the generic
// init). margin_frac=0.05 here trades a little of that for headroom on
// pairs not stress-tested at the very edge.
template <int n_ort, int n_soc1, int n_soc2>
StackVec<n_ort, n_soc1, n_soc2> pushToRelativeMargin(const StackVec<n_ort, n_soc1, n_soc2>& r, double margin_frac) {
    double lambda = 0.0;
    if constexpr (n_ort > 0) {
        const double min_ort = r.template head<n_ort>().minCoeff();
        lambda = std::max(lambda, 0.05 - min_ort); // ORT has no self-scale to be relative to; small absolute floor
    }
    if constexpr (n_soc1 > 0) {
        const Vec<n_soc1> r1 = r.template segment<n_soc1>(n_ort);
        const double tail_norm = r1.template tail<n_soc1 - 1>().norm();
        lambda = std::max(lambda, tail_norm / (1.0 - margin_frac) - r1(0));
    }
    if constexpr (n_soc2 > 0) {
        const Vec<n_soc2> r2 = r.template segment<n_soc2>(n_ort + n_soc1);
        const double tail_norm = r2.template tail<n_soc2 - 1>().norm();
        lambda = std::max(lambda, tail_norm / (1.0 - margin_frac) - r2(0));
    }
    if (lambda <= 0.0) return r;
    return StackVec<n_ort, n_soc1, n_soc2>(r + lambda * gen_e<n_ort, n_soc1, n_soc2>());
}
// z0, take 4 -- least-squares PROJECTION, not a restricted ansatz.
//
// History: two earlier designs were tried and superseded, not just
// patched, because they shared the same structural flaw -- both hand-pick
// which blocks are "the active set" (every existing SOC block; every ORT
// row assumed inactive) and solve for a magnitude *within* that fixed
// ansatz. That works well when the assumption holds (Sphere: unconditionally,
// since a sphere's SOC block IS its whole membership constraint) and badly
// when it doesn't (Capsule/Cone endcap contact, Polygon edge contact --
// cases where an ORT row is genuinely active too, which the ansatz has no
// way to represent). That forced a residual-based "trustworthy" gate plus
// a fallback to the fully x0-*unaware* generic dual whenever the gate
// failed -- and the gate itself needed careful, shape-specific tuning
// (PolygonSphere measured ~100% falsely "trustworthy" at a loose
// kResidualTol=0.15, regressing that pair to 0.58x before tightening to
// 0.005 fixed it -- see git history of this file).
//
// This version drops the ansatz restriction entirely. Build a *preferred*
// direction z_pref -- each existing SOC block's reflected ray (tail
// negated per the s-o-z=0 Jordan-algebra derivation), zero on ORT rows --
// then PROJECT it exactly onto the KKT stationarity constraint
// {z : G^T z = -c} using the same (G^T G)^{-1} machinery already computed
// for x0's own least-squares fit:
//
//     z = z_pref + G (G^T G)^{-1} (-c - G^T z_pref)
//
// This satisfies G^T z = -c EXACTLY, for ANY z_pref, by construction --
// no residual check needed, because the projection unconditionally
// supplies whatever correction feasibility requires (verify: G^T z =
// G^T z_pref + G^T G (G^T G)^{-1}(-c - G^T z_pref) = G^T z_pref + (-c -
// G^T z_pref) = -c). When z_pref already happens to be exact (Sphere: the
// correction term is exactly zero) nothing is lost; when it doesn't
// (Capsule/Cone/Polygon), the projection blends toward feasibility across
// ALL blocks instead of the old binary "trust it fully or discard it
// fully" choice a residual-gated ansatz forced.
// No mu0-targeting either: pushed into the cone with the same relative
// margin as s0 (pushToRelativeMargin) and left to the solver's normal
// pdip_tol-driven iteration -- simpler, and empirically both faster AND
// more uniform across all 9 shape pairs than the ansatz+mu0+fallback
// design this superseded (every pair 1.04x-1.43x faster than the generic
// init, vs. the old design's 0.99x-8.77x spread; 0 wrong answers, 0
// failures against a 1e-12-tol reference across the full ergodic sweep).
//
// Known limitation, tried and NOT fixed: giving ORT rows an informative
// z_pref too (elementwise 1/max(s_tilde_i,floor), same complementary-
// slackness intuition as the SOC reflection) fixes the one reference case
// this path is currently known to marginally miss (below) but, tuned
// against that case, pushes a *different* reference case (PolytopeEllipsoid
// #4) over the same tolerance instead -- whack-a-mole across the existing
// Julia-parity fixture, not a real fix, for every floor value tried
// ({0.05, 0.1, 0.2, 0.5, 1.0} and several capped variants). A warm-restart
// "polish" pass (re-solving from the converged (x,s,z) as a fresh hint) was
// also tried and does nothing -- solveSocp's own convergence check (mu AND
// the actual KKT residuals small, solver.hpp) already accepts the first
// pass's point as genuinely converged, so a second pass exits at iteration
// 1 with an identical answer. Reverted to the simpler zero-on-ORT z_pref
// above (only one known regression, not a shifting one).
//
// The one known regression: tests/test_socp_julia_parity.cpp, PolygonSphere
// reference case 3 -- alpha matches to ~1e-10 (well inside its own 1e-6
// tolerance) but the witness point misses the 1e-5 tolerance by about 50%
// (~1.6e-5). Traced to a genuinely active ORT row (the converged in-plane
// coordinate sits exactly on its scaled polygon-face bound,
// 0.4*alpha) that this path's zero-on-ORT z_pref has no information about,
// converging along a different (still KKT-valid to pdip_tol=1e-10) path
// than Julia's own solver -- not a wrong answer, a different valid point
// within the tolerance ball in a weakly-determined direction, of the same
// character as the near-degenerate sensitivity DEVIATIONS.md §5/§6a
// already documents on the dual/Hessian side.
template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpInit<n_ort, n_soc1, n_soc2, nx> initializeSocpFromGuess(const DecisionVec<nx>& c,
                                                             const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                             const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                             const DecisionVec<nx>& x0) {
    using Z = StackVec<n_ort, n_soc1, n_soc2>;
    constexpr double kMarginFrac = 0.05;

    const Z s_tilde = h - G * x0;
    const Z s0 = pushToRelativeMargin<n_ort, n_soc1, n_soc2>(s_tilde, kMarginFrac);

    Z z_pref = Z::Zero(); // ORT rows: see this function's own comment above -- z_pref_ort=0
                          // ("assume inactive") kept; an elementwise-informed variant was
                          // tried and reverted (see git history / DEVIATIONS.md).
    if constexpr (n_soc1 > 0) {
        Vec<n_soc1> blk = s_tilde.template segment<n_soc1>(n_ort);
        blk.template tail<n_soc1 - 1>() *= -1.0;
        z_pref.template segment<n_soc1>(n_ort) = blk;
    }
    if constexpr (n_soc2 > 0) {
        Vec<n_soc2> blk = s_tilde.template segment<n_soc2>(n_ort + n_soc1);
        blk.template tail<n_soc2 - 1>() *= -1.0;
        z_pref.template segment<n_soc2>(n_ort + n_soc1) = blk;
    }

    SmallLLT<nx> F;
    F.compute(gramLower<n_ort + n_soc1 + n_soc2, nx>(G));
    const DecisionVec<nx> w = F.solve(DecisionVec<nx>(-c - G.transpose() * z_pref));
    const Z z_exact = Z(z_pref + G * w);
    const Z z0 = pushToRelativeMargin<n_ort, n_soc1, n_soc2>(z_exact, kMarginFrac);

    return SocpInit<n_ort, n_soc1, n_soc2, nx>{x0, s0, z0};
}

} // namespace dcolpp::socp
