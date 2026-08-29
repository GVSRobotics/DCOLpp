#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/solvers/coneqp/static_solver2.jl (Kevin Tracy, MIT License).
// See NOTICE.md at the repository root for full attribution.
//
// A primal-dual interior-point solver for second-order-cone programs
//     minimize    c'x
//     subject to  Gx + s = h,  s in K
// where K = R^{n_ort}_+ x Q^{n_soc1} x Q^{n_soc2} (nonnegative orthant times
// up to two second-order cones), using Nesterov-Todd scaling and a
// Mehrotra-style predictor-corrector step. All sizes (nx, n_ort, n_soc1,
// n_soc2) are compile-time template parameters, exactly mirroring how the
// Julia source carries them via StaticArrays type parameters.

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

#include "dcolpp/socp/cone_utils.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/small_llt.hpp"

namespace dcolpp::socp {

template <int nx>
using DecisionVec = Vec<nx>;

template <int n_ort, int n_soc1, int n_soc2, int nx>
using ConstraintMat = Mat<n_ort + n_soc1 + n_soc2, nx>;

// Plain scalar loops, not Eigen block/dot/norm expressions -- x and dx are
// typically Block views (from lineSearch's segment<>() calls below), and
// templating on the Eigen expression type here (instead of taking a
// concrete Vec<n_ort>&) means indexing them costs nothing beyond a strided
// read, no copy into a temporary. Measured: the equivalent Eigen-expression
// version (tail<>()/dot()/squaredNorm()/norm()) ran ~2.2x slower than
// Julia's StaticArrays here even at -O3 -- this mirrors SmallLLT, the one
// other hot-path block already at parity with Julia, being hand-scalar too.
template <int n_ort, typename D1, typename D2>
DCOLPP_INLINE double ortLinesearch(const Eigen::MatrixBase<D1>& x, const Eigen::MatrixBase<D2>& dx) {
    double alpha = 1.0;
    for (int i = 0; i < n_ort; ++i) {
        if (dx(i) < 0.0) alpha = std::min(alpha, -x(i) / dx(i));
    }
    return alpha;
}

template <int n_soc, typename D1, typename D2>
DCOLPP_INLINE double socLinesearch(const Eigen::MatrixBase<D1>& y, const Eigen::MatrixBase<D2>& delta) {
    static_assert(n_soc >= 1, "socLinesearch requires a nonempty SOC block");
    double yv_sq = 0.0, yv_dot_dv = 0.0;
    for (int i = 1; i < n_soc; ++i) {
        yv_sq += y(i) * y(i);
        yv_dot_dv += y(i) * delta(i);
    }
    const double nu = y(0) * y(0) - yv_sq;
    const double zeta = y(0) * delta(0) - yv_dot_dv;
    const double sqrt_nu = std::sqrt(nu);

    const double rho0 = zeta / nu;
    const double coeff = ((zeta / sqrt_nu) + delta(0)) / (y(0) / sqrt_nu + 1.0);
    double rho_tail_sq = 0.0;
    for (int i = 1; i < n_soc; ++i) {
        const double r = delta(i) / sqrt_nu - coeff * (y(i) / nu);
        rho_tail_sq += r * r;
    }
    const double rho_tail_norm = std::sqrt(rho_tail_sq);
    if (rho_tail_norm > rho0) {
        return std::min(1.0, 1.0 / (rho_tail_norm - rho0));
    }
    return 1.0;
}

template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE double lineSearch(const StackVec<n_ort, n_soc1, n_soc2>& x, const StackVec<n_ort, n_soc1, n_soc2>& dx) {
    double alpha = 1.0;
    if constexpr (n_ort > 0) {
        alpha = std::min(alpha, ortLinesearch<n_ort>(x.template head<n_ort>(), dx.template head<n_ort>()));
    }
    if constexpr (n_soc1 > 0) {
        alpha = std::min(alpha, socLinesearch<n_soc1>(x.template segment<n_soc1>(n_ort), dx.template segment<n_soc1>(n_ort)));
    }
    if constexpr (n_soc2 > 0) {
        alpha = std::min(alpha, socLinesearch<n_soc2>(x.template segment<n_soc2>(n_ort + n_soc1), dx.template segment<n_soc2>(n_ort + n_soc1)));
    }
    return alpha;
}

template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> bring2cone(const StackVec<n_ort, n_soc1, n_soc2>& r) {
    double alpha = -1.0;

    if constexpr (n_ort > 0) {
        const Vec<n_ort> r_ort = r.template head<n_ort>();
        if ((r_ort.array() <= 0.0).any()) alpha = -r_ort.minCoeff();
    }
    if constexpr (n_soc1 > 0) {
        const Vec<n_soc1> r_soc1 = r.template segment<n_soc1>(n_ort);
        const double res = r_soc1(0) - r_soc1.template tail<n_soc1 - 1>().norm();
        if (res <= 0.0) alpha = std::max(alpha, -res);
    }
    if constexpr (n_soc2 > 0) {
        const Vec<n_soc2> r_soc2 = r.template segment<n_soc2>(n_ort + n_soc1);
        const double res = r_soc2(0) - r_soc2.template tail<n_soc2 - 1>().norm();
        if (res <= 0.0) alpha = std::max(alpha, -res);
    }

    if (alpha < 0.0) return r;
    return r + (1.0 + alpha) * gen_e<n_ort, n_soc1, n_soc2>();
}

template <int n_ort, int n_soc1, int n_soc2, int nx>
struct SocpInit {
    DecisionVec<nx> x;
    StackVec<n_ort, n_soc1, n_soc2> s;
    StackVec<n_ort, n_soc1, n_soc2> z;
};

template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpInit<n_ort, n_soc1, n_soc2, nx> initializeSocp(const DecisionVec<nx>& c,
                                                    const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                    const StackVec<n_ort, n_soc1, n_soc2>& h) {
    const auto GT = G.transpose();
    SmallLLT<nx> F;
    F.compute(gramLower<n_ort + n_soc1 + n_soc2, nx>(G));

    const DecisionVec<nx> x_for_s = F.solve(DecisionVec<nx>(GT * h));
    const StackVec<n_ort, n_soc1, n_soc2> s_tilde = G * x_for_s - h;
    const StackVec<n_ort, n_soc1, n_soc2> s0 = bring2cone<n_ort, n_soc1, n_soc2>(s_tilde);

    const DecisionVec<nx> x0 = F.solve(DecisionVec<nx>(-c));
    const StackVec<n_ort, n_soc1, n_soc2> z_tilde = G * x0;
    const StackVec<n_ort, n_soc1, n_soc2> z0 = bring2cone<n_ort, n_soc1, n_soc2>(z_tilde);

    return SocpInit<n_ort, n_soc1, n_soc2, nx>{x_for_s, s0, z0};
}

template <int n_ort, int n_soc1, int n_soc2, int nx>
struct SocpResult {
    DecisionVec<nx> x;
    StackVec<n_ort, n_soc1, n_soc2> s;
    StackVec<n_ort, n_soc1, n_soc2> z;
    int iters = 0;
    bool converged = false;
    // Complementarity gap s.z/degree at the returned iterate (== the last
    // `mu` on a converged solve, ~= pdip_tol). Surfaced so a caller doing
    // temporally-continuous queries can decide whether to warm-start off
    // this solve and, if so, pick a re-centering target (warm_start.hpp).
    double mu = 0.0;
};

// Generic: solveSocp's own initializeSocp (unconstrained least-squares fit
// of x, then bring2cone) -- what this port originally shipped, matching
// DifferentiableCollisions.jl's own `initialize` (DEVIATIONS.md, "Unchanged
// from Julia"). Geometric: the shape-bounding-radii-seeded scheme
// (geometric_init.hpp, DEVIATIONS.md "geometric initial guess" -- a DCOL++
// addition, not present in Julia). proximity()/proximityJacobian()/
// proximityGradient() branch on this; solveSocp itself only ever sees
// whatever init_hint (or none) those functions pass it.
enum class SocpInitStrategy { Generic, Geometric };

struct SocpOptions {
    double pdip_tol = 1e-6;
    int max_iters = 50;
    // Geometric is the default: every pair faster (1.04x-1.45x, mean
    // ~17%), 0 wrong answers / 0 failures verified against a 1e-12-tol
    // reference across the full ergodic sweep and the existing test suite
    // (DEVIATIONS.md). Set to Generic to match Julia's own exact numerical
    // trajectory (e.g. bit-level parity checks) or as a fallback if a
    // specific pose/shape combination is ever found to need it.
    SocpInitStrategy init_strategy = SocpInitStrategy::Geometric;
    // proximityContactJacobian (proximity_contact.hpp) only: opt-in switch
    // for ContactDegeneracy diagnostics (contact_manifold_dim/
    // normal_cone_dim/witness_jacobian_valid/normal_jacobian_valid).
    // Default off -- measured at ~10-20% of that call's total cost (two
    // small JacobiSVDs), not negligible next to it. Off by default so the
    // Jacobian path's cost doesn't change for callers who don't ask for
    // this; set true to get the diagnostic fields populated.
    bool compute_degeneracy_info = false;
    // proximityContactJacobian only: opt-in switch for the multi-point
    // ContactManifold (proximity_contact.hpp) -- when contact_manifold_dim
    // > 0, a single witness point under-represents a shared edge/face; this
    // returns 2 (line) or contact_manifold_points (surface, default 4,
    // raise for finer resolution) points spanning it instead. Implies
    // compute_degeneracy_info's fields get populated too (computing the
    // manifold needs contact_manifold_dim anyway), so setting only this one
    // is enough -- but it's strictly more work than compute_degeneracy_info
    // alone (its own SVD-of-A additionally requests the right singular
    // vectors), so leave both off unless the manifold points themselves are
    // wanted. Default off.
    bool compute_contact_manifold = false;
    int contact_manifold_points = 4; // K for the 2D (surface) case; ignored for 0D/1D
    // NOTE on Jacobians once contact_manifold_dim > 0: the single jacobian/
    // normal_jacobian this solve already returns is evaluated at x* alone.
    // Once contact_manifold_dim > 0, x* is just one (generally off-center)
    // point among many valid ones, and grad = -q^T z is NOT point-invariant
    // across the manifold -- d(alpha)/d(rotation) genuinely varies point to
    // point (a moment-arm/lever effect: rotating shape 2 about its own
    // origin moves a far manifold point differently than a near one). So
    // whenever compute_contact_manifold is true, proximityContactJacobian
    // ALWAYS also fills in a per-point jacobian for every contact_manifold_
    // points entry (ProximityContactJacobianResult::contact_manifold_point_
    // jacobians) -- not a separate opt-in: if you're asking for the multi-
    // point manifold at all, you want each point's own Jacobian, the same
    // way a single-point query always gets its Jacobian.
    //
    // normal_jacobian is the one exception: d(alpha)/d(translation) (and
    // hence the normal direction and ITS Jacobian, which is built only from
    // that) IS point-invariant across the manifold -- a rigid translation of
    // shape 2 shifts the gap identically regardless of which manifold point
    // the solve happened to converge to, and this holds for curved manifolds
    // too (verified exactly, not just to noise, on axial-parallel Cylinder-
    // Cylinder, not only flat Polytope patches). So every per-point entry's
    // normal_jacobian is just the single already-computed value, copied, not
    // recomputed -- recomputing it per point would be pure waste (it's the
    // expensive Hessian-based part of the whole computation).
    //
    // Computed under the "same active set" assumption: s* and z* from the
    // original solve are reused UNCHANGED at every point, only x's position
    // is swapped in. This is deliberate, not a shortcut taken for speed --
    // recomputing s at a manifold point (h - Gx there) exposes that point's
    // OWN true active set, which for boundary/corner points (exactly what
    // ContactManifold returns: every point is ray-clipped out to where an
    // extra constraint becomes newly active) is larger than x*'s, breaking
    // the strict-complementarity premise the NT-scaling-based analytic
    // formulas assume -- verified to produce spurious, badly-off values.
    // Reusing s*,z* instead treats the whole manifold as one flat piece with
    // x*'s own (smaller, interior) active set throughout, which is exactly
    // the "piecewise-jacobian, same-active-set" contract these formulas were
    // ever valid under in the first place.
};

template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpResult<n_ort, n_soc1, n_soc2, nx> solveSocp(const DecisionVec<nx>& c,
                                                 const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                 const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                 const SocpOptions& opt = SocpOptions{},
                                                 const SocpInit<n_ort, n_soc1, n_soc2, nx>* init_hint = nullptr) {
    auto init = init_hint ? *init_hint : initializeSocp<n_ort, n_soc1, n_soc2, nx>(c, G, h);
    DecisionVec<nx> x = init.x;
    StackVec<n_ort, n_soc1, n_soc2> s = init.s;
    StackVec<n_ort, n_soc1, n_soc2> z = init.z;

    const StackVec<n_ort, n_soc1, n_soc2> e = gen_e<n_ort, n_soc1, n_soc2>();

    int cone_degree = n_ort;
    if constexpr (n_soc1 > 0) cone_degree += 1;
    if constexpr (n_soc2 > 0) cone_degree += 1;

    SocpResult<n_ort, n_soc1, n_soc2, nx> result;
    const Mat<nx, n_ort + n_soc1 + n_soc2> GT = G.transpose();

    for (int main_iter = 1; main_iter <= opt.max_iters; ++main_iter) {
        const double mu = s.dot(z) / static_cast<double>(cone_degree);
        const DecisionVec<nx> rx = GT * z + c;
        const StackVec<n_ort, n_soc1, n_soc2> rz = s + G * x - h;

        // mu=s.z/degree alone is a necessary but not sufficient convergence
        // proxy for the very FIRST check: an externally supplied (s,z) pair
        // (solveSocp's init_hint) can be constructed to make s.z tiny by
        // algebraic alignment without x actually being near-feasible
        // (rz=s+Gx-h measures exactly that gap) -- verified concretely: a
        // deliberately-aligned hint reported false convergence with alpha
        // off by O(1). From main_iter==2 onward this can't happen: s,z are
        // by then always derived from the *previous* iterate's rx,rz via
        // the Newton solve below, so mu shrinking is never divorced from
        // the residuals shrinking too -- same as with the untouched generic
        // (least-squares) init, which was never close enough to trip mu<tol
        // on iteration 1 in the first place. Scoping the extra check to
        // iteration 1 only (not a blanket added tolerance on every
        // iteration) avoids re-litigating rx/rz's natural convergence
        // scale against mu's -- an earlier attempt at an unconditional
        // check broke 12 existing tests on the untouched default path.
        const bool mu_ok = mu < opt.pdip_tol;
        const bool first_iter_residuals_ok = (main_iter > 1) || (rx.norm() < opt.pdip_tol && rz.norm() < opt.pdip_tol);
        if (mu_ok && first_iter_residuals_ok) {
            result.x = x;
            result.s = s;
            result.z = z;
            result.iters = main_iter;
            result.converged = true;
            result.mu = mu;
            return result;
        }

        NTScaling<n_ort, n_soc1, n_soc2> W = calcNTScalings<n_ort, n_soc1, n_soc2>(s, z);

        const StackVec<n_ort, n_soc1, n_soc2> lambda = W.apply(z);
        const StackVec<n_ort, n_soc1, n_soc2> lambda_lambda = cone_product<n_ort, n_soc1, n_soc2>(lambda, lambda);

        // affine step
        const DecisionVec<nx> bx = -rx;
        StackVec<n_ort, n_soc1, n_soc2> lambda_ds =
            inverse_cone_product<n_ort, n_soc1, n_soc2>(lambda, -lambda_lambda);
        StackVec<n_ort, n_soc1, n_soc2> bz_tilde = W.solve(-rz - W.apply(lambda_ds));
        ConstraintMat<n_ort, n_soc1, n_soc2, nx> Gt = W.template solveMat<nx>(G);
        // Lazy transpose view, not a materialized copy: GtT*bz_tilde below
        // computes directly off Gt's own (non-transposed) memory layout.
        const auto GtT = Gt.transpose();
        SmallLLT<nx> F;
        F.compute(gramLower<n_ort + n_soc1 + n_soc2, nx>(Gt));

        DecisionVec<nx> dxa = F.solve(DecisionVec<nx>(bx + GtT * bz_tilde));
        StackVec<n_ort, n_soc1, n_soc2> dza = W.solve(StackVec<n_ort, n_soc1, n_soc2>(Gt * dxa - bz_tilde));
        StackVec<n_ort, n_soc1, n_soc2> dsa = W.apply(lambda_ds - W.apply(dza));

        const double alpha_a = std::min(lineSearch<n_ort, n_soc1, n_soc2>(s, dsa),
                                         lineSearch<n_ort, n_soc1, n_soc2>(z, dza));
        const double rho = (s + alpha_a * dsa).dot(z + alpha_a * dza) / s.dot(z);
        const double rho_clamped = std::max(0.0, std::min(1.0, rho));
        const double sigma = rho_clamped * rho_clamped * rho_clamped;

        // centering + correcting step
        StackVec<n_ort, n_soc1, n_soc2> ds =
            -lambda_lambda - cone_product<n_ort, n_soc1, n_soc2>(W.solve(dsa), W.apply(dza)) + sigma * mu * e;
        lambda_ds = inverse_cone_product<n_ort, n_soc1, n_soc2>(lambda, ds);
        bz_tilde = W.solve(-rz - W.apply(lambda_ds));

        DecisionVec<nx> dx = F.solve(DecisionVec<nx>(bx + GtT * bz_tilde));
        StackVec<n_ort, n_soc1, n_soc2> dz = W.solve(Gt * dx - bz_tilde);
        StackVec<n_ort, n_soc1, n_soc2> ds_final = W.apply(lambda_ds - W.apply(dz));

        const double alpha = std::min(1.0, 0.99 * std::min(lineSearch<n_ort, n_soc1, n_soc2>(s, ds_final),
                                                             lineSearch<n_ort, n_soc1, n_soc2>(z, dz)));

        x += alpha * dx;
        s += alpha * ds_final;
        z += alpha * dz;

        result.iters = main_iter;
    }

    result.x = x;
    result.s = s;
    result.z = z;
    result.converged = false;
    result.mu = s.dot(z) / static_cast<double>(cone_degree);
    return result;
}

} // namespace dcolpp::socp
