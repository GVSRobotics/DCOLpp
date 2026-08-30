#pragma once
// dcolpp::socp -- primal-dual interior-point solver for second-order-cone
// programs
//     minimize    c'x
//     subject to  Gx + s = h,   s in K
// with K = R^{n_ort}_+ x Q^{n_soc1} x Q^{n_soc2} (nonnegative orthant times
// up to two second-order cones). Nesterov-Todd scaling, Mehrotra
// predictor-corrector step. All sizes (nx, n_ort, n_soc1, n_soc2) are
// compile-time template parameters.

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>

#include "dcolpp/socp/cone_utils.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/small_llt.hpp"

namespace dcolpp::socp {

template <int nx>
using DecisionVec = Vec<nx>;

template <int n_ort, int n_soc1, int n_soc2, int nx>
using ConstraintMat = Mat<n_ort + n_soc1 + n_soc2, nx>;

// Plain scalar loops for performance: avoids temp allocation and block
// expression overhead.
template <int n_ort, typename D1, typename D2>
DCOLPP_INLINE double ortLinesearch(const Eigen::MatrixBase<D1>& x, const Eigen::MatrixBase<D2>& dx) {
    double alpha = 1.0;
    for (int i = 0; i < n_ort; ++i) {
        if (dx(i) < 0.0) alpha = std::min(alpha, -x(i) / dx(i));
    }
    return alpha;
}

// Largest alpha in (0, 1] keeping y + alpha*delta inside the second-order
// cone Q = { (y0, y_v) : y0 >= ||y_v|| }, given y already strictly interior.
// Closed form (the cone boundary along a ray is a single quadratic):
//   nu   = y0^2 - ||y_v||^2            Lorentz form of y  (> 0 iff y in int Q)
//   zeta = y0*delta0 - y_v . delta_v   Lorentz inner product <y, delta>_L
// In coordinates scaled by sqrt(nu), split delta into a cone-radial part
//   rho0 = zeta / nu
// and a cone-tangential vector rho_v,
//   coeff    = (zeta/sqrt(nu) + delta0) / (y0/sqrt(nu) + 1)
//   rho_v[i] = delta_v[i]/sqrt(nu) - coeff * y_v[i]/nu
// The ray leaves the cone iff ||rho_v|| > rho0, crossing the boundary at
//   alpha = 1 / (||rho_v|| - rho0);
// otherwise the whole unit step is feasible (alpha = 1). Returned as
// min(1, alpha).
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

// Push r strictly inside the cone K = ORT x Q1 x Q2 by a uniform shift
// along the cone axis e (1 per ORT row, (1, 0..0) per SOC block).
//
// Per-block margin (how far inside its own cone r sits):
//   ORT : m_i = r_i                       (feasible iff r_i > 0)
//   SOC : m_k = r0^(k) - ||r_v^(k)||      (feasible iff r0 > ||r_v||)
// Let m = min of those margins over all blocks. Then
//   m  > 0 : r already strictly interior  -> return r
//   m <= 0 : return r + (1 - m) * e
// Adding t*e lifts every margin by exactly t, so t = 1 - m drives the
// worst margin to m + (1 - m) = 1: strictly interior, unit cushion. This
// is a one-off centering shift, unrelated to the step-length alpha in
// solveSocp.
template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> bring2cone(const StackVec<n_ort, n_soc1, n_soc2>& r) {
    double m = std::numeric_limits<double>::infinity(); // smallest per-block margin

    if constexpr (n_ort > 0) {
        m = std::min(m, r.template head<n_ort>().minCoeff());
    }
    if constexpr (n_soc1 > 0) {
        const Vec<n_soc1> r_soc1 = r.template segment<n_soc1>(n_ort);
        m = std::min(m, r_soc1(0) - r_soc1.template tail<n_soc1 - 1>().norm());
    }
    if constexpr (n_soc2 > 0) {
        const Vec<n_soc2> r_soc2 = r.template segment<n_soc2>(n_ort + n_soc1);
        m = std::min(m, r_soc2(0) - r_soc2.template tail<n_soc2 - 1>().norm());
    }

    if (m > 0.0) return r;
    return r + (1.0 - m) * gen_e<n_ort, n_soc1, n_soc2>();
}

template <int n_ort, int n_soc1, int n_soc2, int nx>
struct SocpInit {
    DecisionVec<nx> x;
    StackVec<n_ort, n_soc1, n_soc2> s;
    StackVec<n_ort, n_soc1, n_soc2> z;
};

// Cold starting triple (x, s, z) for solveSocp: drop the cone membership
// s,z in K and the complementarity s o z = 0, solve the leftover linear
// least-squares problem, then push s and z strictly interior. One Gram
// factor F = (G'G)^-1 is formed and reused for both solves.
//
//   x = (G'G)^-1 G'h          least-squares fit of  G x ~= h
//   s = bring2cone(G x - h)   its residual, then lifted into K
//
//   z_tilde = G (G'G)^-1 (-c) projection of -c onto range(G); this
//                             satisfies dual feasibility G'z + c = 0
//                             exactly (G' z_tilde = -c)
//   z = bring2cone(z_tilde)   lifted into K (may perturb G'z+c=0 off zero;
//                             the IP iterations restore it)
//
// The auxiliary x0 = (G'G)^-1 (-c) exists only to form z_tilde and is
// discarded -- the returned primal is the x from the first solve.
// See warm_start.hpp for a more sophisticated geometric initial guess that
// can be used instead of this one (default).
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
    // Complementarity gap (last `mu` on convergence, ~= pdip_tol). For
    // warm-starting via temporally-continuous queries (see warm_start.hpp).
    double mu = 0.0;
};

// Generic: least-squares fit + bring2cone. Geometric: shape-bounding-radii-seeded
// (see socp_init.hpp). proximity()/proximityJacobian()/
// proximityGradient() branch on this.
enum class SocpInitStrategy { Generic, Geometric };

struct SocpOptions {
    double pdip_tol = 1e-6;
    int max_iters = 50;
    // Geometric is the default: every pair found to be faster. Set to Generic
    // for bit-level reproducibility with Julia package, or as a fallback.
    SocpInitStrategy init_strategy = SocpInitStrategy::Geometric;
    // proximityContactJacobian (proximity_contact.hpp) only: opt-in switch
    // for ContactDegeneracy diagnostics.
    bool compute_degeneracy_info = false;
    // Optional multi-point contact manifold for degenerate contacts. When
    // contact_manifold_dim > 0, a single witness point under-represents a
    // shared edge/face; this returns 2 (line) or contact_manifold_points
    // (surface, default 4) points spanning it instead. Also populates the
    // degeneracy diagnostics. Default off.
    bool compute_contact_manifold = false;
    int contact_manifold_points = 4; // K for the 2D (surface) case; ignored for 0D/1D
    // NOTE on Jacobians with contact_manifold_dim > 0: contact_manifold_point_
    // jacobians are computed per point (moment-arm effects vary by location).
    // normal_jacobian is point-invariant (rigid translation shifts gap equally)
    // and thus reused for all manifold points. Computed under "same active set"
    // assumption: s* and z* from original solve are reused unchanged at every
    // point. This treats the manifold as one flat piece with x*'s active set,
    // which is the contract these analytic formulas require.
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

        // On iteration 1, an externally supplied hint can make s.z tiny
        // through algebraic alignment while x is still infeasible. So we
        // check residuals too. From iteration 2 on, mu shrinking implies
        // residuals shrink, making the extra check unnecessary. (The
        // warm-start caller in proximity.hpp re-verifies rx/rz at a looser
        // 1e4*pdip_tol after this returns; this gate is the strict one, and
        // also covers hint paths that have no downstream check, e.g.
        // SocpInitStrategy::Geometric.)
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
