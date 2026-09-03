#pragma once
// dcolpp::socp::runtime_poly -- primal-dual interior-point SOCP solver for the
// PolytopeX path. This is the ORT-only case (n_soc = 0, nx = 4): a
// runtime-row-count copy of solver.hpp's solveSocp with the SOC branches
// dropped and the Nesterov-Todd scaling collapsed to the diagonal
// w = sqrt(s./z). Operation order matches solver.hpp 1:1 so a PolytopeX pair
// reproduces the fixed Polytope<NH> pair bit-for-bit.
//
// M3 will add a single fixed SOC block (PolytopeX vs a primitive); the ORT
// machinery here is unchanged by that.

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "dcolpp/socp/runtime_poly/types.hpp"
#include "dcolpp/socp/small_llt.hpp" // dcolpp::socp::SmallLLT<4>
#include "dcolpp/socp/solver.hpp"    // dcolpp::socp::SocpOptions / SocpInitStrategy

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::SmallLLT;
using dcolpp::socp::SocpInitStrategy;
using dcolpp::socp::SocpOptions;

// --- ORT cone helpers (runtime length) -----------------------------------

// Largest step in (0,1] keeping x + a*dx >= 0 elementwise (x already > 0).
inline double ortLinesearch(const StackVecX& x, const StackVecX& dx) {
    double a = 1.0;
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i)
        if (dx(i) < 0.0) a = std::min(a, -x(i) / dx(i));
    return a;
}

// bring2cone (solver.hpp): if the worst margin m <= 0, lift every row by
// (1 - m) so the worst margin becomes 1 (unit cushion).
inline StackVecX bring2cone(const StackVecX& r) {
    const double m = r.minCoeff();
    if (m > 0.0) return r;
    return (r.array() + (1.0 - m)).matrix();
}

// pushToRelativeMargin (socp_init.hpp), ORT branch only: small absolute floor
// (0.05) on the least row; SOC terms and margin_frac don't apply here.
inline StackVecX pushToRelativeMargin(const StackVecX& r) {
    const double lambda = std::max(0.0, 0.05 - r.minCoeff());
    if (lambda <= 0.0) return r;
    return (r.array() + lambda).matrix();
}

// G'G (4x4). SmallLLT<4>::compute reads only the lower triangle; the product
// is symmetric so the whole matrix is fine to hand it.
inline Eigen::Matrix4d gram4(const ConstraintMatX<4>& G) { return G.transpose() * G; }

// --- init (cold) --------------------------------------------------------

struct SocpInitX {
    Eigen::Vector4d x;
    StackVecX s;
    StackVecX z;
};

// solver.hpp initializeSocp (Generic strategy): least-squares x, bring2cone
// residuals into the cone; one Gram factor reused for both solves.
inline SocpInitX initializeSocp(const Eigen::Vector4d& c, const ConstraintMatX<4>& G, const StackVecX& h) {
    const auto GT = G.transpose();
    SmallLLT<4> F;
    F.compute(gram4(G));

    const Eigen::Vector4d x_for_s = F.solve(Eigen::Vector4d(GT * h));
    const StackVecX s0 = bring2cone(StackVecX(G * x_for_s - h));

    const Eigen::Vector4d x0 = F.solve(Eigen::Vector4d(-c));
    const StackVecX z0 = bring2cone(StackVecX(G * x0));

    return {x_for_s, s0, z0};
}

// socp_init.hpp geometricPrimalGuess for PolytopeX vs PolytopeX (both
// non-halfspace, no extras -> NX = 4): seed [p; alpha] from the two shapes'
// bounding-sphere radii.
inline Eigen::Vector4d geometricPrimalGuess(const BoundingSphere& b1, const BoundingSphere& b2,
                                            const Eigen::Matrix4d& g) {
    const Eigen::Vector3d r1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d r2 = g.block<3, 1>(0, 3);
    const double eps = 1e-9;

    const Eigen::Vector3d rvec = r2 - r1;
    const double dist = rvec.norm();
    const Eigen::Vector3d rhat = (dist > 1e-9) ? Eigen::Vector3d(rvec / dist) : Eigen::Vector3d(1, 0, 0);

    const double alpha_min = dist / std::max(b1.outer + b2.outer, eps);
    double alpha_max = dist / std::max(b1.inner + b2.inner, eps);
    if (alpha_max < alpha_min) alpha_max = 2.0 * alpha_min + eps;
    const double alpha0 = std::sqrt(std::max(alpha_min, eps) * std::max(alpha_max, eps));

    Eigen::Vector4d x0;
    x0.head<3>() = r1 + (alpha0 * b1.outer) * rhat;
    x0(3) = alpha0;
    return x0;
}

// socp_init.hpp geometricPrimalGuess, IsHalfspace<Shape1> branch: fixed plane
// n.p = d, only body 2 scales. b2 = shape 2's bounding sphere.
inline Eigen::Vector4d geometricPrimalGuessPlane(const Eigen::Vector3d& n, double d, const BoundingSphere& b2,
                                                 const Eigen::Matrix4d& g) {
    const Eigen::Vector3d r2 = g.block<3, 1>(0, 3);
    const double eps = 1e-9;
    const double sd = std::abs(n.dot(r2) - d);
    const double a_lo = sd / std::max(b2.outer, eps);
    double a_hi = sd / std::max(b2.inner, eps);
    if (a_hi < a_lo) a_hi = 2.0 * a_lo + eps;
    const double a0 = std::sqrt(std::max(a_lo, eps) * std::max(a_hi, eps));

    Eigen::Vector4d x0;
    x0.head<3>() = r2 - (n.dot(r2) - d) * n;
    x0(3) = a0;
    return x0;
}

// socp_init.hpp initializeSocpFromGuess, ORT-only: s0 from the primal
// residual, z0 = pushToRelativeMargin(G (G'G)^-1 (-c))  (z_pref = 0).
inline SocpInitX initializeSocpFromGuess(const Eigen::Vector4d& c, const ConstraintMatX<4>& G, const StackVecX& h,
                                         const Eigen::Vector4d& x0) {
    const StackVecX s0 = pushToRelativeMargin(StackVecX(h - G * x0));

    SmallLLT<4> F;
    F.compute(gram4(G));
    const Eigen::Vector4d w = F.solve(Eigen::Vector4d(-c));
    const StackVecX z0 = pushToRelativeMargin(StackVecX(G * w));

    return {x0, s0, z0};
}

// --- solve ------------------------------------------------------------

struct SocpResultX {
    Eigen::Vector4d x = Eigen::Vector4d::Zero();
    StackVecX s;
    StackVecX z;
    int iters = 0;
    bool converged = false;
    double mu = 0.0;
};

// solver.hpp solveSocp, ORT-only. W is the diagonal NT scaling w = sqrt(s./z);
// W.apply(g) = g.*w, W.solve(g) = g./w, W.solveMat(G) = (1./w) row-scaled G.
inline SocpResultX solveSocp(const Eigen::Vector4d& c, const ConstraintMatX<4>& G, const StackVecX& h,
                             const SocpOptions& opt = SocpOptions{}, const SocpInitX* init_hint = nullptr) {
    const SocpInitX init = init_hint ? *init_hint : initializeSocp(c, G, h);
    Eigen::Vector4d x = init.x;
    StackVecX s = init.s;
    StackVecX z = init.z;

    const int ns = static_cast<int>(h.size());
    const StackVecX e = StackVecX::Ones(ns);
    const double cone_degree = static_cast<double>(ns);
    const auto GT = G.transpose();

    SocpResultX result;
    result.s = s;
    result.z = z;

    for (int main_iter = 1; main_iter <= opt.max_iters; ++main_iter) {
        const double mu = s.dot(z) / cone_degree;
        const Eigen::Vector4d rx = GT * z + c;
        const StackVecX rz = s + G * x - h;

        const bool mu_ok = mu < opt.pdip_tol;
        const bool first_iter_residuals_ok =
            (main_iter > 1) || (rx.norm() < opt.pdip_tol && rz.norm() < opt.pdip_tol);
        if (mu_ok && first_iter_residuals_ok) {
            result.x = x;
            result.s = s;
            result.z = z;
            result.iters = main_iter;
            result.converged = true;
            result.mu = mu;
            return result;
        }

        const StackVecX w = (s.cwiseQuotient(z)).cwiseSqrt();
        const auto Wapply = [&](const StackVecX& g) -> StackVecX { return g.cwiseProduct(w); };
        const auto Wsolve = [&](const StackVecX& g) -> StackVecX { return g.cwiseQuotient(w); };

        const StackVecX lambda = Wapply(z);
        const StackVecX lambda_lambda = lambda.cwiseProduct(lambda);

        // affine step
        const Eigen::Vector4d bx = -rx;
        StackVecX lambda_ds = (-lambda_lambda).cwiseQuotient(lambda); // inverse_cone_product(lambda, -ll)
        StackVecX bz_tilde = Wsolve(StackVecX(-rz - Wapply(lambda_ds)));
        ConstraintMatX<4> Gt = w.cwiseInverse().asDiagonal() * G; // W \ G
        const auto GtT = Gt.transpose();
        SmallLLT<4> F;
        F.compute(Eigen::Matrix4d(Gt.transpose() * Gt));

        Eigen::Vector4d dxa = F.solve(Eigen::Vector4d(bx + GtT * bz_tilde));
        StackVecX dza = Wsolve(StackVecX(Gt * dxa - bz_tilde));
        StackVecX dsa = Wapply(StackVecX(lambda_ds - Wapply(dza)));

        const double alpha_a = std::min(ortLinesearch(s, dsa), ortLinesearch(z, dza));
        const double rho = (s + alpha_a * dsa).dot(z + alpha_a * dza) / s.dot(z);
        const double rho_clamped = std::max(0.0, std::min(1.0, rho));
        const double sigma = rho_clamped * rho_clamped * rho_clamped;

        // centering + correcting step
        StackVecX ds = -lambda_lambda - Wsolve(dsa).cwiseProduct(Wapply(dza)) + sigma * mu * e;
        lambda_ds = ds.cwiseQuotient(lambda); // inverse_cone_product(lambda, ds)
        bz_tilde = Wsolve(StackVecX(-rz - Wapply(lambda_ds)));

        Eigen::Vector4d dx = F.solve(Eigen::Vector4d(bx + GtT * bz_tilde));
        StackVecX dz = Wsolve(StackVecX(Gt * dx - bz_tilde));
        StackVecX ds_final = Wapply(StackVecX(lambda_ds - Wapply(dz)));

        const double alpha =
            std::min(1.0, 0.99 * std::min(ortLinesearch(s, ds_final), ortLinesearch(z, dz)));

        x += alpha * dx;
        s += alpha * ds_final;
        z += alpha * dz;
        result.iters = main_iter;
    }

    result.x = x;
    result.s = s;
    result.z = z;
    result.converged = false;
    result.mu = s.dot(z) / cone_degree;
    return result;
}

// Cold solve with strategy pick (solveProximitySocp, minus warm-start).
inline SocpResultX solveProximitySocp(const BoundingSphere& b1, const BoundingSphere& b2, const Eigen::Matrix4d& g,
                                      const Eigen::Vector4d& c, const ConstraintMatX<4>& G, const StackVecX& h,
                                      const SocpOptions& opt) {
    if (opt.init_strategy == SocpInitStrategy::Geometric) {
        const Eigen::Vector4d x0 = geometricPrimalGuess(b1, b2, g);
        const SocpInitX init = initializeSocpFromGuess(c, G, h, x0);
        return solveSocp(c, G, h, opt, &init);
    }
    return solveSocp(c, G, h, opt);
}

// Plane (shape 1) vs hull (shape 2): the halfspace geometric guess.
inline SocpResultX solveProximitySocpPlane(const Eigen::Vector3d& n, double d, const BoundingSphere& b2,
                                           const Eigen::Matrix4d& g, const Eigen::Vector4d& c,
                                           const ConstraintMatX<4>& G, const StackVecX& h, const SocpOptions& opt) {
    if (opt.init_strategy == SocpInitStrategy::Geometric) {
        const Eigen::Vector4d x0 = geometricPrimalGuessPlane(n, d, b2, g);
        const SocpInitX init = initializeSocpFromGuess(c, G, h, x0);
        return solveSocp(c, G, h, opt, &init);
    }
    return solveSocp(c, G, h, opt);
}

} // namespace dcolpp::socp::runtime_poly
