#pragma once
// dcolpp::socp -- the proximity query: minimum uniform scaling `alpha`
// (< 1 penetrating, == 1 touching, > 1 separated) and the witness point,
// both in shape 1's frame, and their derivatives w.r.t. shape 2's relative
// pose (analytic_derivatives.hpp).
//
//   proximity          -- alpha, witness (one SOCP solve)
//   alphaGradient      -- + d(alpha)/dxi   (O(1) envelope gradient)
//   proximityJacobian  -- + d[witness;alpha]/dxi   (the full IFT solve)
//
// The contact normal and its Jacobian, and degenerate-contact handling,
// live in contact.hpp on top of these.

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp"
#include "dcolpp/socp/combine_problem_matrices.hpp"
#include "dcolpp/socp/socp_init.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/solver.hpp"
#include "dcolpp/socp/warm_start.hpp"

namespace dcolpp::socp {

// Shared by proximity()/proximityJacobian()/alphaGradient(): branches on
// opt.init_strategy so solveSocp only ever sees a plain (c,G,h,opt) call or
// one with an explicit init_hint.
template <int n_ort, int n_soc1, int n_soc2, int nx, typename Shape1, typename Shape2>
SocpResult<n_ort, n_soc1, n_soc2, nx> solveProximitySocp(const Shape1& shape1, const Shape2& shape2,
                                                          const Eigen::Matrix4d& g, const DecisionVec<nx>& c,
                                                          const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                          const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                          const SocpOptions& opt) {
    if (opt.init_strategy == SocpInitStrategy::Geometric) {
        const auto x0 = geometricPrimalGuess(shape1, shape2, g);
        const auto init = initializeSocpFromGuess<n_ort, n_soc1, n_soc2, nx>(c, G, h, x0);
        return solveSocp<n_ort, n_soc1, n_soc2, nx>(c, G, h, opt, &init);
    }
    return solveSocp<n_ort, n_soc1, n_soc2, nx>(c, G, h, opt);
}

struct ProximityResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero(); // reference (shape-1) frame
    int iters = 0;
    bool converged = false;
};

template <typename Shape1, typename Shape2>
ProximityResult proximity(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                           const SocpOptions& opt = SocpOptions{}) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    const auto P1 = problemMatrices(shape1, I4);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(shape1, shape2, g, combined.c, combined.G, combined.h, opt);

    ProximityResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    return res;
}

struct AlphaGradientResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 1, 6> grad = Eigen::Matrix<double, 1, 6>::Zero(); // d(alpha)/dxi, xi=[w;v]
    int iters = 0;
    bool converged = false;
};

// Solve + d(alpha)/dxi only -- the envelope-theorem gradient
// (computeProximityGradient), O(1) after the solve. Cheaper than
// proximityJacobian, which also returns the witness point's rows.
template <typename Shape1, typename Shape2>
AlphaGradientResult alphaGradient(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                   const SocpOptions& opt = SocpOptions{}) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto P1 = problemMatrices(shape1, I4);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        shape1, shape2, g, combined.c, combined.G, combined.h, opt);

    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
    constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;

    AlphaGradientResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.grad = computeProximityGradient<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
            shape1, shape2, sol.x, sol.z, g);
    }
    return res;
}

// d[p;alpha]/dxi (4x6) via the implicit function theorem at the converged
// (x,s,z) (analytic_derivatives.hpp). Shape widths (v1,v2) and shape 1's ORT
// row count are read off the deduced problemMatrices types, so either shape
// may carry any decision-vector width.
template <typename Shape1, typename Shape2, int n_ort, int n_soc1, int n_soc2, int nx>
Eigen::Matrix<double, 4, 6> diffSocp(const Shape1& shape1, const Shape2& shape2, const DecisionVec<nx>& x,
                                      const StackVec<n_ort, n_soc1, n_soc2>& s,
                                      const StackVec<n_ort, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0) {
    const auto P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int v2 = nx - v1 + 4;
    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int n_ort2 = n_ort - n_ort1;
    const auto sens = computeSocpSensitivityAuto<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, x, s, z, g0);
    return sens.dx.template topRows<4>();
}

// As above but takes G directly -- proximityJacobian already built it for
// the forward solve.
template <typename Shape1, typename Shape2, int n_ort, int n_soc1, int n_soc2, int nx>
Eigen::Matrix<double, 4, 6> diffSocp(const Shape1& shape1, const Shape2& shape2, const DecisionVec<nx>& x,
                                      const StackVec<n_ort, n_soc1, n_soc2>& s,
                                      const StackVec<n_ort, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0,
                                      const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G) {
    const auto P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int v2 = nx - v1 + 4;
    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int n_ort2 = n_ort - n_ort1;
    const auto sens = computeSocpSensitivityAutoWithG<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, x, s, z, g0, G);
    return sens.dx.template topRows<4>();
}

struct ProximityJacobianResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero(); // rows [wx,wy,wz,alpha], cols xi=[w;v]
    int iters = 0;
    bool converged = false;
};

// The optional `warm` handle warm-starts the SOCP for temporally-continuous
// queries (warm_start.hpp); nullptr is exactly the cold path. This path's
// outputs tolerate the looser (s,z) a fast warm solve reaches, so it accepts
// at mu < pdip_tol -- unlike proximityContactJacobian, whose d(normal)/dxi
// needs a central-path point.
template <typename Shape1, typename Shape2>
ProximityJacobianResult proximityJacobian(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                           const SocpOptions& opt = SocpOptions{},
                                           ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    // Body 1's problem matrices are pose-independent (g = Identity). A warm
    // handle caches them across calls; otherwise they're rebuilt each call.
    using P1T = decltype(problemMatrices(shape1, I4));
    P1T P1_local;
    const P1T* P1 = &P1_local;
    if (warm != nullptr) {
        if (!warm->P1_valid) {
            warm->P1 = problemMatrices(shape1, I4);
            warm->P1_valid = true;
        }
        P1 = &warm->P1;
    } else {
        P1_local = problemMatrices(shape1, I4);
    }

    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(*P1, P2);

    SocpResult<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx> sol;
    bool warm_used = false;
    const double pose_move = (warm != nullptr && warm->valid) ? poseMoveMetric(warm->g_ref, g) : 0.0;
    if (warm != nullptr && warm->valid && pose_move <= warm->rho) {
        const auto wi = warmStartInit<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, warm->x, warm->z, pose_move);
        SocpOptions wopt = opt;
        wopt.max_iters = std::min(opt.max_iters, WarmStartConfig::kMaxIters);
        sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, wopt, &wi);
        if (sol.converged) {
            // Looser re-check (kResidTolMul * pdip_tol) on the final iterate:
            // keep the warm result or fall back to a cold solve.
            const double res_tol = WarmStartConfig::kResidTolMul * opt.pdip_tol;
            const double rx = (combined.G.transpose() * sol.z + combined.c).norm();
            const double rz = (sol.s + combined.G * sol.x - combined.h).norm();
            warm_used = (rx <= res_tol && rz <= res_tol);
        }
    }
    if (!warm_used) {
        sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, g, combined.c, combined.G, combined.h, opt);
        if (warm != nullptr && warm->valid)
            warm->rho = std::max(WarmStartConfig::kRhoShrink * warm->rho, WarmStartConfig::kRhoMin);
    } else if (warm != nullptr && sol.iters <= WarmStartConfig::kCheapIters) {
        warm->rho = std::min(WarmStartConfig::kRhoGrow * warm->rho, WarmStartConfig::kRhoMax);
    }

    ProximityJacobianResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.jacobian = diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, sol.x, sol.s, sol.z, g, combined.G);
    }

    if (warm != nullptr) {
        if (res.converged) {
            warm->g_ref = g;
            warm->x = sol.x;
            warm->s = sol.s;
            warm->z = sol.z;
            warm->mu = sol.mu;
            warm->valid = true;
        } else {
            warm->valid = false;
        }
    }
    return res;
}

} // namespace dcolpp::socp
